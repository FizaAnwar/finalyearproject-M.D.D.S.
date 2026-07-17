"""
stroke_severity_pipeline.py

Integrates two INDEPENDENTLY TRAINED models into a single prediction pipeline:

  1. StrokeMRIPredictor        -> ViT-B/16 image classifier (stroke vs non-stroke),
                                   trained on the Kaggle "brain-stroke-mri-images" dataset.
  2. DysarthriaSeverityPredictor -> Wav2Vec2 features + Keras MLP (Low/Medium/High severity),
                                   trained on the UA-Speech dysarthria (cerebral palsy) corpus.

IMPORTANT / HONEST LIMITATION (read before using in your FYP report):
-----------------------------------------------------------------------
These two models were trained on two DIFFERENT, UNRELATED patient cohorts
(no shared subject IDs between the MRI dataset and the speech dataset).
There is therefore no ground-truth data pairing an MRI scan with a speech
sample from the same patient. Because of that:

  - We CANNOT train a real stacking/meta-classifier here (that requires
    paired [MRI_output, speech_output] -> true_combined_label examples).
  - CombinedStrokeSystem below is a RULE-BASED ORCHESTRATION layer, not a
    learned fusion model. It runs both models independently and merges
    their outputs into one structured report.
  - Also note the severity labels come from a cerebral-palsy dysarthria
    corpus, not a stroke-specific corpus -- treat "severity" as dysarthria
    severity, and be explicit about this scope limitation in your report.

If you later obtain PAIRED data (same patient, both modalities, with a
true combined outcome label), see `train_real_meta_classifier()` at the
bottom for how to plug in an actual learned meta-classifier.
"""

import argparse
import json
import os
import sys
import warnings


# =========================================================
# EDIT THESE PATHS FOR YOUR MACHINE
# =========================================================
IMAGE_PATH = "dataset/data/test/stroke/Amina DWI-18.jpg_Ischemic_23.png"
AUDIO_PATH = "archive/cleared_voices/M16/M16_B1_C1_M3.wav"

VIT_CHECKPOINT_PATH = "vit_stroke_model.pth"
VIT_CLASS_NAMES_PATH = "vit_class_names.json"
SEVERITY_MODEL_PATH = "severity_classifier.h5"
SEVERITY_MEAN_PATH = "severity_feature_mean.npy"
SEVERITY_STD_PATH = "severity_feature_std.npy"
OUTPUT_REPORT_PATH = "combined_report.json"

# If True, severity is only computed when the MRI model predicts "stroke"
ALWAYS_RUN_SEVERITY = False

import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image

import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

import tensorflow as tf


# =========================================================
# STAGE 1: MRI STROKE DETECTOR (ViT-B/16)
# =========================================================
class StrokeMRIPredictor:
    def __init__(self, checkpoint_path, class_names_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"MRI model checkpoint not found: {checkpoint_path}\n"
                "Train and save it first (see vit_train.py, add "
                "torch.save(model.state_dict(), 'vit_stroke_model.pth') after training)."
            )
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(
                f"Class names file not found: {class_names_path}\n"
                "Save it with json.dump(train_dataset.classes, open('vit_class_names.json','w'))."
            )

        with open(class_names_path) as f:
            self.class_names = json.load(f)

        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=len(self.class_names)
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        return {
            "prediction": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
            },
        }


# =========================================================
# STAGE 2: DYSARTHRIA SEVERITY PREDICTOR (Wav2Vec2 + MLP)
# =========================================================
class DysarthriaSeverityPredictor:
    CLASS_NAMES = ["Low", "Medium", "High"]

    def __init__(self, classifier_path, mean_path, std_path,
                 wav2vec_name="facebook/wav2vec2-base-960h"):
        for p in (classifier_path, mean_path, std_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Required file not found: {p}\n"
                    "Train and save the severity classifier first "
                    "(model.save(...), np.save(mean...), np.save(std...))."
                )

        print("Loading Wav2Vec2 (feature extractor)...")
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_name)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_name)
        self.wav2vec_model.eval()

        self.classifier = tf.keras.models.load_model(classifier_path)
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

    def _extract_features(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.wav2vec_model(inputs.input_values)
        features = torch.mean(outputs.last_hidden_state, dim=1)
        return features.squeeze().cpu().numpy()

    def predict(self, audio_path):
        feats = self._extract_features(audio_path)
        feats_norm = (feats - self.mean) / (self.std + 1e-6)
        feats_norm = feats_norm.reshape(1, -1)

        probs = self.classifier.predict(feats_norm, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        return {
            "prediction": self.CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                self.CLASS_NAMES[i]: float(probs[i]) for i in range(len(self.CLASS_NAMES))
            },
        }


# =========================================================
# FUSION LAYER (rule-based orchestration, NOT a trained meta-classifier)
# =========================================================
class CombinedStrokeSystem:
    """
    Orchestrates the two independent predictors and merges their outputs
    into a single report. This is deliberately NOT presented as a trained
    joint model -- see module docstring for why.
    """

    def __init__(self, mri_predictor: StrokeMRIPredictor,
                 severity_predictor: DysarthriaSeverityPredictor,
                 stroke_class_label: str = "stroke",
                 severity_only_if_stroke: bool = True):
        self.mri_predictor = mri_predictor
        self.severity_predictor = severity_predictor
        self.stroke_class_label = stroke_class_label
        self.severity_only_if_stroke = severity_only_if_stroke

    def predict(self, image_path=None, audio_path=None):
        if image_path is None and audio_path is None:
            raise ValueError("Provide at least one of image_path or audio_path.")

        report = {"inputs": {"image_path": image_path, "audio_path": audio_path}}

        mri_result = None
        if image_path:
            mri_result = self.mri_predictor.predict(image_path)
            report["mri_prediction"] = mri_result

        should_run_severity = audio_path is not None and (
            not self.severity_only_if_stroke
            or mri_result is None
            or mri_result["prediction"].lower() == self.stroke_class_label.lower()
        )

        if audio_path and not should_run_severity:
            report["speech_severity"] = {
                "skipped": True,
                "reason": (
                    "MRI prediction indicates non-stroke; severity scoring skipped by "
                    "default. Pass severity_only_if_stroke=False to always run it."
                ),
            }
        elif audio_path:
            report["speech_severity"] = self.severity_predictor.predict(audio_path)

        report["disclaimer"] = (
            "MRI and speech models were trained on separate, unpaired patient "
            "cohorts (brain-stroke-mri-images vs UA-Speech dysarthria corpus). "
            "This is a rule-based orchestration of two independent predictions, "
            "not a statistically trained joint/meta-classifier. Severity reflects "
            "dysarthria severity from a cerebral-palsy speech corpus, used here as "
            "a proxy -- it has not been validated on stroke patients specifically. "
            "For research/demo purposes only; not a clinical diagnostic tool."
        )
        return report


# =========================================================
# FUTURE WORK: real meta-classifier, once paired data exists
# =========================================================
def train_real_meta_classifier(paired_feature_matrix, paired_labels):
    """
    Only meaningful once you have PAIRED data: for each patient you'd need
    [mri_stroke_prob, mri_nonstroke_prob, severity_low_prob,
     severity_medium_prob, severity_high_prob] as features (5-dim vector)
    and a true combined outcome label to fit against.

    paired_feature_matrix: shape (n_patients, 5)
    paired_labels: shape (n_patients,)

    Example (sketch, not run automatically):

        from sklearn.linear_model import LogisticRegression
        meta_clf = LogisticRegression(max_iter=1000)
        meta_clf.fit(paired_feature_matrix, paired_labels)
        return meta_clf
    """
    raise NotImplementedError(
        "No paired dataset available yet. This is a placeholder for future "
        "work once you collect subjects with both an MRI scan and a speech "
        "sample plus a true combined ground-truth label."
    )


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Combined stroke + severity prediction (demo pipeline)")
    parser.add_argument("--image", type=str, default=IMAGE_PATH, help="Path to brain MRI image")
    parser.add_argument("--audio", type=str, default=AUDIO_PATH, help="Path to speech .wav file")
    parser.add_argument("--vit_ckpt", type=str, default=VIT_CHECKPOINT_PATH)
    parser.add_argument("--vit_classes", type=str, default=VIT_CLASS_NAMES_PATH)
    parser.add_argument("--severity_model", type=str, default=SEVERITY_MODEL_PATH)
    parser.add_argument("--severity_mean", type=str, default=SEVERITY_MEAN_PATH)
    parser.add_argument("--severity_std", type=str, default=SEVERITY_STD_PATH)
    parser.add_argument("--always_run_severity", action="store_true", default=ALWAYS_RUN_SEVERITY,
                         help="Run severity model even if MRI predicts non-stroke")
    parser.add_argument("--out", type=str, default=OUTPUT_REPORT_PATH)
    args = parser.parse_args()

    if not args.image and not args.audio:
        print("Provide at least --image or --audio. Use --help for options.")
        sys.exit(1)

    mri_predictor = None
    severity_predictor = None

    if args.image:
        mri_predictor = StrokeMRIPredictor(args.vit_ckpt, args.vit_classes)
    if args.audio:
        severity_predictor = DysarthriaSeverityPredictor(
            args.severity_model, args.severity_mean, args.severity_std
        )

    # If only one modality is provided, just run that model directly.
    if mri_predictor and not severity_predictor:
        report = {"mri_prediction": mri_predictor.predict(args.image)}
    elif severity_predictor and not mri_predictor:
        report = {"speech_severity": severity_predictor.predict(args.audio)}
    else:
        system = CombinedStrokeSystem(
            mri_predictor, severity_predictor,
            severity_only_if_stroke=not args.always_run_severity,
        )
        report = system.predict(image_path=args.image, audio_path=args.audio)

    print("\n" + json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()