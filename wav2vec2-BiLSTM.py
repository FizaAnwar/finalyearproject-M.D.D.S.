import os
import numpy as np
import librosa
import torch

from transformers import Wav2Vec2Processor, Wav2Vec2Model

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

# =========================
# CONFIG
# =========================

DATASET_PATH = "archive/cleared_voices"

# Remove Very Low class (M2, M3)
# Train only on Low, Medium, High

label_map = {
    "M4": 0,  # Low
    "M5": 0,  # Low

    "M6": 1,  # Medium
    "M7": 1,  # Medium

    "M8": 2   # High
}

CLASS_NAMES = ["Low", "Medium", "High"]
NUM_CLASSES = 3

# =========================
# LOAD WAV2VEC2
# =========================

print("Loading Wav2Vec2...")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

wav2vec_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

wav2vec_model.eval()

# =========================
# LABEL EXTRACTION
# =========================

def extract_label(file_name):
    """
    Example:
    M07_B3_CW88_M7.wav -> M7 -> class
    """

    label = file_name.split("_")[-1].replace(".wav", "").upper()

    return label_map.get(label, None)

# =========================
# FEATURE EXTRACTION
# =========================

def extract_wav2vec_features(file_path):

    audio, sr = librosa.load(file_path, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = wav2vec_model(inputs.input_values)

    # Mean Pooling
    features = torch.mean(
        outputs.last_hidden_state,
        dim=1
    )

    return features.squeeze().cpu().numpy()

# =========================
# LOAD DATASET
# =========================

X = []
y = []

print("Extracting features...")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if not file.endswith(".wav"):
            continue

        label = extract_label(file)

        # Skip M2 and M3 automatically
        if label is None:
            continue

        file_path = os.path.join(root, file)

        try:
            features = extract_wav2vec_features(file_path)

            X.append(features)
            y.append(label)

        except Exception as e:
            print(f"Error processing {file}: {e}")

X = np.array(X)
y = np.array(y)

print("\nDataset Loaded")
print("Total Samples:", len(X))
print("Feature Shape:", X.shape)

if len(X) == 0:
    raise ValueError("No valid samples found!")

print("\nClass Distribution:")
print(Counter(y))

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# NORMALIZATION
# =========================

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / (std + 1e-6)
X_test = (X_test - mean) / (std + 1e-6)

# =========================
# ONE HOT ENCODING
# =========================

y_train_cat = to_categorical(
    y_train,
    num_classes=NUM_CLASSES
)

y_test_cat = to_categorical(
    y_test,
    num_classes=NUM_CLASSES
)

# =========================
# CLASS WEIGHTS
# =========================

class_weights = dict(
    zip(
        np.unique(y_train),
        compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
    )
)

print("\nClass Weights:")
print(class_weights)

# =========================
# CLASSIFIER
# =========================

model = Sequential([
    Input(shape=(X.shape[1],)),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),

    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# =========================
# TRAINING
# =========================

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=30,
    batch_size=16,
    class_weight=class_weights,
    verbose=1
)

# =========================
# EVALUATION
# =========================

loss, accuracy = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print("\nTest Accuracy:", accuracy)

# =========================
# PREDICTIONS
# =========================

y_pred_probs = model.predict(X_test)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

model.save("severity_classifier.h5")
np.save("severity_feature_mean.npy", mean)
np.save("severity_feature_std.npy", std)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("Wav2Vec2 Severity Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    )
)

# =========================
# TRAINING CURVES
# =========================

plt.figure(figsize=(10, 4))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()