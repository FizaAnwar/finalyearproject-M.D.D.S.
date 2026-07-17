import os
import librosa
import numpy as np
import soundfile as sf

INPUT_FOLDER = "archive/noisereduced-uaspeech/M16"
OUTPUT_FOLDER = "archive/cleared_voices/M16"

TARGET_SR = 16000

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_audio(file_path, output_path):
    try:
        # 1. Load & Resample (mono by default)
        audio, sr = librosa.load(file_path, sr=TARGET_SR)

        # 2. Trim Silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

        # 3. Normalize Amplitude
        max_val = np.max(np.abs(audio_trimmed))
        if max_val > 0:
            audio_normalized = audio_trimmed / max_val
        else:
            audio_normalized = audio_trimmed

        # 4. Save Clean Audio
        sf.write(output_path, audio_normalized, TARGET_SR)

        print(f"✅ Processed: {file_path}")

    except Exception as e:
        print(f"❌ Error: {file_path} → {e}")


# Process all files
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".wav"):
        input_path = os.path.join(INPUT_FOLDER, file)
        output_path = os.path.join(OUTPUT_FOLDER, file)

        preprocess_audio(input_path, output_path)

print("\n🎯 All audio files cleaned and saved!")