import os
import librosa
import numpy as np

INPUT_FOLDER = "archive/cleared_voices/M16"

# MFCC parameters
N_MFCC = 40
MAX_LEN = 120

data = []
file_names = []

def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        mfcc = mfcc.T  # (time_steps, features)

        # Normalize (IMPORTANT)
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

        # Pad / Truncate
        if len(mfcc) < MAX_LEN:
            pad_width = MAX_LEN - len(mfcc)
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:MAX_LEN, :]

        return mfcc

    except Exception as e:
        print(f"❌ Error: {file_path} → {e}")
        return None


# 🔁 Process all files
for file in os.listdir(INPUT_FOLDER):
    if file.lower().endswith(".wav"):
        file_path = os.path.join(INPUT_FOLDER, file)

        mfcc = extract_mfcc(file_path)

        if mfcc is not None:
            data.append(mfcc)
            file_names.append(file)

            print(f"✅ Processed: {file}")

# Convert to numpy
X = np.array(data)

print("\n🎯 FINAL OUTPUT")
print("MFCC Shape:", X.shape)