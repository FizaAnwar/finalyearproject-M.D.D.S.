import os
import librosa
import numpy as np
INPUT_FOLDER = "archive/cleared_voices/M16"

# -----------------------------
# PARAMETERS
# -----------------------------
N_MFCC = 40
MAX_LEN = 120

SEQ_LEN = 20       # length of each sequence
STEP = 10          # overlap step

data = []
file_names = []

# -----------------------------
# MFCC EXTRACTION
# -----------------------------
def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        mfcc = mfcc.T  # (time_steps, features)

        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

        # Pad / Truncate
        if len(mfcc) < MAX_LEN:
            pad_width = MAX_LEN - len(mfcc)
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:MAX_LEN, :]

        return mfcc

    except Exception as e:
        print(f"Error: {file_path} → {e}")
        return None


# -----------------------------
# SEQUENCE CREATION FUNCTION
# -----------------------------
def create_sequences(mfcc):
    sequences = []

    for start in range(0, len(mfcc) - SEQ_LEN + 1, STEP):
        seq = mfcc[start:start + SEQ_LEN]
        sequences.append(seq)

    return sequences


# -----------------------------
# PROCESS ALL FILES
# -----------------------------
for file in os.listdir(INPUT_FOLDER):

    if file.lower().endswith(".wav"):
        file_path = os.path.join(INPUT_FOLDER, file)

        mfcc = extract_mfcc(file_path)

        if mfcc is not None:
            seqs = create_sequences(mfcc)

            for seq in seqs:
                data.append(seq)
                file_names.append(file)

            print(f"Processed: {file} | Sequences: {len(seqs)}")

# -----------------------------
# FINAL OUTPUT
# -----------------------------
X = np.array(data)

print("\nFINAL OUTPUT")
print("Shape:", X.shape)
print("Format: (samples, sequence_length, features)")