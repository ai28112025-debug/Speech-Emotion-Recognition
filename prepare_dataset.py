"""
prepare_dataset.py
--------------------
Processes the ENTIRE RAVDESS dataset (all 1,440 audio files) using the
AVERAGED MFCC method (extract_mfcc) - this feeds our final DNN model.

Run this ONCE. Produces:
- features.npy  (X data - MFCC numbers for every clip, shape 1440x40)
- labels.npy    (y data - emotion name for every clip)
"""

import os
import numpy as np
from feature_extraction import extract_mfcc, get_emotion_label


DATASET_PATH = "dataset"

X = []
y = []

actor_folders = sorted(os.listdir(DATASET_PATH))
total_files_processed = 0

for actor_folder in actor_folders:
    actor_path = os.path.join(DATASET_PATH, actor_folder)

    if not os.path.isdir(actor_path):
        continue

    wav_files = os.listdir(actor_path)

    for wav_file in wav_files:
        if not wav_file.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, wav_file)

        try:
            features = extract_mfcc(file_path)
            label = get_emotion_label(file_path)

            X.append(features)
            y.append(label)

            total_files_processed += 1

            if total_files_processed % 100 == 0:
                print(f"Processed {total_files_processed} files so far...")

        except Exception as error:
            print(f"Skipped {file_path} due to error: {error}")


X = np.array(X)
y = np.array(y)

print(f"\nFinished processing {total_files_processed} files.")
print(f"X shape (features): {X.shape}")   # Expect: (1440, 40)
print(f"y shape (labels): {y.shape}")     # Expect: (1440,)

np.save("features.npy", X)
np.save("labels.npy", y)

print("\nSaved features.npy and labels.npy successfully!")
