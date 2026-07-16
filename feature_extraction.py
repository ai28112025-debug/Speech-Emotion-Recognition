"""
feature_extraction.py
----------------------
This file has two jobs:
1. Read an audio (.wav) file and turn it into MFCC numbers (features).
2. Read a RAVDESS filename and figure out which emotion it represents (label).

Think of this file as our "translator" - it converts raw audio files
into a language (numbers) that our neural network can learn from.
"""

import librosa
import numpy as np
import os


# -----------------------------------------------------------------------
# STEP 1: Map RAVDESS emotion codes (from the filename) to actual emotion names.
# -----------------------------------------------------------------------
# Example filename: 03-01-06-01-02-01-12.wav -> code "06" -> Fearful
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def get_emotion_label(file_path):
    """
    Purpose: Extract the emotion label from a RAVDESS filename.
    Input: file_path (string) - e.g. "dataset/Actor_01/03-01-06-01-02-01-01.wav"
    Output: emotion (string) - e.g. "fearful"
    """
    filename = os.path.basename(file_path)
    parts = filename.split("-")
    emotion_code = parts[2]
    emotion = EMOTION_MAP[emotion_code]
    return emotion


# -----------------------------------------------------------------------
# STEP 2: Extract MFCC features (AVERAGED version - used by our final DNN model)
# -----------------------------------------------------------------------
def extract_mfcc(file_path, n_mfcc=40):
    """
    Purpose: Convert a WAV audio file into a fixed-size list of numbers (MFCCs)
             that represent the sound's characteristics. This is the version
             used by our FINAL model (train.py / DNN).

    Input:
        file_path (string) - path to the .wav file
        n_mfcc (int) - how many MFCC coefficients to extract per time-frame

    Output:
        mfcc_mean (numpy array) - shape (40,), a single vector summarizing
                                   the entire audio clip (averaged across time)
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    # Average across time-frames - collapses (40, time_frames) into (40,).
    # Every clip becomes the same fixed size regardless of length.
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


# -----------------------------------------------------------------------
# STEP 2b: Extract 2D MFCC features (kept for reference - used in our CNN
# experiment, documented in the project but NOT used by the final model)
# -----------------------------------------------------------------------
MAX_PAD_LENGTH = 130


def extract_mfcc_2d(file_path, n_mfcc=40, max_pad_length=MAX_PAD_LENGTH):
    """
    Purpose: Convert a WAV file into a 2D MFCC grid, keeping time information.
    NOTE: This was used in our CNN experiment. The final project uses
    extract_mfcc() (above) with the DNN model instead, since it generalized
    better on our dataset size. Kept here for documentation/reference.
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    current_length = mfcc.shape[1]

    if current_length < max_pad_length:
        pad_amount = max_pad_length - current_length
        mfcc_padded = np.pad(mfcc, pad_width=((0, 0), (0, pad_amount)), mode='constant')
    else:
        mfcc_padded = mfcc[:, :max_pad_length]

    return mfcc_padded


# -----------------------------------------------------------------------
# STEP 3: Quick test - only runs if you execute this file directly.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    test_file = "dataset/Actor_01/03-01-06-01-02-01-01.wav"

    print(f"Testing file: {test_file}")

    label = get_emotion_label(test_file)
    print(f"Detected emotion label: {label}")

    features = extract_mfcc(test_file)
    print(f"MFCC feature vector shape: {features.shape}")
    print(f"First 5 MFCC values: {features[:5]}")
