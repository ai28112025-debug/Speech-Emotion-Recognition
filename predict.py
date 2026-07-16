"""
predict.py
--------------------
Takes a NEW audio file and predicts its emotion.

Usage (from terminal):
    python predict.py path/to/your/audiofile.wav
"""

import sys
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from feature_extraction import extract_mfcc


def predict_emotion(file_path):
    """
    Purpose: Predict the emotion of a single audio file.
    Input: file_path (string) - path to a .wav file
    Output: prints the predicted emotion and confidence percentage.
    """
    print(f"Loading model and preprocessing tools...")

    model = load_model("emotion_model.keras")
    scaler = joblib.load("scaler.pkl")
    label_classes = np.load("label_classes.npy", allow_pickle=True)

    print(f"Processing audio file: {file_path}")

    features = extract_mfcc(file_path)
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction_probabilities = model.predict(features_scaled, verbose=0)
    predicted_index = np.argmax(prediction_probabilities)
    predicted_emotion = label_classes[predicted_index]
    confidence = prediction_probabilities[0][predicted_index] * 100

    print("\n" + "="*40)
    print(f"Detected Emotion: {predicted_emotion.upper()}")
    print(f"Confidence: {confidence:.1f}%")
    print("="*40)

    print("\nFull breakdown:")
    sorted_indices = np.argsort(prediction_probabilities[0])[::-1]
    for idx in sorted_indices:
        emotion_name = label_classes[idx]
        prob = prediction_probabilities[0][idx] * 100
        print(f"  {emotion_name:12s}: {prob:5.1f}%")

    return predicted_emotion, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/audiofile.wav")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    predict_emotion(audio_file_path)
