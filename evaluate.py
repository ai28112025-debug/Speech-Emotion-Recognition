"""
evaluate.py
--------------------
Evaluates our trained DNN model:
1. Loads the saved model, scaler, and test data
2. Generates predictions on the test set
3. Prints a classification report (precision, recall, F1-score per emotion)
4. Plots a confusion matrix

Run this AFTER train.py has completed and saved emotion_model.keras.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


# -----------------------------------------------------------------------
# STEP 1: Reload the same data and apply the SAME split/scaling as training.
# -----------------------------------------------------------------------
print("Loading data...")
X = np.load("features.npy")
y = np.load("labels.npy", allow_pickle=True)

label_encoder = LabelEncoder()
y_integers = label_encoder.fit_transform(y)
y_encoded = to_categorical(y_integers, num_classes=8)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

scaler = joblib.load("scaler.pkl")
X_test = scaler.transform(X_test)


# -----------------------------------------------------------------------
# STEP 2: Load the trained model and generate predictions.
# -----------------------------------------------------------------------
print("Loading trained model...")
model = load_model("emotion_model.keras")

y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)
y_true = np.argmax(y_test, axis=1)


# -----------------------------------------------------------------------
# STEP 3: Print classification report.
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)

report = classification_report(
    y_true, y_pred,
    target_names=label_encoder.classes_
)
print(report)


# -----------------------------------------------------------------------
# STEP 4: Generate and plot the confusion matrix.
# -----------------------------------------------------------------------
print("Generating confusion matrix...")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Emotion Recognition")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
print("Confusion matrix saved to outputs/confusion_matrix.png")
