"""
train.py
--------------------
This is our FINAL model training script:
1. Loads features.npy and labels.npy
2. Converts text labels into one-hot encoded numbers
3. Splits data into training (80%) and testing (20%) sets
4. SCALES features (critical fix - prevents unstable training)
5. Builds and trains a Dense Neural Network
6. Saves the trained model + graphs of accuracy/loss

Run this AFTER prepare_dataset.py has created features.npy and labels.npy.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# -----------------------------------------------------------------------
# STEP 1: Load our previously saved data.
# -----------------------------------------------------------------------
print("Loading saved features and labels...")
X = np.load("features.npy")
y = np.load("labels.npy", allow_pickle=True)

print(f"Loaded X shape: {X.shape}")
print(f"Loaded y shape: {y.shape}")


# -----------------------------------------------------------------------
# STEP 2: Encode labels (text -> integers -> one-hot vectors)
# -----------------------------------------------------------------------
print("\nEncoding labels...")

label_encoder = LabelEncoder()
y_integers = label_encoder.fit_transform(y)
y_encoded = to_categorical(y_integers, num_classes=8)

print("Emotion classes found:", label_encoder.classes_)
np.save("label_classes.npy", label_encoder.classes_)


# -----------------------------------------------------------------------
# STEP 3: Split into training and testing sets (80% / 20%)
# -----------------------------------------------------------------------
print("\nSplitting data into train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# -----------------------------------------------------------------------
# STEP 3b: Scale features (CRITICAL - fixes unstable training).
# -----------------------------------------------------------------------
# WHY THIS MATTERS: MFCC values have wildly different scales (e.g. -646
# vs 4.9). Without scaling, training is unstable and accuracy gets stuck
# near random-guessing level. This was diagnosed and fixed early on -
# always scale your features before feeding them to a neural network.
print("\nScaling features...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")


# -----------------------------------------------------------------------
# STEP 4: Build the Dense Neural Network model.
# -----------------------------------------------------------------------
print("\nBuilding the model...")

model = Sequential([
    Dense(128, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# -----------------------------------------------------------------------
# STEP 5: Train the model.
# -----------------------------------------------------------------------
print("\nStarting training...\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# -----------------------------------------------------------------------
# STEP 6: Save the trained model.
# -----------------------------------------------------------------------
model.save("emotion_model.keras")
print("\nModel saved as emotion_model.keras")


# -----------------------------------------------------------------------
# STEP 7: Plot training graphs (accuracy and loss over epochs).
# -----------------------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_history.png")
print("Training graphs saved to outputs/training_history.png")
print("\nOpen the outputs folder and view training_history.png to see the graphs.")
# Note: plt.show() intentionally omitted - it opens a popup window and
# PAUSES the script until closed, which looks like a frozen terminal.
