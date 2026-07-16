"""
app.py
--------------------
A simple GUI (Graphical User Interface) for our Speech Emotion Recognition
project, built with Tkinter (Python's built-in GUI toolkit - no extra
installation needed).

Features:
- "Select Audio" button - choose a .wav file from your computer
- "Predict Emotion" button - runs our trained model on it
- Displays the predicted emotion and confidence score
- Status messages to show what's happening

Run with: python app.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from feature_extraction import extract_mfcc


# -----------------------------------------------------------------------
# Load model and preprocessing tools ONCE, when the app starts.
# -----------------------------------------------------------------------
# WHY LOAD ONCE, NOT EVERY PREDICTION: loading a TensorFlow model takes
# a few seconds. If we reloaded it every time the user clicks "Predict",
# the app would feel slow and unresponsive. Loading once at startup means
# every prediction after that is nearly instant.
print("Loading model, please wait...")
model = load_model("emotion_model.keras")
scaler = joblib.load("scaler.pkl")
label_classes = np.load("label_classes.npy", allow_pickle=True)
print("Model loaded successfully!")


# -----------------------------------------------------------------------
# Global variable to remember which file the user selected.
# -----------------------------------------------------------------------
# WHY A GLOBAL VARIABLE: Tkinter's button clicks call separate functions
# with no direct connection between them. We need somewhere to "remember"
# the selected file path between the Select button click and the Predict
# button click - a global variable is the simplest beginner-friendly way.
selected_file_path = None


# -----------------------------------------------------------------------
# Function: runs when "Select Audio" button is clicked.
# -----------------------------------------------------------------------
def select_audio_file():
    """
    Purpose: Opens a file browser dialog so the user can pick a .wav file.
    Updates the global selected_file_path and the status label on screen.
    """
    global selected_file_path

    # filedialog.askopenfilename() opens the native Windows file browser.
    # filetypes restricts the dialog to only show .wav files, making it
    # harder for the user to accidentally pick the wrong file type.
    file_path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[("WAV files", "*.wav")]
    )

    # If the user clicks "Cancel", file_path will be an empty string.
    # We only update our state if they actually picked a real file.
    if file_path:
        selected_file_path = file_path

        # Show just the filename (not the full path) to keep the UI clean.
        filename_only = file_path.split("/")[-1].split("\\")[-1]
        status_label.config(text=f"Selected: {filename_only}", fg="black")

        # Clear any previous prediction result when a new file is chosen.
        result_label.config(text="")
        confidence_label.config(text="")


# -----------------------------------------------------------------------
# Function: runs when "Predict Emotion" button is clicked.
# -----------------------------------------------------------------------
def predict_emotion():
    """
    Purpose: Runs our trained model on the selected audio file and
    displays the predicted emotion + confidence on screen.
    """
    global selected_file_path

    # Safety check: make sure a file was actually selected first.
    if selected_file_path is None:
        messagebox.showwarning("No file selected", "Please select an audio file first.")
        return

    try:
        status_label.config(text="Processing... please wait", fg="blue")
        root.update()  # Forces the GUI to redraw immediately, so the
                        # "Processing..." message actually appears before
                        # the (brief) prediction work blocks the interface.

        # Same prediction logic as predict.py - extract features, scale,
        # predict, find highest probability.
        features = extract_mfcc(selected_file_path)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction_probabilities = model.predict(features_scaled, verbose=0)
        predicted_index = np.argmax(prediction_probabilities)
        predicted_emotion = label_classes[predicted_index]
        confidence = prediction_probabilities[0][predicted_index] * 100

        # Update the GUI labels with our results.
        result_label.config(text=f"Detected Emotion: {predicted_emotion.upper()}", fg="dark green")
        confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        status_label.config(text="Prediction complete!", fg="green")

    except Exception as error:
        # If anything goes wrong (corrupted file, wrong format, etc.),
        # show a friendly error message instead of crashing the whole app.
        messagebox.showerror("Error", f"Something went wrong:\n{error}")
        status_label.config(text="Error occurred - try another file", fg="red")


# -----------------------------------------------------------------------
# Build the GUI window and widgets.
# -----------------------------------------------------------------------
root = tk.Tk()
root.title("Speech Emotion Recognition")
root.geometry("450x350")  # Width x Height in pixels
root.configure(bg="#f0f0f0")

# Title text at the top.
title_label = tk.Label(
    root,
    text="Speech Emotion Recognition",
    font=("Arial", 16, "bold"),
    bg="#f0f0f0"
)
title_label.pack(pady=20)

# "Select Audio" button.
select_button = tk.Button(
    root,
    text="Select Audio File",
    font=("Arial", 11),
    command=select_audio_file,   # Runs select_audio_file() when clicked
    width=20,
    bg="#4a90d9",
    fg="white",
    cursor="hand2"
)
select_button.pack(pady=10)

# Status label - shows selected filename or processing messages.
status_label = tk.Label(
    root,
    text="No file selected",
    font=("Arial", 10),
    bg="#f0f0f0",
    fg="gray"
)
status_label.pack(pady=5)

# "Predict Emotion" button.
predict_button = tk.Button(
    root,
    text="Predict Emotion",
    font=("Arial", 11),
    command=predict_emotion,   # Runs predict_emotion() when clicked
    width=20,
    bg="#28a745",
    fg="white",
    cursor="hand2"
)
predict_button.pack(pady=10)

# Result label - shows the predicted emotion (large, bold text).
result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14, "bold"),
    bg="#f0f0f0"
)
result_label.pack(pady=15)

# Confidence label - shows the confidence percentage.
confidence_label = tk.Label(
    root,
    text="",
    font=("Arial", 11),
    bg="#f0f0f0"
)
confidence_label.pack(pady=5)

# -----------------------------------------------------------------------
# Start the GUI event loop.
# -----------------------------------------------------------------------
# root.mainloop() keeps the window open and listening for clicks/events
# until the user closes it. This line BLOCKS (nothing after it runs)
# until the window is closed - this is normal and expected for GUI apps.
root.mainloop()
