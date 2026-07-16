# Speech Emotion Recognition (SER)

A machine learning system that recognizes human emotions from speech audio, built with TensorFlow/Keras and trained on the RAVDESS dataset. Includes a full data pipeline, model training/evaluation, and a Tkinter desktop GUI for real-time predictions.

## Overview

This project classifies speech audio into one of 8 emotions: **angry, calm, disgust, fearful, happy, neutral, sad, surprised** — using MFCC (Mel-Frequency Cepstral Coefficient) audio features and a Dense Neural Network.

**Final model performance:** 65% validation accuracy (vs. 12.5% random-guess baseline for 8 classes).

## Features

- Automated MFCC feature extraction from raw audio
- Dense Neural Network classifier (TensorFlow/Keras)
- Full evaluation suite: classification report, confusion matrix
- Command-line prediction script for any WAV file
- Desktop GUI (Tkinter) for interactive predictions
- CPU-only training, optimized for machines without a GPU

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 1,440 speech audio files, 24 professional actors (12 male, 12 female)
- 8 emotions, WAV format, 48kHz sampling rate
- Official source: https://zenodo.org/record/1188976
- Download `Audio_Speech_Actors_01-24.zip` and extract into a `dataset/` folder (not included in this repo due to size)

## Project Structure

```
Emotion_Recognition/
├── dataset/                  # RAVDESS audio files (download separately)
├── outputs/                  # Generated graphs (confusion matrix, training curves)
├── feature_extraction.py     # MFCC extraction + label parsing
├── prepare_dataset.py        # Processes entire dataset into features/labels
├── train.py                  # Trains the DNN model
├── evaluate.py                # Generates classification report + confusion matrix
├── predict.py                 # Command-line prediction on a single audio file
├── app.py                     # Tkinter GUI application
├── requirements.txt
└── README.md
```

## Installation

1. **Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd Emotion_Recognition
   ```

2. **Create and activate a virtual environment** (Python 3.11 recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate       # Windows
   source venv/bin/activate    # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   Download `Audio_Speech_Actors_01-24.zip` from [Zenodo](https://zenodo.org/record/1188976) and extract into a `dataset/` folder in the project root, so you have `dataset/Actor_01/`, `dataset/Actor_02/`, etc.

## Usage

**1. Process the dataset into features**
```bash
python prepare_dataset.py
```

**2. Train the model**
```bash
python train.py
```

**3. Evaluate performance**
```bash
python evaluate.py
```

**4. Predict on a new audio file**
```bash
python predict.py path/to/audiofile.wav
```

**5. Launch the GUI**
```bash
python app.py
```

## Model Architecture

- Input: 40 MFCC coefficients (averaged across time per clip)
- Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(8, Softmax)
- Optimizer: Adam | Loss: Categorical Crossentropy
- Feature scaling via StandardScaler (critical for stable training)

## Results

| Metric | Score |
|---|---|
| Overall Accuracy | 65% |
| Best-performing emotions | Angry (75% F1), Fearful (72% F1) |
| Most-confused pair | Neutral ↔ Calm (acoustically similar) |

See `outputs/confusion_matrix.png` and `outputs/training_history.png` for detailed visualizations.

## Technical Notes & Experiments

An alternative CNN architecture using 2D MFCC features (preserving temporal information) was also explored. Across multiple configurations, the CNN did not outperform the DNN baseline on this dataset, likely due to the limited training set size (1,152 samples) relative to CNN's larger parameter space causing overfitting even after regularization. This reinforced that model complexity should be matched to available data — the simpler, well-regularized DNN generalized better here.

**Known limitation:** the model was trained exclusively on RAVDESS's professional, scripted, studio-recorded speech. It shows reduced accuracy on natural/unscripted speech from different speakers, accents, and recording conditions — a common generalization challenge in speech emotion recognition, and an area for future improvement (e.g. training on more diverse, naturalistic datasets).

## Technologies Used

Python, TensorFlow/Keras, librosa, scikit-learn, NumPy, Matplotlib, Tkinter

## Future Improvements

- Train on additional/more diverse datasets (multiple languages, accents, natural speech)
- Experiment with LSTM/attention-based architectures for temporal modeling
- Data augmentation (pitch shift, noise injection) to improve generalization
- Real-time microphone input support in the GUI

## Author

Quratulain (Anie) — BS Artificial Intelligence, The Islamia University of Bahawalpur

## License

This project is for educational purposes. RAVDESS dataset is licensed under CC BY-NA-SC 4.0 — see the [official dataset page](https://zenodo.org/record/1188976) for details.
