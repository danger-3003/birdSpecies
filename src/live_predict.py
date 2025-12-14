import sys
import numpy as np
import librosa
import joblib
from keras.models import load_model

from config import (
    SAMPLE_RATE,
    MODEL_PATH,
    ENCODER_PATH,
    N_MFCC,
    MAX_LEN,
)

# -----------------------------
# Predict from audio file
# -----------------------------
def live_predict_from_file(file_path):
    model = load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # Load audio (supports .wav, .mp3)
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    ).T

    # Pad / truncate
    if len(mfcc) < MAX_LEN:
        mfcc = np.pad(
            mfcc,
            ((0, MAX_LEN - len(mfcc)), (0, 0))
        )
    else:
        mfcc = mfcc[:MAX_LEN]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    # Predict
    prediction = model.predict(mfcc)
    predicted_label = encoder.inverse_transform(
        [np.argmax(prediction)]
    )[0]

    confidence = np.max(prediction) * 100

    print(f"🦜 Predicted Bird Species: {predicted_label}")
    print(f"📊 Confidence: {confidence:.2f}%")

# -----------------------------
# CLI entry point
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide an audio file path")
        print("👉 Usage:")
        print('   python src/live_predict.py "path/to/audio.wav"')
        sys.exit(1)

    audio_path = sys.argv[1]
    live_predict_from_file(audio_path)
