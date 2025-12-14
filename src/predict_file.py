import sys
import joblib
import numpy as np
from keras.models import load_model
from feature_extraction import extract_mel_spectrogram
from config import MODEL_PATH, ENCODER_PATH

def predict_audio(file_path):
    model = load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    mfcc = extract_mel_spectrogram(file_path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    prediction = model.predict(mfcc)
    predicted_label = encoder.inverse_transform([prediction.argmax()])[0]

    print("🦜 Predicted Bird Species:", predicted_label)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide an audio file path")
        print("👉 Example: python src/predict_file.py test.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    predict_audio(audio_path)
