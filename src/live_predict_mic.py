import sys
import numpy as np
import librosa
import sounddevice as sd
import joblib
from keras.models import load_model

from config import (
    SAMPLE_RATE,
    MODEL_PATH,
    ENCODER_PATH,
    N_MFCC,
    MAX_LEN,
)

# ---------------------------------
# Record audio from microphone
# ---------------------------------
def record_from_mic(duration):
    print(f"🎙️ Recording from microphone for {duration} seconds...")
    print("🔊 Please make the bird sound now")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    return audio.flatten()


# ---------------------------------
# Live prediction
# ---------------------------------
def predict_from_mic(duration):
    model = load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    audio = record_from_mic(duration)

    # Remove silence
    audio, _ = librosa.effects.trim(audio)

    if len(audio) == 0:
        print("❌ No sound detected")
        return

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    ).T

    # Padding / Trimming
    if len(mfcc) < MAX_LEN:
        mfcc = np.pad(
            mfcc,
            ((0, MAX_LEN - len(mfcc)), (0, 0))
        )
    else:
        mfcc = mfcc[:MAX_LEN]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    prediction = model.predict(mfcc)
    idx = np.argmax(prediction)
    confidence = prediction[0][idx] * 100
    bird = encoder.inverse_transform([idx])[0]

    print(f"\n🦜 Predicted Bird Species: {bird}")
    print(f"📊 Confidence: {confidence:.2f}%")


# ---------------------------------
# CLI entry point
# ---------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide recording duration in seconds")
        sys.exit(1)

    try:
        duration = float(sys.argv[1])
        predict_from_mic(duration)
    except ValueError:
        print("❌ Duration must be a number")
