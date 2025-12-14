import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from feature_extraction import extract_mel_spectrogram
from config import ENCODER_PATH

SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".ogg")

def load_dataset(dataset_path):
    X, y = [], []

    print(f"📂 Reading dataset from: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    for bird_name in os.listdir(dataset_path):
        bird_folder = os.path.join(dataset_path, bird_name)

        if not os.path.isdir(bird_folder):
            continue

        print(f"🦜 Loading bird class: '{bird_name}'")

        files_found = 0

        for file in os.listdir(bird_folder):
            if file.lower().endswith(SUPPORTED_FORMATS):
                files_found += 1
                file_path = os.path.join(bird_folder, file)

                mfcc = extract_mel_spectrogram(file_path)
                if mfcc is not None:
                    X.append(mfcc)
                    y.append(bird_name)

        if files_found == 0:
            print(f"⚠️ No audio files in {bird_folder}")

    if len(X) == 0:
        raise RuntimeError(
            "❌ No audio files were loaded.\n"
            "✔ Check file extensions\n"
            "✔ Check dataset path\n"
            "✔ Ensure files are not empty"
        )

    X = np.array(X, dtype="float32")[..., np.newaxis]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_encoded = to_categorical(y_encoded, num_classes=len(set(y)))

    joblib.dump(encoder, ENCODER_PATH)

    print(f"✅ Loaded {len(X)} samples")
    print(f"✅ Classes: {list(encoder.classes_)}")

    return X, y_encoded, encoder
