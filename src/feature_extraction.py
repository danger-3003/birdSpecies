import librosa
import numpy as np
from config import SAMPLE_RATE, MAX_LEN

N_MELS = 128

def extract_mel_spectrogram(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        fmax=8000
    )

    mel_db = librosa.power_to_db(mel, ref=np.max).T

    if len(mel_db) < MAX_LEN:
        mel_db = np.pad(mel_db, ((0, MAX_LEN - len(mel_db)), (0, 0)))
    else:
        mel_db = mel_db[:MAX_LEN]

    return mel_db
