# Bird Species Recognition using Audio

# Dataset folder strucutre
root/data/raw/<bird_sound_folder>/<sounds>(.mp3, .wav)

## Install dependencies
uv sync
uv add -r requirements.txt

## Train model
python src/train.py

## Predict from audio file
python src/predict_file.py "data/raw/Andean Guan_sound/Andean Guan2.mp3"

## Live prediction
python src/live_predict.py "data/raw/Andean Guan_sound/Andean Guan2.mp3"

## Live predection with microphone
python src/live_predict_mic.py