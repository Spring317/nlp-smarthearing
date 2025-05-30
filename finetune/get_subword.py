import os
import torch
import torchaudio
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === SETTINGS ===
AUDIO_PATH = "your_audio.wav"  # Path to input WAV file
OUTPUT_DIR = "kws_segments"    # Folder to store subword WAV clips
MODEL_NAME = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load model and processor ===
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

# === Load and preprocess audio ===
waveform, sr = torchaudio.load(AUDIO_PATH)
if sr != 16000:
    waveform_np = librosa.resample(waveform.numpy()[0], orig_sr=sr, target_sr=16000)
    waveform = torch.tensor(waveform_np).unsqueeze(0)
    sr = 16000

# Tokenize input
inputs = processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# === Decode tokens and track segments ===
pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
tokens = processor.tokenizer.convert_ids_to_tokens(pred_ids)

logit_len = logits.shape[1]
audio_len_sec = waveform.shape[1] / sr
frame_duration = audio_len_sec / logit_len

segments = []
last_token = None
for i, token_id in enumerate(pred_ids):
    token = processor.tokenizer.convert_ids_to_tokens([token_id])[0]
    
    if token == last_token or token in [processor.tokenizer.pad_token, processor.tokenizer.blank_token]:
        continue

    end_time = i * frame_duration
    segments.append((token, end_time))
    last_token = token

# === Export segments to .wav files ===
segment_start = 0.0
for idx, (token, end_time) in enumerate(segments):
    start_sample = int(segment_start * sr)
    end_sample = int(end_time * sr)
    segment_waveform = waveform[:, start_sample:end_sample]

    # Sanitize file name
    token_clean = token.replace("▁", "").replace("/", "_").strip("_")
    if token_clean == "":
        token_clean = f"unk_{idx}"

    file_path = os.path.join(OUTPUT_DIR, f"{idx:03d}_{token_clean}.wav")
    torchaudio.save(file_path, segment_waveform, sample_rate=sr)

    print(f"Saved: {file_path}  | Duration: {end_time - segment_start:.2f}s | Token: {token}")

    segment_start = end_time
