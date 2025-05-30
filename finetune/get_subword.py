import os
import torch
import torchaudio
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset

# === SETTINGS ===
COMMON_VOICE_LANG = "vi"  # Language code for Vietnamese
COMMON_VOICE_SPLIT = "test"  # Dataset split: "train", "test", "validation"
SAMPLE_INDEX = 0  # Index of the sample to process
OUTPUT_DIR = "kws_segments"  # Folder to store subword WAV clips
MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
MAX_DURATION = 1.0  # Maximum duration for KWS samples (typically 1s)
MIN_DURATION = 0.1  # Minimum duration for meaningful segments

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load model and processor ===
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

# === Load audio from Common Voice dataset (using updated path) ===
dataset = load_dataset("mozilla-foundation/common_voice_11_0", COMMON_VOICE_LANG, 
                      split=COMMON_VOICE_SPLIT, trust_remote_code=True)
sample = dataset[SAMPLE_INDEX]
audio_path = sample["audio"]["path"]
print(f"Processing: {audio_path}")
print(f"Text: {sample['sentence']}")

# === Load and preprocess audio ===
waveform, sr = torchaudio.load(audio_path)
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
    
    # Fixed condition - check for pad token and other special tokens without using blank_token
    if token == last_token or token == processor.tokenizer.pad_token or token in ["<pad>", "<s>", "</s>"]:
        continue

    end_time = i * frame_duration
    segments.append((token, end_time))
    last_token = token

# === Export segments to .wav files ===
segment_start = 0.0
valid_subwords = 0
for idx, (token, end_time) in enumerate(segments):
    start_sample = int(segment_start * sr)
    end_sample = int(end_time * sr)
    segment_waveform = waveform[:, start_sample:end_sample]
    duration = end_time - segment_start
    
    # Skip segments that are too short or too long for KWS
    if duration < MIN_DURATION or duration > MAX_DURATION:
        segment_start = end_time
        continue
    
    # Sanitize file name - handle Vietnamese characters carefully
    token_clean = token.replace("▁", "").replace("/", "_").strip("_")
    if token_clean == "":
        token_clean = f"unk_{idx}"

    file_path = os.path.join(OUTPUT_DIR, f"{valid_subwords:03d}_{token_clean}.wav")
    torchaudio.save(file_path, segment_waveform, sample_rate=sr)
    
    print(f"Saved: {file_path}  | Duration: {duration:.2f}s | Token: {token}")
    valid_subwords += 1
    segment_start = end_time

print(f"Extracted {valid_subwords} valid KWS segments")

