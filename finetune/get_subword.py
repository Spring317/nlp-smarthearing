import os
import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from itertools import groupby

# === SETTINGS ===
COMMON_VOICE_LANG = "vi"  # Language code for Vietnamese
COMMON_VOICE_SPLIT = "test"  # Dataset split: "train", "test", "validation"
SAMPLE_INDEX = 0  # Index of the sample to process
OUTPUT_DIR = "kws_segments"  # Folder to store subword WAV clips
MODEL_NAME = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
MAX_DURATION = 1.0  # Maximum duration for KWS samples (typically 1s)
MIN_DURATION = 0.1  # Minimum duration for meaningful segments
CONFIDENCE_THRESHOLD = 0.5  # Reduced from 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load model and processor ===
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

# === Load audio from Common Voice dataset ===
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
    outputs = model(**inputs)
    logits = outputs.logits

# === Decode with proper word-level segmentation ===
pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
# Calculate token probabilities for confidence filtering
probs = torch.nn.functional.softmax(logits, dim=-1)
confidence = torch.max(probs, dim=-1)[0][0].tolist()  # Get confidence scores

logit_len = logits.shape[1]
audio_len_sec = waveform.shape[1] / sr
frame_duration = audio_len_sec / logit_len

# === Group tokens with CTC decoding ===
# Decode with CTC rules (collapse repeated tokens and remove special tokens)
collapsed_tokens = []
token_times = []

# Group duplicate consecutive tokens (CTC decoding)
for i, ((token_id, _), time_idx) in enumerate(zip(groupby(zip(pred_ids, confidence)), range(len(pred_ids)))):
    token = processor.tokenizer.convert_ids_to_tokens([token_id])[0]
    if token not in ["<pad>", "<s>", "</s>"] and token != processor.tokenizer.pad_token:
        collapsed_tokens.append(token)
        token_times.append(time_idx * frame_duration)

# Group into words (Vietnamese words are typically single syllables)
words = []
for i, (token, time) in enumerate(zip(collapsed_tokens, token_times)):
    # Each token is treated as a separate word in Vietnamese
    if token.strip():  # Skip empty tokens
        end_time = (time + frame_duration * 5)  # Extend slightly for better word capture
        start_time = max(0, time - frame_duration)  # Start slightly before 
        words.append((start_time, end_time, token, 1.0))  # Assuming confidence = 1.0 for now

# Add before word extraction code
print(f"Total tokens: {len(pred_ids)}")
print(f"Sample tokens: {processor.tokenizer.convert_ids_to_tokens(pred_ids[:20])}")
print(f"Sample confidence scores: {confidence[:5]}")

# === Export word segments to .wav files ===
valid_words = 0
for idx, (start_time, end_time, word, confidence) in enumerate(words):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    if end_sample > waveform.shape[1]:
        end_sample = waveform.shape[1]
    
    word_waveform = waveform[:, start_sample:end_sample]
    duration = end_time - start_time
    
    # Skip segments that are too short, too long, or low confidence
    if duration < MIN_DURATION or duration > MAX_DURATION or confidence < CONFIDENCE_THRESHOLD:
        continue
    
    # Sanitize and clean up the word for filename
    word_clean = "".join(c for c in word if c.isalnum() or c in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")
    if not word_clean:
        word_clean = f"unk_{idx}"

    file_path = os.path.join(OUTPUT_DIR, f"{valid_words:03d}_{word_clean}.wav")
    torchaudio.save(file_path, word_waveform, sample_rate=sr)
    
    print(f"Saved: {file_path} | Duration: {duration:.2f}s | Word: {word} | Confidence: {confidence:.2f}")
    valid_words += 1

print(f"Extracted {valid_words} valid word segments")

# Add before the end of the script
print(f"Words found before filtering: {len(words)}")
print(f"Words with low confidence: {sum(1 for _, _, _, conf in words if conf < CONFIDENCE_THRESHOLD)}")
print(f"Words outside duration range: {sum(1 for start, end, _, _ in words if (end-start) < MIN_DURATION or (end-start) > MAX_DURATION)}")
