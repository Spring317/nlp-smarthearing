import os
import torch
import torchaudio
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from itertools import groupby
import re

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

# === Group characters into syllables ===
# Get transcription from ASR model
with torch.no_grad():
    logits = model(**inputs).logits
    
# Get token predictions with CTC decoding
pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
char_tokens = processor.decode(pred_ids).lower()

# Split into syllables using regex pattern for Vietnamese
# Vietnamese syllables are typically separated by spaces
syllables = re.findall(r'\S+', char_tokens)
print(f"Detected syllables: {syllables}")

# Calculate approximate time per character
audio_len_sec = waveform.shape[1] / sr
chars_per_sec = len(char_tokens) / audio_len_sec

# Estimate syllable positions in audio
valid_subwords = 0
current_pos = 0
for syllable in syllables:
    # Estimate syllable duration based on character count
    syllable_len = len(syllable)
    syllable_duration = syllable_len / chars_per_sec
    
    # Skip if too short or too long
    if syllable_duration < MIN_DURATION or syllable_duration > MAX_DURATION:
        current_pos += syllable_len
        continue
    
    # Calculate start and end samples
    start_sample = int(current_pos / len(char_tokens) * waveform.shape[1])
    end_sample = int((current_pos + syllable_len) / len(char_tokens) * waveform.shape[1])
    
    # Extract audio segment
    segment_waveform = waveform[:, start_sample:end_sample]
    
    # Clean syllable for filename
    syllable_clean = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', syllable)
    
    # Save syllable audio
    file_path = os.path.join(OUTPUT_DIR, f"{valid_subwords:03d}_{syllable_clean}.wav")
    torchaudio.save(file_path, segment_waveform, sample_rate=sr)
    
    print(f"Saved: {file_path} | Duration: {syllable_duration:.2f}s | Syllable: {syllable}")
    valid_subwords += 1
    current_pos += syllable_len

print(f"Extracted {valid_subwords} valid KWS segments")

