import os
import torch
import torchaudio
import librosa
import re
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import argparse
import pandas as pd
from utils.get_syllables import extract_syllables
import shutil
import numpy as np
from scipy.signal import resample
import soundfile as sf
import random

class VietnameseKeywordExtractor:
    """A class for extracting Vietnamese keywords from audio files."""
    
    def __init__(
        self,
        model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        output_dir: str = "kws_segments",
        min_duration: float = 0.1,
        max_duration: float = 1.0
    ) -> None:
        """Initialize the keyword extractor.
        
        Args:
            model_name: The pretrained ASR model to use
            output_dir: Directory to save extracted keyword segments
            min_duration: Minimum duration (seconds) for a valid keyword segment
            max_duration: Maximum duration (seconds) for a valid keyword segment
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load ASR model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        
    def load_dataset(self, lang: str = "vi", split: str = "test") -> Dataset:
        """Load a dataset from Common Voice.
        
        Args:
            lang: Language code
            split: Dataset split (train, test, validation)
            
        Returns:
            A dataset object
        """
        return load_dataset("mozilla-foundation/common_voice_11_0", lang, 
                           split=split, trust_remote_code=True)
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform_np = librosa.resample(waveform.numpy()[0], orig_sr=sr, target_sr=16000)
            waveform = torch.tensor(waveform_np).unsqueeze(0)
            sr = 16000
        return waveform, sr
    
    def transcribe_audio(self, waveform: torch.Tensor, sr: int) -> Tuple[str, List[int]]:
        """Transcribe audio using the ASR model.
        
        Args:
            waveform: Audio waveform tensor
            sr: Sample rate
            
        Returns:
            Tuple of (transcription, token_ids)
        """
        inputs = self.processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
        # Get token predictions with CTC decoding
        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
        transcription = self.processor.decode(pred_ids).lower()
        
        return transcription, pred_ids
    
    def extract_syllables(self, text: str) -> List[str]:
        """Extract Vietnamese syllables from text.
        
        Args:
            text: Vietnamese text
            
        Returns:
            List of syllables
        """
        # Vietnamese syllables are typically separated by spaces
        return re.findall(r'\S+', text)
    
    def process_sample(self, sample_index: int, dataset: Dataset) -> int:
        """Process a single audio sample and extract keywords.
        
        Args:
            sample_index: Index of the sample to process
            dataset: Dataset containing the sample
            
        Returns:
            Number of valid keyword segments extracted
        """
        sample = dataset[sample_index]
        audio_path = sample["audio"]["path"]
        print(f"Processing: {audio_path}")
        print(f"Text: {sample['sentence']}")
        
        # Load and preprocess audio
        waveform, sr = self.load_audio(audio_path)
        
        # Transcribe audio
        char_tokens, pred_ids = self.transcribe_audio(waveform, sr)
        
        # Extract syllables
        syllables = self.extract_syllables(char_tokens)
        print(f"Detected syllables: {syllables}")
        
        # Calculate timing information
        audio_len_sec = waveform.shape[1] / sr
        chars_per_sec = len(char_tokens) / audio_len_sec
        
        # Extract segments for each syllable
        return self.extract_syllable_segments(waveform, sr, syllables, 
                                             char_tokens, chars_per_sec)
    
    def extract_syllable_segments(
        self, 
        waveform: torch.Tensor, 
        sr: int, 
        syllables: List[str],
        char_tokens: str,
        chars_per_sec: float,
        num_chunks: int = 3  # Number of chunks to generate per syllable
    ) -> int:
        """Extract audio segments for each syllable with multiple chunks.
        
        Args:
            waveform: Audio waveform tensor
            sr: Sample rate
            syllables: List of syllables
            char_tokens: Full character transcription
            chars_per_sec: Characters per second for timing estimation
            num_chunks: Number of different chunk positions to generate
            
        Returns:
            Number of valid segments extracted
        """
        valid_subwords = 0
        current_pos = 0
        
        for syllable in syllables:
            # Estimate syllable duration based on character count
            syllable_len = len(syllable)
            syllable_duration = syllable_len / chars_per_sec
            
            # Skip if too short or too long
            if syllable_duration < self.min_duration or syllable_duration > self.max_duration:
                current_pos += syllable_len
                continue
            
            # Calculate base start and end samples
            base_start = int(current_pos / len(char_tokens) * waveform.shape[1])
            base_end = int((current_pos + syllable_len) / len(char_tokens) * waveform.shape[1])
            segment_length = base_end - base_start
            
            # Generate multiple chunks with different positions
            for chunk_idx in range(num_chunks):
                # Calculate random offset within 20% of the segment length
                max_offset = int(segment_length * 0.2)
                start_offset = random.randint(-min(max_offset, base_start), max_offset)
                end_offset = random.randint(-max_offset, max_offset)
                
                # Apply offsets to get chunk position
                chunk_start = max(0, base_start + start_offset)
                chunk_end = min(waveform.shape[1], base_end + end_offset)
                
                # Extract audio segment
                segment_waveform = waveform[:, chunk_start:chunk_end]
                
                # Clean syllable for filename
                syllable_clean = re.sub(
                    r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', 
                    '', syllable
                )
                
                # Save syllable audio with chunk identifier
                file_path = os.path.join(
                    self.output_dir, 
                    f"{valid_subwords:03d}_{syllable_clean}_chunk{chunk_idx}.wav"
                )
                
                # Only save if the chunk is long enough
                chunk_duration = (chunk_end - chunk_start) / sr
                if chunk_duration >= self.min_duration:
                    torchaudio.save(file_path, segment_waveform, sample_rate=sr)
                    print(f"Saved: {file_path} | Duration: {chunk_duration:.2f}s | Syllable: {syllable}")
                    valid_subwords += 1
        
        current_pos += syllable_len
    
        return valid_subwords
    
    def process_dataset(self, dataset: Dataset, max_samples: Optional[int] = None) -> Dict[str, int]:
        """Process multiple samples from the dataset and extract keywords.
        
        Args:
            dataset: Dataset containing samples
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            Dictionary with statistics about extracted keywords
        """
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        print(f"Processing {num_samples} samples from dataset...")
        
        # Track statistics
        stats = {
            "total_samples": num_samples,
            "processed_samples": 0,
            "total_segments": 0,
            "unique_syllables": set(),
            "failed_samples": 0
        }
        
        # Create a metadata DataFrame
        metadata = []
        
        # Process each sample with progress bar
        for i in tqdm(range(num_samples)):
            try:
                # Process the sample
                segments = self.process_sample(i, dataset)
                
                stats["processed_samples"] += 1
                stats["total_segments"] += segments
                
                # Add sample info to metadata
                if segments > 0:
                    sample = dataset[i]
                    metadata.append({
                        "sample_id": i,
                        "text": sample["sentence"],
                        "segments_extracted": segments,
                        "audio_path": sample["audio"]["path"]
                    })
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                stats["failed_samples"] += 1
        
        # Save metadata to CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(self.output_dir, "metadata.csv"), index=False)
        
        print(f"Processed {stats['processed_samples']} samples")
        print(f"Failed to process {stats['failed_samples']} samples")
        print(f"Extracted {stats['total_segments']} keyword segments total")
        
        return stats


def apply_augmentation(audio_path: str, out_path: str, sr: int = 16000):
    """Apply random augmentation to audio file.
    
    Args:
        audio_path: Path to input audio file
        out_path: Path to save augmented audio
        sr: Sample rate
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    # Sanitize output filename - replace non-ASCII characters
    sanitized_path = out_path
    if not all(ord(c) < 128 for c in out_path):
        # Replace non-ASCII characters with ASCII equivalents
        import unicodedata
        sanitized_name = unicodedata.normalize('NFKD', os.path.basename(out_path))
        sanitized_name = ''.join([c for c in sanitized_name if not unicodedata.combining(c)])
        sanitized_name = ''.join([c if ord(c) < 128 else '_' for c in sanitized_name])
        sanitized_path = os.path.join(os.path.dirname(out_path), sanitized_name)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Randomly choose augmentation method
    aug_type = random.choice(['pitch', 'speed', 'noise', 'shift'])
    
    if aug_type == 'pitch':
        # Pitch shift up or down by 0-2 semitones
        n_steps = random.uniform(-2, 2)
        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    elif aug_type == 'speed':
        # Change speed by ±10%
        speed_factor = random.uniform(0.9, 1.1) 
        y_aug = librosa.effects.time_stretch(y, rate=speed_factor)
    
    elif aug_type == 'noise':
        # Add small random noise
        noise_factor = random.uniform(0.001, 0.005)
        noise = np.random.randn(len(y))
        y_aug = y + noise_factor * noise
    
    else:  # shift
        # Shift the audio slightly
        shift_max = int(sr * 0.05)  # max 50ms shift
        shift = random.randint(-shift_max, shift_max)
        y_aug = np.roll(y, shift)
    
    # Save augmented audio
    sf.write(sanitized_path, y_aug, sr)
    
    # If we sanitized the path, inform the user
    if sanitized_path != out_path:
        print(f"Saved augmented audio to sanitized filename: {sanitized_path} (original: {out_path})")

def organize_keywords_for_kws(base_dir: str, train_ratio: float = 0.8, 
                            target_samples: int = 50) -> None:
    """Create metadata file for train/test splits instead of copying files."""
    # Get all syllables and wav files
    syllables = extract_syllables(base_dir)
    wav_files = [f for f in os.listdir(base_dir) if f.endswith('.wav')]
    
    print("Creating dataset splits metadata...")
    
    # Initialize metadata dictionary
    metadata = {
        'train': [],
        'test': [],
        'statistics': {}
    }
    
    for syllable in tqdm(syllables, desc="Processing syllables"):
        # Find all files for this syllable
        syllable_files = [f for f in wav_files if f"_{syllable}." in f]
        num_original = len(syllable_files)
        
        if num_original == 0:
            continue
            
        # Split files into train/test
        min_train_samples = min(1, num_original)
        split_idx = max(min_train_samples, int(len(syllable_files) * train_ratio))
        random.shuffle(syllable_files)
        train_files = syllable_files[:split_idx]
        test_files = syllable_files[split_idx:]
        
        # Add to metadata
        for f in train_files:
            metadata['train'].append({
                'file': f,
                'path': os.path.join(base_dir, f),
                'syllable': syllable,
                'split': 'train',
                'augmented': False
            })
            
        for f in test_files:
            metadata['test'].append({
                'file': f,
                'path': os.path.join(base_dir, f),
                'syllable': syllable,
                'split': 'test',
                'augmented': False
            })
            
        # Generate augmented sample entries if needed
        train_augment_needed = max(0, target_samples - len(train_files))
        if train_augment_needed > 0 and len(train_files) > 0:
            for i in range(train_augment_needed):
                # Select source file by rotating through available files
                source_file = train_files[i % len(train_files)]
                aug_filename = f"aug_{i}_{source_file}"
                
                metadata['train'].append({
                    'file': aug_filename,
                    'original_file': source_file,
                    'path': os.path.join(base_dir, source_file),
                    'syllable': syllable,
                    'split': 'train',
                    'augmented': True,
                    'aug_index': i
                })
        
        # Add statistics
        metadata['statistics'][syllable] = {
            'num_original': num_original,
            'num_train': len(train_files),
            'num_test': len(test_files),
            'num_augmented': train_augment_needed if train_augment_needed > 0 else 0
        }
        
        print(f"Syllable '{syllable}': {num_original} original, "
              f"{len(train_files)} train, {len(test_files)} test, "
              f"{train_augment_needed if train_augment_needed > 0 else 0} augmented")
    
    # Save metadata to JSON
    import json
    metadata_path = os.path.join(base_dir, 'splits_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Create CSV files for easier loading
    import pandas as pd
    
    # Train split
    train_df = pd.DataFrame(metadata['train'])
    train_df.to_csv(os.path.join(base_dir, 'train_metadata.csv'), index=False)
    
    # Test split
    test_df = pd.DataFrame(metadata['test'])
    test_df.to_csv(os.path.join(base_dir, 'test_metadata.csv'), index=False)
    
    print("Created CSV files: train_metadata.csv, test_metadata.csv")

def load_dataset_split(base_dir: str, split: str = 'train'):
    """Load dataset split using metadata."""
    try:
        metadata_path = os.path.join(base_dir, f'{split}_metadata.csv')
        df = pd.read_csv(metadata_path)
        
        # If this is training data, handle augmentation
        if split == 'train':
            augmented_samples = df[df['augmented'] == True]
            for _, row in augmented_samples.iterrows():
                # Generate full output path
                out_file = os.path.join(base_dir, row['file'])
                # Apply augmentation if file doesn't exist
                if not os.path.exists(out_file):
                    apply_augmentation(row['path'], out_file)
        
        return df
    except Exception as e:
        print(f"Error loading dataset split: {str(e)}")
        raise

def main() -> None:
    """Main function to extract Vietnamese keywords."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract Vietnamese keywords from audio")
    parser.add_argument("--lang", default="vi", help="Language code")
    parser.add_argument("--split", default="test", help="Dataset split (train, test, validation)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process (None for all)")
    parser.add_argument("--output-dir", default="kws_segments", help="Output directory")
    parser.add_argument("--model", default="nguyenvulebinh/wav2vec2-base-vietnamese-250h", 
                       help="Pretrained ASR model")
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = VietnameseKeywordExtractor(
        model_name=args.model,
        output_dir=args.output_dir
    )
    
    # Load dataset
    print(f"Loading {args.lang} dataset ({args.split} split)...")
    dataset = extractor.load_dataset(args.lang, args.split)
    
    # Process dataset with multiple chunks per syllable
    stats = extractor.process_dataset(dataset, args.max_samples)
    
    # Organize dataset with augmentation and train/test split, then cleanup
    print("Organizing extracted keywords for KWS training...")
    organize_keywords_for_kws(args.output_dir, train_ratio=0.8, target_samples=50, cleanup=True)


if __name__ == "__main__":
    main()


