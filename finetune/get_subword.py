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
# This line of code is checking if the `extractor` object has a method named `get_extracted_keywords`.
        sr: int, 
        syllables: List[str],
        char_tokens: str,
        chars_per_sec: float
    ) -> int:
        """Extract audio segments for each syllable.
        
        Args:
            waveform: Audio waveform tensor
            sr: Sample rate
            syllables: List of syllables
            char_tokens: Full character transcription
            chars_per_sec: Characters per second for timing estimation
            
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
            
            # Calculate start and end samples
            start_sample = int(current_pos / len(char_tokens) * waveform.shape[1])
            end_sample = int((current_pos + syllable_len) / len(char_tokens) * waveform.shape[1])
            
            # Extract audio segment
            segment_waveform = waveform[:, start_sample:end_sample]
            
            # Clean syllable for filename
            syllable_clean = re.sub(
                r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', 
                '', syllable
            )
            
            # Save syllable audio
            file_path = os.path.join(self.output_dir, f"{valid_subwords:03d}_{syllable_clean}.wav")
            torchaudio.save(file_path, segment_waveform, sample_rate=sr)
            
            print(f"Saved: {file_path} | Duration: {syllable_duration:.2f}s | Syllable: {syllable}")
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
    
    # Process entire dataset (or subset)
    stats = extractor.process_dataset(dataset, args.max_samples)
    
    # Create syllable subfolders for KWS training
    print("Organizing extracted keywords for KWS training...")
    organize_keywords_for_kws(args.output_dir)


def organize_keywords_for_kws(base_dir: str) -> None:
    """Organize extracted keywords into syllable-based folders for KWS training.
    
    Args:
        base_dir: Base directory containing extracted WAV files
    """
    syllables = extract_syllables(base_dir)
    # Create syllable directories and copy files
    for syllable in syllables:
        # Create directory for syllable
        syllable_dir = os.path.join(base_dir, "syllables", syllable)
        os.makedirs(syllable_dir, exist_ok=True)
        
        # Count files for this syllable
        count = 0
        
        # Find all files for this syllable
        for filename in wav_files:
            if f"_{syllable}." in filename:
                # Copy or move the file
                src_path = os.path.join(base_dir, filename)
                dst_path = os.path.join(syllable_dir, filename)
                # Just create a symbolic link to save space
                if not os.path.exists(dst_path):
                    os.symlink(os.path.abspath(src_path), dst_path)
                count += 1
        
        print(f"Syllable '{syllable}': {count} samples")


if __name__ == "__main__":
    main()


