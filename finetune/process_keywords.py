import os
import torch
import torchaudio
import librosa
import re
import json
import pandas as pd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, Dataset, Audio
from typing import List, Dict, Any, Set, Union, Optional
from tqdm import tqdm
import argparse
import random
import shutil
from pathlib import Path
import glob


class KeywordProcessor:
    """Process audio files and extract keywords from transcriptions."""
    
    def __init__(
        self,
        model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        output_dir: str = "kws_dataset",
        min_keyword_length: int = 2,
        max_keyword_length: int = 10
    ) -> None:
        """Initialize the keyword processor."""
        self.model_name = model_name
        self.output_dir = output_dir
        self.min_keyword_length = min_keyword_length
        self.max_keyword_length = max_keyword_length
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load ASR model and processor
        print(f"Loading ASR model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
    
    def load_common_voice_dataset(self, lang: str = "vi", split: str = "test") -> Dataset:
        """Load a dataset from Common Voice."""
        print(f"Loading Common Voice dataset ({lang}, {split})...")
        try:
            return load_dataset("mozilla-foundation/common_voice_11_0", lang, 
                               split=split, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading Common Voice dataset: {e}")
            return None
    
    def load_vivos_dataset(self, vivos_path: str) -> Optional[Dataset]:
        """Load VIVOS dataset from disk."""
        if not vivos_path or not os.path.exists(vivos_path):
            print(f"VIVOS dataset path not found: {vivos_path}")
            return None
            
        print(f"Loading VIVOS dataset from {vivos_path}...")
        try:
            # VIVOS has train and test folders
            data = []
            
            # Process training data
            train_wavs = glob.glob(os.path.join(vivos_path, "train", "waves", "**", "*.wav"), recursive=True)
            for wav_path in tqdm(train_wavs, desc="Processing VIVOS train files"):
                # Get transcript path from wav path
                rel_path = os.path.relpath(wav_path, os.path.join(vivos_path, "train", "waves"))
                transcript_path = os.path.join(vivos_path, "train", "prompts", rel_path.replace('.wav', '.txt'))
                
                if os.path.exists(transcript_path):
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    data.append({
                        "audio": {"path": wav_path},
                        "sentence": text,
                        "split": "train"
                    })
            
            # Process test data
            test_wavs = glob.glob(os.path.join(vivos_path, "test", "waves", "**", "*.wav"), recursive=True)
            for wav_path in tqdm(test_wavs, desc="Processing VIVOS test files"):
                # Get transcript path from wav path
                rel_path = os.path.relpath(wav_path, os.path.join(vivos_path, "test", "waves"))
                transcript_path = os.path.join(vivos_path, "test", "prompts", rel_path.replace('.wav', '.txt'))
                
                if os.path.exists(transcript_path):
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    data.append({
                        "audio": {"path": wav_path},
                        "sentence": text,
                        "split": "test"
                    })
            
            print(f"Loaded {len(data)} samples from VIVOS dataset")
            if not data:
                return None
                
            # Create a Hugging Face dataset
            dataset = Dataset.from_dict({
                "audio": [item["audio"] for item in data],
                "sentence": [item["sentence"] for item in data],
                "split": [item["split"] for item in data]
            })
            
            return dataset
            
        except Exception as e:
            print(f"Error loading VIVOS dataset: {e}")
            return None
    
    def load_vivos_hf_dataset(self):
        """Load VIVOS dataset from Hugging Face datasets."""
        print(f"Loading VIVOS dataset from Hugging Face...")
        try:
            train_dataset = load_dataset("AILAB-VNUHCM/vivos", split="train", trust_remote_code=True)
            test_dataset = load_dataset("AILAB-VNUHCM/vivos", split="test", trust_remote_code=True)
            print(f"Loaded VIVOS: {len(train_dataset)} train, {len(test_dataset)} test samples")
            return [
                ("vivos_train", train_dataset),
                ("vivos_test", test_dataset)
            ]
        except Exception as e:
            print(f"Error loading VIVOS dataset from Hugging Face: {e}")
            return []
    
    def load_datasets(self, common_voice_lang="vi", common_voice_split="test", use_vivos=True):
        """Load and combine multiple datasets."""
        datasets = []
        
        # Load Common Voice
        cv_dataset = self.load_common_voice_dataset(common_voice_lang, common_voice_split)
        if cv_dataset:
            datasets.append(("common_voice", cv_dataset))
        
        # Load VIVOS from Hugging Face
        if use_vivos:
            vivos_datasets = self.load_vivos_hf_dataset()
            if vivos_datasets:
                datasets.extend(vivos_datasets)

        if not datasets:
            raise ValueError("No datasets could be loaded")

        print(f"Loaded {len(datasets)} datasets")
        return datasets
    
    def transcribe_audio(self, audio_path: str):
        """Transcribe audio file using the ASR model."""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform_np = waveform.numpy()[0]
            waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=16000)
            waveform = torch.tensor(waveform_np).unsqueeze(0)
            sr = 16000
            
        # Process through model
        inputs = self.processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Get transcription
        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
        transcription = self.processor.decode(pred_ids).lower()
        
        return transcription
    
    def extract_keywords(self, text: str) -> Set[str]:
        """Extract valid keywords from text."""
        # Split text into words (syllables for Vietnamese)
        words = re.findall(r'\S+', text)
        
        # Filter by length and validity
        valid_keywords = set()
        for word in words:
            # Clean word - remove non-alphanumeric characters
            clean_word = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', word)
            
            # Check if word is valid
            if (len(clean_word) >= self.min_keyword_length and 
                len(clean_word) <= self.max_keyword_length and
                clean_word):
                valid_keywords.add(clean_word)
                
        return valid_keywords
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single audio sample."""
        audio_path = sample["audio"]["path"]
        
        # Get ground truth text from sample
        gt_text = sample.get("sentence", "").lower()
        
        # Extract keywords from ground truth text
        gt_keywords = self.extract_keywords(gt_text)
        
        # Try ASR if ground truth has no valid keywords
        if not gt_keywords:
            asr_text = self.transcribe_audio(audio_path)
            asr_keywords = self.extract_keywords(asr_text)
            keywords = asr_keywords
            source = "asr"
        else:
            keywords = gt_keywords
            source = "ground_truth"
            
        # Return results
        return {
            "audio_path": audio_path,
            "keywords": list(keywords),
            "text": gt_text if source == "ground_truth" else asr_text,
            "source": source
        }
    
    def process_dataset(self, dataset, max_samples=None):
        """Process multiple samples from dataset."""
        # Determine number of samples to process
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        print(f"Processing {num_samples} samples from dataset...")
        
        # Process each sample
        results = []
        for i in tqdm(range(num_samples)):
            try:
                sample_data = self.process_sample(dataset[i])
                if sample_data["keywords"]:  # Only keep samples with keywords
                    results.append(sample_data)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                
        print(f"Successfully processed {len(results)} samples with keywords")
        
        return results
    
    def create_dataset(self, processed_samples, train_ratio=0.8):
        """Create training and testing datasets from processed samples."""
        # Collect all unique keywords
        all_keywords = set()
        for sample in processed_samples:
            all_keywords.update(sample["keywords"])
            
        keywords_list = sorted(list(all_keywords))
        print(f"Found {len(keywords_list)} unique keywords")
        
        # Create keyword to index mapping
        keyword_to_idx = {kw: i for i, kw in enumerate(keywords_list)}
        
        # Randomly split into train and test
        random.shuffle(processed_samples)
        split_idx = int(len(processed_samples) * train_ratio)
        
        train_samples = processed_samples[:split_idx]
        test_samples = processed_samples[split_idx:]
        
        print(f"Split into {len(train_samples)} training and {len(test_samples)} testing samples")
        
        # Create dataset directories
        audio_dir = os.path.join(self.output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Copy audio files and create metadata
        train_metadata = []
        test_metadata = []
        
        # Process training samples
        print("Processing training samples...")
        for i, sample in enumerate(tqdm(train_samples)):
            # Copy audio file
            src_path = sample["audio_path"]
            filename = f"train_{i:04d}.wav"
            dst_path = os.path.join(audio_dir, filename)
            
            try:
                shutil.copy(src_path, dst_path)
                
                # Create metadata entry
                entry = {
                    "id": f"train_{i:04d}",
                    "file": filename,
                    "path": dst_path,
                    "keywords": sample["keywords"],
                    "keyword_indices": [keyword_to_idx[kw] for kw in sample["keywords"]],
                    "text": sample["text"],
                    "split": "train"
                }
                train_metadata.append(entry)
            except Exception as e:
                print(f"Error copying audio file {src_path}: {e}")
        
        # Process test samples
        print("Processing test samples...")
        for i, sample in enumerate(tqdm(test_samples)):
            # Copy audio file
            src_path = sample["audio_path"]
            filename = f"test_{i:04d}.wav"
            dst_path = os.path.join(audio_dir, filename)
            
            try:
                shutil.copy(src_path, dst_path)
                
                # Create metadata entry
                entry = {
                    "id": f"test_{i:04d}",
                    "file": filename,
                    "path": dst_path,
                    "keywords": sample["keywords"],
                    "keyword_indices": [keyword_to_idx[kw] for kw in sample["keywords"]],
                    "text": sample["text"],
                    "split": "test"
                }
                test_metadata.append(entry)
            except Exception as e:
                print(f"Error copying audio file {src_path}: {e}")
        
        # Save metadata
        metadata = {
            "train": train_metadata,
            "test": test_metadata,
            "keywords": keywords_list,
            "keyword_to_idx": keyword_to_idx
        }
        
        # Save as JSON
        with open(os.path.join(self.output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        # Save as CSV files
        pd.DataFrame(train_metadata).to_csv(os.path.join(self.output_dir, "train_metadata.csv"), index=False)
        pd.DataFrame(test_metadata).to_csv(os.path.join(self.output_dir, "test_metadata.csv"), index=False)
        
        # Save keywords list
        with open(os.path.join(self.output_dir, "keywords.txt"), 'w', encoding='utf-8') as f:
            for keyword in keywords_list:
                f.write(f"{keyword}\n")
                
        print(f"Dataset created with {len(train_metadata)} training and {len(test_metadata)} testing samples")
        print(f"Dataset saved to {self.output_dir}")
    
    def process_datasets(self, datasets_list):
        """Process multiple datasets."""
        all_processed_samples = []
        
        for dataset_name, dataset in datasets_list:
            print(f"\nProcessing {dataset_name} dataset ({len(dataset)} samples)...")
            processed = self.process_dataset(dataset)
            all_processed_samples.extend(processed)
            print(f"Added {len(processed)} samples from {dataset_name}")
        
        print(f"\nTotal processed samples: {len(all_processed_samples)}")
        return all_processed_samples


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process audio files and extract keywords")
    parser.add_argument("--lang", default="vi", help="Language code for Common Voice")
    parser.add_argument("--cv-split", default="test", help="Common Voice dataset split (train, test, validation)")
    parser.add_argument("--use-vivos", action="store_true", default=True, help="Use VIVOS dataset from Hugging Face")
    parser.add_argument("--output-dir", default="kws_dataset", help="Output directory")
    parser.add_argument("--model", default="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
                       help="Pretrained ASR model")
    parser.add_argument("--min-length", type=int, default=2, help="Min keyword length")
    parser.add_argument("--max-length", type=int, default=10, help="Max keyword length")
    args = parser.parse_args()
    
    # Initialize processor
    processor = KeywordProcessor(
        model_name=args.model,
        output_dir=args.output_dir,
        min_keyword_length=args.min_length,
        max_keyword_length=args.max_length
    )
    
    # Load datasets
    datasets = processor.load_datasets(
        common_voice_lang=args.lang,
        common_voice_split=args.cv_split,
        use_vivos=args.use_vivos
    )
    
    # Process all datasets
    processed_samples = processor.process_datasets(datasets)
    
    # Create dataset
    processor.create_dataset(processed_samples, train_ratio=0.8)


if __name__ == "__main__":
    main()