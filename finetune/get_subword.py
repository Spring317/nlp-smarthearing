import os
import torch
import torchaudio
import librosa
import re
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Tuple, Optional, Dict, Any


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


def main() -> None:
    """Main function to extract Vietnamese keywords."""
    # Settings
    COMMON_VOICE_LANG = "vi"
    COMMON_VOICE_SPLIT = "test"
    SAMPLE_INDEX = 0
    MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    OUTPUT_DIR = "kws_segments"
    
    # Initialize extractor
    extractor = VietnameseKeywordExtractor(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR
    )
    
    # Load dataset
    dataset = extractor.load_dataset(COMMON_VOICE_LANG, COMMON_VOICE_SPLIT)
    
    # Process sample
    valid_segments = extractor.process_sample(SAMPLE_INDEX, dataset)
    
    print(f"Extracted {valid_segments} valid KWS segments")


if __name__ == "__main__":
    main()

