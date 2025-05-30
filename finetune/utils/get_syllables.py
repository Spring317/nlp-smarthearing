from typing import Set
import os 

def extract_syllables(base_dir: str) -> Set[str]:
    """
    Extract unique syllables from WAV files in the specified directory.

    Args:
        base_dir (str): Path to the directory containing WAV files.

    Returns:
        Set[str]: A set of unique syllables extracted from the filenames.
    """

    wav_files = [f for f in os.listdir(base_dir) if f.endswith('.wav')]

    # Extract unique syllables
    syllables = set()
    for filename in wav_files:
        # Extract syllable from filename (after index prefix)
        parts = filename.split('_', 1)
        if len(parts) > 1:
            syllable = parts[1].split('.')[0]
            if syllable:
                syllables.add(syllable)

    return syllables

