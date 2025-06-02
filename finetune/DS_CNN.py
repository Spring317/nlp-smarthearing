import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from utils.get_syllables import extract_syllables

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise Separable Convolution implementation"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=True
        )
        
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=True
        )
    
    def forward(self, x):
        # Depthwise
        x = self.depthwise(x)
        x = F.relu(x)
        
        # Pointwise
        x = self.pointwise(x)
        x = F.relu(x)
        
        return x


class DS_CNN_KWS(nn.Module):
    """Depthwise Separable CNN for Keyword Spotting"""
    
    def __init__(self, num_classes=12, num_mfcc=10, num_frames=49):
        super(DS_CNN_KWS, self).__init__()
        
        # Define dimensions based on ds_cnn.h file
        self.num_mfcc_features = num_mfcc  # NUM_MFCC_COEFFS
        self.num_frames = num_frames       # NUM_FRAMES
        self.num_classes = num_classes     # OUT_DIM
        
        # CONV1: Regular Convolution (matches CONV1 parameters in ds_cnn.h)
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=(4, 10), 
            stride=(2, 2), padding=(1, 4), bias=True
        )
        
        # CONV2-5: Depthwise Separable Convolutions
        # Parameters match the definitions in ds_cnn.h
        self.conv2 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Final fully connected layer (matches FINAL_FC in ds_cnn.cpp)
        self.fc = nn.Linear(64 * 5 * 25, num_classes)
    
    def forward(self, x):
        # x expected to be [batch, 1, num_mfcc, num_frames]
        
        # CONV1
        x = self.conv1(x)
        x = F.relu(x)
        
        # CONV2-5 (depthwise separable convolutions)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten for FC layer
        x = torch.flatten(x, start_dim=1)
        
        # FC layer
        x = self.fc(x)
        
        return x


def extract_mfcc_features(audio_file, sample_rate=16000, n_mfcc=10, n_frames=49):
    """Extract MFCC features from audio file, matching the C++ implementation"""
    
    # Load audio file
    y, sr = librosa.load(audio_file, sr=sample_rate)
    
    # Parameters from ds_cnn.h
    frame_len_ms = 40
    frame_shift_ms = 20
    frame_len = int(sample_rate * frame_len_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)
    
    # Extract MFCC features (set parameters to match the C++ implementation)
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=frame_len, hop_length=frame_shift,
        window='hann'
    )
    
    # Pad or truncate to get the expected number of frames
    if mfccs.shape[1] < n_frames:
        pad_width = n_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
    elif mfccs.shape[1] > n_frames:
        mfccs = mfccs[:, :n_frames]
    
    return mfccs


def load_weights_from_cpp_model(model, weights_file=None):
    """
    Load weights from the C++ model
    
    Args:
        model: PyTorch DS_CNN_KWS model instance
        weights_file: Path to weights file if available
    """
    if weights_file:
        # Load weights from file (implementation needed)
        pass
    else:
        # For now, we're just initializing with PyTorch's default initialization
        print("Using default PyTorch weight initialization")
        
    # Convert fixed-point weights to floating point if needed
    # (Original model uses fixed-point Q7 format)
def get_keyword_labels():
    label = extract_syllables("kws_segments")
    return label


def predict_keyword(model, audio_file, keyword_labels):
    """
    Run keyword prediction on audio file
    
    Args:
        model: PyTorch DS_CNN_KWS model
        audio_file: Path to audio file
        keyword_labels: List of keyword labels (optional)
        
    Returns:
        class_idx: Predicted class index
        confidence: Confidence score
        label: Predicted label (if keyword_labels provided)
    """
    # Default keyword labels from the original model
   # This part of the code in the `predict_keyword` function is responsible for handling the scenario
   # when `keyword_labels` is not provided:
    
    # Extract MFCC features

    mfccs = extract_mfcc_features(audio_file)
    
    # Convert to tensor and add batch and channel dimensions
    x = torch.from_numpy(mfccs).float().unsqueeze(0).unsqueeze(0)
    
    # Run prediction
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze()
        
    # Get predicted class
    class_idx = torch.argmax(probs).item()
    confidence = probs[class_idx].item()
    label = keyword_labels[class_idx] if class_idx < len(keyword_labels) else "unknown"
    
    return class_idx, confidence, label



def main():
    # Create model
    
    labels = get_keyword_labels()
    NUM_CLASSES = len(labels) 
    model = DS_CNN_KWS(num_classes=NUM_CLASSES)
    print("DS-CNN KWS model created")
    
    # Load weights if available
    load_weights_from_cpp_model(model)
    
    # Example usage
    print("Model architecture:")
    print(model)
    
    # Example input tensor to verify shapes
    example_input = torch.randn(1, 1, 10, 49)  # batch_size, channels, num_mfcc, num_frames
    with torch.no_grad():
        example_output = model(example_input)
    print(f"\nInput shape: {example_input.shape}")
    print(f"Output shape: {example_output.shape}")
    
    print("\nModel is ready for keyword spotting!")


if __name__ == "__main__":
    main()
