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
        
        # Keep existing model architecture
        self.num_mfcc_features = num_mfcc 
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        # CONV1: Regular Convolution
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=(4, 10), 
            stride=(2, 2), padding=(1, 4), bias=True
        )
        
        # CONV2-5: Depthwise Separable Convolutions 
        self.conv2 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened features size
        self._fc_in_features = 64 * 5 * 25  # From conv output shape
        
        # Final FC layer with correct dimensions
        self.fc = nn.Linear(self._fc_in_features, num_classes)
    
    def forward(self, x):
        # Debug shape transformations
        batch_size = x.size(0)
        
        x = self.conv1(x)  # Shape: (batch, 64, 5, 25)
        x = F.relu(x)
        
        x = self.conv2(x)  # Shape maintained
        x = self.conv3(x)  # Shape maintained
        x = self.conv4(x)  # Shape maintained  
        x = self.conv5(x)  # Shape maintained
        
        # Flatten preserving batch dimension
        x = x.view(batch_size, -1)  # Shape: (batch, 64*5*25)
        
        x = self.fc(x)  # Shape: (batch, num_classes)
        
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
    Load weights from the C++ model's weight definitions
    
    Args:
        model: PyTorch DS_CNN_KWS model instance
        weights_file: Not used, kept for backward compatibility
    """
    import re
    
    def parse_weight_string(weight_str):
        # Remove curly braces and split by comma
        values = weight_str.strip('{}').split(',')
        # Convert to integers
        return [int(x.strip()) for x in values if x.strip()]
    
    def q7_to_float(q7_value):
        # Convert Q7 fixed-point to float
        # Q7 format has 7 fractional bits
        return float(q7_value) / (2**7)
    
    # Read the weights file
    with open('Hello_edge/src/ds_cnn_weights.h', 'r') as f:
        weights_content = f.read()
    
    # Extract weights using regex
    weight_patterns = {
        'conv1_wt': 'CONV1_WT\s*{\s*([-0-9,\s]+)}',
        'conv1_bias': 'CONV1_BIAS\s*{\s*([-0-9,\s]+)}',
        'conv2_ds_wt': 'CONV2_DS_WT\s*{\s*([-0-9,\s]+)}',
        'conv2_ds_bias': 'CONV2_DS_BIAS\s*{\s*([-0-9,\s]+)}',
        'conv2_pw_wt': 'CONV2_PW_WT\s*{\s*([-0-9,\s]+)}',
        'conv2_pw_bias': 'CONV2_PW_BIAS\s*{\s*([-0-9,\s]+)}',
        'conv3_ds_wt': 'CONV3_DS_WT\s*{\s*([-0-9,\s]+)}',
        'conv3_ds_bias': 'CONV3_DS_BIAS\s*{\s*([-0-9,\s]+)}',
        'conv3_pw_wt': 'CONV3_PW_WT\s*{\s*([-0-9,\s]+)}',
        'conv3_pw_bias': 'CONV3_PW_BIAS\s*{\s*([-0-9,\s]+)}',
        'conv4_ds_wt': 'CONV4_DS_WT\s*{\s*([-0-9,\s]+)}',
        'conv4_ds_bias': 'CONV4_DS_BIAS\s*{\s*([-0-9,\s]+)}',
        'conv4_pw_wt': 'CONV4_PW_WT\s*{\s*([-0-9,\s]+)}',
        'conv4_pw_bias': 'CONV4_PW_BIAS\s*{\s*([-0-9,\s]+)}',
        'conv5_ds_wt': 'CONV5_DS_WT\s*{\s*([-0-9,\s]+)}',
        'conv5_ds_bias': 'CONV5_DS_BIAS\s*{\s*([-0-9,\s]+)}',
        'conv5_pw_wt': 'CONV5_PW_WT\s*{\s*([-0-9,\s]+)}',
        'conv5_pw_bias': 'CONV5_PW_BIAS\s*{\s*([-0-9,\s]+)}',
        'final_fc_wt': 'FINAL_FC_WT\s*{\s*([-0-9,\s]+)}',
        'final_fc_bias': 'FINAL_FC_BIAS\s*{\s*([-0-9,\s]+)}'
    }
    
    weights = {}
    for name, pattern in weight_patterns.items():
        match = re.search(pattern, weights_content)
        if match:
            weights[name] = parse_weight_string(match.group(1))
    
    # Convert weights to PyTorch tensors and load into model
    # CONV1
    conv1_wt = torch.tensor([q7_to_float(w) for w in weights['conv1_wt']])
    conv1_wt = conv1_wt.reshape(64, 1, 4, 10)  # [out_ch, in_ch, kx, ky]
    model.conv1.weight.data = conv1_wt
    model.conv1.bias.data = torch.tensor([q7_to_float(w) for w in weights['conv1_bias']])
    
    # CONV2-5
    for i, conv in enumerate([model.conv2, model.conv3, model.conv4, model.conv5], 2):
        # Depthwise weights
        ds_wt = torch.tensor([q7_to_float(w) for w in weights[f'conv{i}_ds_wt']])
        ds_wt = ds_wt.reshape(64, 1, 3, 3)  # [out_ch, in_ch/groups, kx, ky]
        conv.depthwise.weight.data = ds_wt
        conv.depthwise.bias.data = torch.tensor([q7_to_float(w) for w in weights[f'conv{i}_ds_bias']])
        
        # Pointwise weights
        pw_wt = torch.tensor([q7_to_float(w) for w in weights[f'conv{i}_pw_wt']])
        pw_wt = pw_wt.reshape(64, 64, 1, 1)  # [out_ch, in_ch, 1, 1]
        conv.pointwise.weight.data = pw_wt
        conv.pointwise.bias.data = torch.tensor([q7_to_float(w) for w in weights[f'conv{i}_pw_bias']])
    
    # Final FC layer
    fc_wt = torch.tensor([q7_to_float(w) for w in weights['final_fc_wt']])
    fc_wt = fc_wt.reshape(model.num_classes, -1)  # [out_features, in_features]
    model.fc.weight.data = fc_wt
    model.fc.bias.data = torch.tensor([q7_to_float(w) for w in weights['final_fc_bias']])
    
    print("Loaded weights from C++ model")
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
    
    # Example input with debug prints
    example_input = torch.randn(1, 1, 10, 49)
    print(f"\nInput shape: {example_input.shape}")
    
    with torch.no_grad():
        example_output = model(example_input)
        print(f"Output shape: {example_output.shape}")
        
    print("\nModel is ready for keyword spotting!")


if __name__ == "__main__":
    main()
