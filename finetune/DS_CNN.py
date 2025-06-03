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
        
        self.num_mfcc_features = num_mfcc 
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        # Calculate output dimensions after CONV1
        h_out = (num_mfcc + 2 - 4) // 2 + 1  # After padding=(1,4) and stride=(2,2)
        w_out = (num_frames + 8 - 10) // 2 + 1
        
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
        self._fc_in_features = 64 * h_out * w_out
        
        # Final FC layer
        self.fc = nn.Linear(self._fc_in_features, num_classes)
    
    def forward(self, x):
        # Debug shape transformations
        batch_size = x.size(0)
        
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten preserving batch dimension
        x = x.view(batch_size, -1)
        
        # Add debug print to verify shape
        print(f"Shape before FC layer: {x.shape}")
        
        x = self.fc(x)
        return x


def extract_mfcc_features(audio_file, n_mfcc=40, n_fft=2048, hop_length=512):
    """Extract MFCC features from an audio file with dynamic FFT size handling."""
    import librosa
    import numpy as np
    
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # If audio is shorter than n_fft, adjust the n_fft size
    if len(y) < n_fft:
        # Set n_fft to the next power of 2 above the signal length
        n_fft = 2**int(np.ceil(np.log2(len(y))))
        # Make sure it's at least 512 (or another reasonable minimum)
        n_fft = max(512, n_fft)
        # Also adjust hop_length if necessary
        hop_length = min(hop_length, n_fft // 4)
    
    # Extract MFCCs with adjusted parameters
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Get the right shape for the model (time, features)
    mfccs = mfccs.T
    
    return mfccs


def load_weights_from_cpp_model(model, weights_file=None, skip_fc=True):
    """Load weights from the C++ model's weight definitions, optionally skipping FC layer"""
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
    with open('../Hello_edge/src/ds_cnn_weights.h', 'r') as f:
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
    
    # Skip loading FC layer weights if skip_fc is True
    if skip_fc:
        print("Skipping FC layer weights - using PyTorch initialization")
        return
        
    # Final FC layer 
    fc_wt = torch.tensor([q7_to_float(w) for w in weights['final_fc_wt']])
    
    # Check if number of weights matches expected size
    expected_size = model.num_classes * model._fc_in_features
    actual_size = len(fc_wt)
    
    if actual_size != expected_size:
        print(f"Warning: FC weight size mismatch!")
        print(f"Expected size: {expected_size} (num_classes={model.num_classes} * in_features={model._fc_in_features})")
        print(f"Actual size from weights file: {actual_size}")
        print("Using default PyTorch initialization for FC layer")
        return
        
    # Only reshape if sizes match
    fc_wt = fc_wt.reshape(model.num_classes, model._fc_in_features)
    model.fc.weight.data = fc_wt
    
    # Only load bias if number of classes matches
    fc_bias = torch.tensor([q7_to_float(w) for w in weights['final_fc_bias']])
    if len(fc_bias) == model.num_classes:
        model.fc.bias.data = fc_bias
    else:
        print(f"Warning: FC bias size mismatch ({len(fc_bias)} != {model.num_classes})")
        print("Using default PyTorch initialization for FC bias")

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
    # Get labels first to determine number of classes
    labels = get_keyword_labels()
    NUM_CLASSES = len(labels)
    print(f"Creating model with {NUM_CLASSES} classes based on dataset")
    
    # Create model with dataset's number of classes
    model = DS_CNN_KWS(num_classes=NUM_CLASSES)
    print("DS-CNN KWS model created")
    
    # Load weights but skip FC layer
    load_weights_from_cpp_model(model, skip_fc=True)
    
    print(f"Number of available labels: {len(labels)}")
    
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
