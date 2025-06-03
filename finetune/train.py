import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Add this import
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from collections import Counter
from DS_CNN import DS_CNN_KWS, extract_mfcc_features, load_weights_from_cpp_model
import matplotlib.pyplot as plt


class MultiLabelKeywordDataset(Dataset):
    """Dataset for multi-label keyword spotting using metadata CSV files"""
    
    def __init__(self, metadata_path, fixed_length=100):
        """
        Args:
            metadata_path: Path to metadata CSV file
            fixed_length: Fixed length for MFCC features
        """
        print(f"Loading dataset from {metadata_path}")
        self.df = pd.read_csv(metadata_path)
        self.fixed_length = fixed_length
        
        # Load keywords
        base_dir = os.path.dirname(metadata_path)
        with open(os.path.join(base_dir, 'keywords.txt'), 'r', encoding='utf-8') as f:
            self.keywords = [line.strip() for line in f]
        
        self.num_keywords = len(self.keywords)
        print(f"Loaded dataset with {len(self.df)} samples and {self.num_keywords} keywords")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            audio_path = row['path']
            
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                # Create dummy tensors with proper dimensions
                mfcc_tensor = torch.zeros(1, 40, self.fixed_length)
                labels = torch.zeros(self.num_keywords)
                return mfcc_tensor, labels
                
            # Extract MFCC features
            mfcc = extract_mfcc_features(audio_path)
            # Ensure consistent feature length
            mfcc = self.pad_or_trim_features(mfcc)
            mfcc_tensor = torch.from_numpy(mfcc).float().unsqueeze(0)  # Add channel dim
            
            # Get multi-hot label vector
            keyword_indices = eval(row['keyword_indices'])  # Convert string repr to list
            labels = np.zeros(self.num_keywords, dtype=np.float32)
            for idx in keyword_indices:
                labels[idx] = 1.0
            
            return mfcc_tensor, torch.from_numpy(labels).float()
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Create new tensors (which are resizable) with proper dimensions
            mfcc_tensor = torch.zeros(1, 40, self.fixed_length)
            labels = torch.zeros(self.num_keywords)
            return mfcc_tensor, labels
    
    def pad_or_trim_features(self, features):
        """Ensure features have consistent length by padding or trimming"""
        _, time_steps = features.shape
        
        if time_steps > self.fixed_length:
            # Trim (take center portion)
            start = (time_steps - self.fixed_length) // 2
            features = features[:, start:start+self.fixed_length]
        elif time_steps < self.fixed_length:
            # Pad with zeros
            padding = np.zeros((features.shape[0], self.fixed_length - time_steps))
            features = np.concatenate([features, padding], axis=1)
            
        return features


class MultiLabelDS_CNN_KWS(DS_CNN_KWS):
    """Modified DS_CNN_KWS for multi-label classification"""
    def __init__(self, num_keywords, *args, **kwargs):
        super().__init__(num_classes=num_keywords, *args, **kwargs)
        self.num_keywords = num_keywords
        self._initialized = False  # Flag to track if FC layer has been properly initialized
    
    def forward(self, x):
        # Get batch size for reshaping
        batch_size = x.size(0)
        
        # Process through convolutional layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten preserving batch dimension
        x = x.view(batch_size, -1)
        
        # Dynamically resize FC layer if needed (first forward pass only)
        if not self._initialized:
            with torch.no_grad():
                in_features = x.shape[1]
                self.fc = nn.Linear(in_features, self.num_keywords).to(x.device)
                print(f"Dynamically resized FC layer: {in_features} -> {self.num_keywords}")
                self._initialized = True
                
                # If the original model was loaded, we need to initialize weights
                # with reasonable values since we're discarding the original FC weights
                nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
                nn.init.zeros_(self.fc.bias)
        
        # Forward through FC layer
        x = self.fc(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc="Training")
    first_batch = True
    for inputs, targets in pbar:
        if first_batch:
            print(f"Input shape: {inputs.shape}")
            first_batch = False
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        pbar.set_description(f"Train loss: {running_loss/len(pbar):.4f}")
        
        # Store predictions and targets for metrics
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics - use macro averaging for imbalanced classes
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    metrics = {
        'loss': running_loss / len(train_loader),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    return metrics


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track metrics
            running_loss += loss.item()
            pbar.set_description(f"Val loss: {running_loss/len(pbar):.4f}")
            
            # Store predictions and targets for metrics
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics - use macro averaging for imbalanced classes
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    metrics = {
        'loss': running_loss / len(val_loader),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    return metrics


def analyze_keyword_distribution(dataset):
    """Analyze and print keyword distribution statistics"""
    print("\n===== KEYWORD DISTRIBUTION =====")
    
    # Count occurrences of each keyword
    keyword_counts = np.zeros(dataset.num_keywords)
    for _, row in dataset.df.iterrows():
        for idx in eval(row['keyword_indices']):
            keyword_counts[idx] += 1
    
    # Get statistics
    total_samples = len(dataset.df)
    keywords_per_sample = sum(keyword_counts) / total_samples
    
    # Find most/least common keywords
    most_common_idx = np.argsort(keyword_counts)[-10:][::-1]
    least_common_idx = np.argsort(keyword_counts)[:10]
    
    print(f"Total samples: {total_samples}")
    print(f"Keywords per sample (avg): {keywords_per_sample:.2f}")
    
    print("\n----- Most common keywords -----")
    for idx in most_common_idx:
        print(f"{dataset.keywords[idx]}: {keyword_counts[idx]} samples ({100*keyword_counts[idx]/total_samples:.1f}%)")
    
    print("\n----- Least common keywords -----")
    for idx in least_common_idx:
        print(f"{dataset.keywords[idx]}: {keyword_counts[idx]} samples ({100*keyword_counts[idx]/total_samples:.1f}%)")
    
    # Class balance statistics
    percentile_25 = np.percentile(keyword_counts, 25)
    percentile_75 = np.percentile(keyword_counts, 75)
    print(f"\nMin count: {np.min(keyword_counts)}, Max count: {np.max(keyword_counts)}")
    print(f"25th percentile: {percentile_25}, 75th percentile: {percentile_75}")
    print(f"Zero-sample keywords: {np.sum(keyword_counts == 0)}")


def custom_collate(batch):
    """Custom collate function with fixed dimensions"""
    # Filter out any None values
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None, None
    
    # Use fixed dimensions - MFCC features should have 40 rows
    fixed_mfcc = 40
    fixed_length = 100  # This should match your dataset's fixed_length
    
    # Prepare batch data
    features = []
    labels = []
    
    for feature, label in batch:
        # Create new tensor with fixed dimensions
        padded_feature = torch.zeros(1, fixed_mfcc, fixed_length)
        
        # Copy original data into the padded tensor (with dimension checks)
        f_dim = min(feature.shape[1], fixed_mfcc)
        t_dim = min(feature.shape[2], fixed_length)
        padded_feature[0, :f_dim, :t_dim] = feature[0, :f_dim, :t_dim]
            
        features.append(padded_feature)
        labels.append(label)
    
    # Stack tensors
    features = torch.stack(features)
    labels = torch.stack(labels)
    
    return features, labels


def filter_rare_keywords(dataset, min_count=5, max_keywords=1000):
    """
    Filter out rare keywords and keep only the top N most common keywords
    
    Args:
        dataset: Dataset object with df attribute containing the dataset
        min_count: Minimum count for a keyword to be kept
        max_keywords: Maximum number of keywords to keep
    
    Returns:
        Filtered dataset
    """
    print("\n===== FILTERING RARE KEYWORDS =====")
    
    # Count keyword occurrences
    keyword_counts = np.zeros(dataset.num_keywords)
    for _, row in dataset.df.iterrows():
        for idx in eval(row['keyword_indices']):
            keyword_counts[idx] += 1
    
    # Find keywords above threshold
    valid_keywords = np.where(keyword_counts >= min_count)[0]
    
    # If too many keywords still remain, take top max_keywords
    if len(valid_keywords) > max_keywords:
        valid_keywords = np.argsort(keyword_counts)[-max_keywords:]
    
    # Create a mapping from old index to new index
    keyword_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_keywords)}
    
    # Filter and remap keywords
    filtered_rows = []
    for _, row in dataset.df.iterrows():
        old_indices = eval(row['keyword_indices'])
        # Keep only valid keywords
        new_indices = [keyword_map[idx] for idx in old_indices if idx in valid_keywords]
        
        if new_indices:  # If this sample has at least one valid keyword
            row_copy = row.copy()
            row_copy['keyword_indices'] = str(new_indices)
            filtered_rows.append(row_copy)
    
    # Create new dataframe with filtered rows
    dataset.df = pd.DataFrame(filtered_rows)
    
    # Update keywords list
    dataset.keywords = [dataset.keywords[idx] for idx in valid_keywords]
    dataset.num_keywords = len(dataset.keywords)
    
    print(f"Filtered from {len(keyword_counts)} to {dataset.num_keywords} keywords")
    print(f"Filtered from {len(dataset.df)} to {len(filtered_rows)} samples")
    
    return dataset


def apply_pca_to_features(model, train_loader, n_components=100, device="cpu"):
    """
    Apply PCA to the feature representations to reduce dimensionality
    
    Args:
        model: The trained model (up to conv layers)
        train_loader: DataLoader with training data
        n_components: Number of PCA components
        device: Device to use
    
    Returns:
        PCA model
    """
    print("\n===== APPLYING PCA TO FEATURES =====")
    
    # Extract features from the convolutional layers
    features = []
    with torch.no_grad():
        for inputs, _ in tqdm(train_loader, desc="Extracting features for PCA"):
            inputs = inputs.to(device)
            
            # Get feature representations (before FC layer)
            x = model.conv1(inputs)
            x = F.relu(x)
            x = model.conv2(x)
            x = model.conv3(x)
            x = model.conv4(x)
            x = model.conv5(x)
            x = x.view(inputs.size(0), -1)
            features.append(x.cpu().numpy())
    
    # Concatenate all features
    features = np.vstack(features)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(features)
    
    print(f"Applied PCA: {features.shape[1]} -> {n_components} dimensions")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    
    return pca


def plot_and_save_metrics(history, save_dir):
    """Plot training metrics and save to files"""
    metrics = ['loss', 'f1', 'precision', 'recall']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(history[f'train_{metric}'], label='Train')
        plt.plot(history[f'val_{metric}'], label='Validation')
        plt.title(f'{metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    print(f"Saved training metrics plot to {os.path.join(save_dir, 'training_metrics.png')}")
    
    # Save individual metrics
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        plt.plot(history[f'train_{metric}'], label='Train')
        plt.plot(history[f'val_{metric}'], label='Validation')
        plt.title(f'{metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}_curve.png'))


def main():
    parser = argparse.ArgumentParser(description="Train multi-label keyword detection model")
    parser.add_argument("--data-dir", default="kws_dataset", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--save-dir", default="checkpoints", help="Directory to save models")
    parser.add_argument("--fixed-length", type=int, default=100, help="Fixed length for MFCC features")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum count for keywords")
    parser.add_argument("--max-keywords", type=int, default=1000, help="Maximum number of keywords to keep")
    parser.add_argument("--use-pca", action="store_true", help="Apply PCA to features")
    parser.add_argument("--pca-components", type=int, default=100, help="Number of PCA components")
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = MultiLabelKeywordDataset(
        os.path.join(args.data_dir, "train_metadata.csv"), 
        fixed_length=args.fixed_length
    )
    val_dataset = MultiLabelKeywordDataset(
        os.path.join(args.data_dir, "test_metadata.csv"),
        fixed_length=args.fixed_length
    )
    
    # Analyze original keyword distribution
    analyze_keyword_distribution(train_dataset)
    
    # Filter rare keywords from both datasets
    train_dataset = filter_rare_keywords(
        train_dataset, 
        min_count=args.min_count,
        max_keywords=args.max_keywords
    )
    
    # Apply the same filtering to validation dataset
    # Note: We filter based on the same keywords as the training set
    valid_keywords = train_dataset.keywords
    
    # Create a mapping from old index to new index for validation set
    orig_keywords = []
    with open(os.path.join(args.data_dir, "keywords.txt"), 'r', encoding='utf-8') as f:
        orig_keywords = [line.strip() for line in f]
    
    valid_indices = [i for i, kw in enumerate(orig_keywords) if kw in valid_keywords]
    keyword_map = {old_idx: train_dataset.keywords.index(orig_keywords[old_idx]) 
                   for old_idx in valid_indices}
    
    # Filter and remap keywords for validation set
    filtered_rows = []
    for _, row in val_dataset.df.iterrows():
        old_indices = eval(row['keyword_indices'])
        # Keep only valid keywords
        new_indices = [keyword_map[idx] for idx in old_indices if idx in valid_indices]
        
        if new_indices:  # If this sample has at least one valid keyword
            row_copy = row.copy()
            row_copy['keyword_indices'] = str(new_indices)
            filtered_rows.append(row_copy)
    
    # Update validation dataset
    val_dataset.df = pd.DataFrame(filtered_rows)
    val_dataset.keywords = train_dataset.keywords.copy()
    val_dataset.num_keywords = len(val_dataset.keywords)
    
    # Analyze filtered keyword distribution
    print("\n===== FILTERED KEYWORD DISTRIBUTION =====")
    analyze_keyword_distribution(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    # Create model
    model = MultiLabelDS_CNN_KWS(
        num_keywords=train_dataset.num_keywords,
        num_mfcc=40,
        num_frames=args.fixed_length
    )
    
    # Load pre-trained weights
    load_weights_from_cpp_model(model, skip_fc=True)
    model = model.to(device)
    
    # Apply PCA if requested
    if args.use_pca:
        # This would need a more complex model architecture to implement
        print("WARNING: PCA implementation requires modifying the model architecture")
        print("         This is not implemented in the current version")
    
    # Calculate positive weight for loss function to handle class imbalance
    pos_weight = torch.ones(train_dataset.num_keywords)
    for i, row in train_dataset.df.iterrows():
        for idx in eval(row['keyword_indices']):
            pos_weight[idx] += 1
    neg_weight = len(train_dataset.df) - pos_weight + 1  # Adding 1 to avoid division by zero
    pos_weight = neg_weight / (pos_weight + 1e-5)
    pos_weight = pos_weight.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_f1 = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Adjust learning rate
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        # Update history
        for k in train_metrics:
            history[f'train_{k}'].append(train_metrics[k])
            history[f'val_{k}'].append(val_metrics[k])
        
        # Save best model based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'keywords': train_dataset.keywords
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model_multi_label.pth'))
            print(f"Saved best model with F1 score: {best_f1:.4f}")
    
    # Save training history
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot and save metrics
    plot_and_save_metrics(history, args.save_dir)
    
    print(f"\nTraining completed! Best validation F1 score: {best_f1:.4f}")


if __name__ == "__main__":
    print("Multi-label keyword detection training script")
    main()