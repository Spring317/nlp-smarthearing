import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from DS_CNN import DS_CNN_KWS, extract_mfcc_features
from get_subword import load_dataset_split

class KeywordSpottingDataset(Dataset):
    """Dataset for keyword spotting using metadata CSV files"""
    def __init__(self, base_dir: str, split: str = 'train'):
        self.base_dir = base_dir
        self.df = load_dataset_split(base_dir, split)
        self.classes = sorted(self.df['syllable'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['path'] if not row['augmented'] else os.path.join(self.base_dir, row['file'])
        
        # Extract MFCC features
        mfcc = extract_mfcc_features(file_path)
        mfcc_tensor = torch.from_numpy(mfcc).float().unsqueeze(0)  # Add channel dimension
        
        # Get class label
        label = self.class_to_idx[row['syllable']]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return mfcc_tensor, label_tensor
    
    def get_num_classes(self):
        return len(self.classes)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss/len(val_loader), 100.*correct/total

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Keyword Spotting Training')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-dir', default='kws_segments')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save-dir', default='checkpoints')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    train_dataset = KeywordSpottingDataset(args.data_dir, split='train')
    test_dataset = KeywordSpottingDataset(args.data_dir, split='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    num_classes = train_dataset.get_num_classes()
    model = DS_CNN_KWS(num_classes=num_classes)
    
    # Load convolutional weights from C++ model but skip FC layer
    model.load_weights_from_cpp_model(skip_fc=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Adjust learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': train_dataset.classes
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Saved new best model with validation accuracy: {val_acc:.2f}%')
        
        # Print epoch statistics
        print(f'\nTraining Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')

if __name__ == '__main__':
    main()