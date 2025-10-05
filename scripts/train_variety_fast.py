"""
Fast Apple Variety Classifier Training
======================================
Optimized version for quick training with reduced dataset and epochs.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import random

class AppleVarietyDataset(Dataset):
    """Dataset for apple variety classification."""
    
    def __init__(self, data_dir, transform=None, max_samples_per_class=200):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['Sharbati', 'Sunehari', 'Maharaji', 'Splendour', 'Himsona', 'Himkiran']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.max_samples_per_class = max_samples_per_class
        
        # Load all image paths and labels (with sampling)
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_samples = []
            
            if os.path.exists(class_dir):
                # Check for images directly in class directory
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        class_samples.append((img_path, self.class_to_idx[class_name]))
                
                # Also check for subdirectories (in case of nested structure)
                for item in os.listdir(class_dir):
                    item_path = os.path.join(class_dir, item)
                    if os.path.isdir(item_path) and not item.startswith('.'):
                        for img_name in os.listdir(item_path):
                            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(item_path, img_name)
                                class_samples.append((img_path, self.class_to_idx[class_name]))
            
            # Sample subset for faster training
            if len(class_samples) > self.max_samples_per_class:
                class_samples = random.sample(class_samples, self.max_samples_per_class)
            
            self.samples.extend(class_samples)
            print(f"  {class_name}: {len(class_samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_fast_transforms():
    """Get optimized data transforms for faster training."""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Smaller size for speed
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_fast_model(num_classes=6):
    """Create a lighter model for faster training."""
    
    # Use MobileNetV2 instead of EfficientNet for speed
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze early layers for faster training
    for param in model.features[:-3].parameters():
        param.requires_grad = False
    
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    
    return model

def train_fast_model(model, train_loader, val_loader, num_epochs=5, device='cpu'):
    """Fast training with fewer epochs and aggressive optimization."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            current_acc = running_corrects.double() / total_samples * 100
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset) * 100
        
        # Validation phase (quick validation)
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset) * 100
        
        # Store history
        history['train_losses'].append(epoch_loss)
        history['val_losses'].append(val_loss)
        history['train_accuracies'].append(epoch_acc.item())
        history['val_accuracies'].append(val_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
    
    return model, history

def main():
    """Fast training main function."""
    
    print("🚀 Fast Apple Variety Classifier Training")
    print("=" * 45)
    
    # Check if data directory exists
    data_dir = "data/varieties"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform, val_transform = get_fast_transforms()
    
    # Create datasets with limited samples for speed
    print("\n📊 Loading datasets (max 200 samples per class)...")
    train_dataset = AppleVarietyDataset(data_dir, transform=train_transform, max_samples_per_class=200)
    val_dataset = AppleVarietyDataset(data_dir, transform=val_transform, max_samples_per_class=50)
    
    print(f"\nTotal training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No training images found. Please add images to the data directories.")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Create lightweight model
    model = create_fast_model(num_classes=6)
    print(f"Model created: MobileNetV2 with 6 classes (optimized for speed)")
    
    # Fast training
    print("\n🚀 Starting fast training (5 epochs)...")
    trained_model, history = train_fast_model(model, train_loader, val_loader, 
                                            num_epochs=5, device=device)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/variety_classifier.pt"
    torch.save(trained_model, model_path)
    print(f"\n✅ Model saved: {model_path}")
    
    # Save training history
    history_path = "models/variety_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training history saved: {history_path}")
    
    # Quick evaluation
    print(f"\n📊 Final Results:")
    print(f"Best Training Accuracy: {max(history['train_accuracies']):.2f}%")
    print(f"Best Validation Accuracy: {max(history['val_accuracies']):.2f}%")
    
    print(f"\n🎉 Fast training completed!")
    print(f"Model is ready for use in the Apple Intelligence System.")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    main()
