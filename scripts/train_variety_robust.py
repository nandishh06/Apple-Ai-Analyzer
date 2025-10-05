"""
Robust Apple Variety Classifier Training
========================================
Creates a more generalizable model with proper train/validation split and regularization.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
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
from pathlib import Path

class RobustAppleVarietyDataset(Dataset):
    """Dataset with proper train/val split and data augmentation."""
    
    def __init__(self, data_dir, transform=None, max_samples_per_class=300):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['Sharbati', 'Sunehari', 'Maharaji', 'Splendour', 'Himsona', 'Himkiran']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.max_samples_per_class = max_samples_per_class
        
        # Load all image paths and labels with balanced sampling
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_samples = []
            
            if os.path.exists(class_dir):
                # Collect all images from this class
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        class_samples.append((img_path, self.class_to_idx[class_name]))
                
                # Also check subdirectories
                for item in os.listdir(class_dir):
                    item_path = os.path.join(class_dir, item)
                    if os.path.isdir(item_path) and not item.startswith('.'):
                        for img_name in os.listdir(item_path):
                            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(item_path, img_name)
                                class_samples.append((img_path, self.class_to_idx[class_name]))
            
            # Randomly sample to balance classes and prevent overfitting
            if len(class_samples) > self.max_samples_per_class:
                class_samples = random.sample(class_samples, self.max_samples_per_class)
            
            self.samples.extend(class_samples)
            print(f"  {class_name}: {len(class_samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            # Return a dummy image if loading fails
            dummy_image = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label

def get_robust_transforms():
    """Get transforms with proper augmentation and normalization."""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)  # Random erasing for regularization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_robust_model(num_classes=6):
    """Create a robust model with proper regularization."""
    
    # Use EfficientNet-B0 (lighter than B3 but still effective)
    model = models.efficientnet_b0(pretrained=True)
    
    # Freeze early layers
    for param in model.features[:-2].parameters():
        param.requires_grad = False
    
    # Replace classifier with dropout for regularization
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    
    return model

def train_robust_model(model, train_loader, val_loader, num_epochs=15, device='cpu'):
    """Train with proper validation and early stopping."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5
    
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
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            current_acc = running_corrects.double() / total_samples * 100
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset) * 100
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset) * 100
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Store history
        history['train_losses'].append(epoch_loss)
        history['val_losses'].append(val_loss)
        history['train_accuracies'].append(epoch_acc.item())
        history['val_accuracies'].append(val_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_variety_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(torch.load('models/best_variety_model.pth'))
    
    return model, history

def main():
    """Main training function."""
    
    print("🚀 Robust Apple Variety Classifier Training")
    print("=" * 50)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Check data directory
    data_dir = "data/varieties"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_robust_transforms()
    
    # Create full dataset
    print("\n📊 Loading dataset...")
    full_dataset = RobustAppleVarietyDataset(data_dir, transform=train_transform, max_samples_per_class=300)
    
    print(f"Total samples: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        print("❌ No training images found.")
        return
    
    # Split dataset properly
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms to validation set
    val_indices = val_dataset.indices
    val_dataset_proper = RobustAppleVarietyDataset(data_dir, transform=val_transform, max_samples_per_class=300)
    val_dataset_proper.samples = [val_dataset_proper.samples[i] for i in val_indices if i < len(val_dataset_proper.samples)]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Create model
    model = create_robust_model(num_classes=6)
    print("Model created: EfficientNet-B0 with regularization")
    
    # Train model
    print("\n🚀 Starting robust training...")
    trained_model, history = train_robust_model(model, train_loader, val_loader, 
                                              num_epochs=15, device=device)
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save(trained_model, "models/variety_classifier.pt")
    print(f"\n✅ Robust model saved: models/variety_classifier.pt")
    
    # Save training history
    with open("models/variety_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Print final results
    best_val_acc = max(history['val_accuracies'])
    final_val_acc = history['val_accuracies'][-1]
    
    print(f"\n📊 Training Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Generalization Gap: {history['train_accuracies'][-1] - final_val_acc:.2f}%")
    
    if best_val_acc > 70:
        print("✅ Model shows good performance!")
    elif best_val_acc > 50:
        print("⚠️ Model shows moderate performance. Consider more data or training.")
    else:
        print("❌ Model shows poor performance. Check data quality and model architecture.")
    
    print("\n🎉 Robust training completed!")

if __name__ == "__main__":
    main()
