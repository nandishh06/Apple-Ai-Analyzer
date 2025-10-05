"""
Fast Surface Classifier Training
================================
Trains robust surface classifier (waxed vs unwaxed) using the same approach as variety and health models.
Isolated training that doesn't affect other models.
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
import cv2

class SurfaceDataset(Dataset):
    """Dataset for waxed vs unwaxed apple classification."""
    
    def __init__(self, data_dir, transform=None, max_samples_per_class=500):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['waxed', 'unwaxed']
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
                
                # Also check subdirectories (variety folders)
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

def get_surface_transforms():
    """Get transforms optimized for surface classification."""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        # Enhanced for surface texture detection
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # Enhance texture details
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_surface_model(num_classes=2):
    """Create a robust model for surface classification."""
    
    # Use EfficientNet-B1 (good balance of accuracy and speed for texture analysis)
    model = models.efficientnet_b1(pretrained=True)
    
    # Freeze early layers but keep more layers trainable for texture detection
    for param in model.features[:-3].parameters():
        param.requires_grad = False
    
    # Replace classifier with enhanced architecture for texture analysis
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, num_classes)
    )
    
    return model

def train_surface_model(model, train_loader, val_loader, num_epochs=15, device='cpu'):
    """Train surface model with proper validation."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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
    patience = 6
    
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
            torch.save(model.state_dict(), 'models/best_surface_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(torch.load('models/best_surface_model.pth'))
    
    return model, history

def evaluate_surface_model(model, val_loader, classes, device='cpu'):
    """Evaluate the surface model and create confusion matrix."""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Surface Classification - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/surface_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return report, cm

def main():
    """Main training function."""
    
    print("🧴 Fast Surface Classifier Training")
    print("=" * 42)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Check data directory
    data_dir = "data/surface"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Expected structure:")
        print("  data/surface/waxed/")
        print("  data/surface/unwaxed/")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_surface_transforms()
    
    # Create full dataset
    print("\n📊 Loading surface dataset...")
    full_dataset = SurfaceDataset(data_dir, transform=train_transform, max_samples_per_class=500)
    
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
    val_dataset_proper = SurfaceDataset(data_dir, transform=val_transform, max_samples_per_class=500)
    val_dataset_proper.samples = [val_dataset_proper.samples[i] for i in val_indices if i < len(val_dataset_proper.samples)]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Create model
    model = create_surface_model(num_classes=2)
    print("Model created: EfficientNet-B1 with enhanced texture analysis for surface classification")
    
    # Train model
    print("\n🚀 Starting surface classifier training...")
    trained_model, history = train_surface_model(model, train_loader, val_loader, 
                                                num_epochs=15, device=device)
    
    # Evaluate model
    classes = ['waxed', 'unwaxed']
    report, cm = evaluate_surface_model(trained_model, val_loader, classes, device)
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save(trained_model, "models/surface_classifier.pt")
    print(f"\n✅ Surface model saved: models/surface_classifier.pt")
    
    # Save training history
    with open("models/surface_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save evaluation report
    with open("models/surface_evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print final results
    best_val_acc = max(history['val_accuracies'])
    final_val_acc = history['val_accuracies'][-1]
    
    print(f"\n📊 Surface Classification Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Waxed Precision: {report['waxed']['precision']:.3f}")
    print(f"Unwaxed Precision: {report['unwaxed']['precision']:.3f}")
    print(f"Overall F1-Score: {report['macro avg']['f1-score']:.3f}")
    
    if best_val_acc > 85:
        print("✅ Excellent surface classification performance!")
    elif best_val_acc > 75:
        print("✅ Good surface classification performance!")
    else:
        print("⚠️ Surface classification needs improvement. Consider more data or training.")
    
    print("\n🎉 Surface classifier training completed!")
    print("The model can now distinguish between waxed and unwaxed apples.")

if __name__ == "__main__":
    main()
