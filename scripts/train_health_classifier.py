"""
Health Classifier Training Script
=================================
Trains ResNet18 model for healthy vs rotten apple classification.
Isolated training that doesn't affect other models.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from model_manager import ModelManager, SafeTraining
import logging

class HealthDataset(Dataset):
    """Dataset for healthy vs rotten apple classification."""
    
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['healthy', 'rotten']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load samples
        self.samples = []
        health_dir = self.data_dir / 'health'
        
        if split in ['train', 'val']:
            health_dir = self.data_dir / f'health_{split}'
        
        if not health_dir.exists():
            health_dir = self.data_dir / 'health'  # Fallback to original
        
        for class_name in self.classes:
            class_dir = health_dir / class_name
            if class_dir.exists():
                # Check for nested structure (variety subfolders)
                variety_dirs = [d for d in class_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                
                if variety_dirs:
                    # Nested structure: health/healthy/Variety/image.jpg
                    for variety_dir in variety_dirs:
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                            for img_path in variety_dir.glob(ext):
                                self.samples.append((str(img_path), self.class_to_idx[class_name]))
                else:
                    # Flat structure: health/healthy/image.jpg
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                        for img_path in class_dir.glob(ext):
                            self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
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

class HealthClassifierTrainer:
    """
    Isolated trainer for health classification model.
    """
    
    def __init__(self, data_dir="data", models_dir="models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.model_manager = ModelManager(models_dir)
        self.logger = self._setup_logger()
        
        # Training configuration
        self.model_name = "health_classifier.pt"
        self.epochs = 50
        self.batch_size = 32
        self.learning_rate = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _setup_logger(self):
        """Setup logger for health classifier training."""
        logger = logging.getLogger('HealthClassifierTrainer')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def get_transforms(self):
        """Get training and validation transforms."""
        
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_model(self):
        """Create ResNet18 model for health classification."""
        
        # Load pre-trained ResNet18
        model = models.resnet18(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in list(model.parameters())[:-10]:  # Freeze all but last 10 parameters
            param.requires_grad = False
        
        # Modify final layer for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)  # healthy, rotten
        )
        
        return model
    
    def prepare_data(self):
        """Prepare training and validation data loaders."""
        
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = HealthDataset(self.data_dir, transform=train_transform, split='train')
        val_dataset = HealthDataset(self.data_dir, transform=val_transform, split='val')
        
        # If no split datasets exist, create them from main dataset
        if len(train_dataset) == 0 and len(val_dataset) == 0:
            self.logger.info("No train/val split found. Using main dataset...")
            
            full_dataset = HealthDataset(self.data_dir, transform=train_transform)
            
            if len(full_dataset) == 0:
                self.logger.error("❌ No health classification data found!")
                return None, None
            
            # Split dataset 80/20
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
            # Apply validation transform to validation set
            val_dataset.dataset.transform = val_transform
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        if len(train_dataset) == 0:
            self.logger.error("❌ No training data available!")
            return None, None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def train_model(self, model, train_loader, val_loader):
        """Train the health classification model."""
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        self.logger.info(f"Training on device: {self.device}")
        model.to(self.device)
        
        for epoch in range(self.epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            self.logger.info("-" * 30)
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc="Training", leave=False)
            for images, labels in train_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation", leave=False)
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_train_acc = 100. * train_correct / train_total
            epoch_val_acc = 100. * val_correct / val_total
            
            # Store history
            history['train_losses'].append(epoch_train_loss)
            history['val_losses'].append(epoch_val_loss)
            history['train_accuracies'].append(epoch_train_acc)
            history['val_accuracies'].append(epoch_val_acc)
            
            # Print epoch results
            self.logger.info(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
            self.logger.info(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
            
            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_model_state = model.state_dict().copy()
                self.logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
            
            scheduler.step()
        
        # Load best model
        model.load_state_dict(best_model_state)
        history['best_val_acc'] = best_val_acc
        
        return model, history
    
    def evaluate_model(self, model, val_loader):
        """Evaluate model and generate classification report."""
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Classification report
        classes = ['healthy', 'rotten']
        report = classification_report(all_labels, all_predictions, target_names=classes)
        self.logger.info("\nClassification Report:")
        self.logger.info(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Health Classification - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = self.models_dir / 'health_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return report, cm
    
    def train_health_classifier(self):
        """Main training function."""
        
        self.logger.info("💚 Starting health classifier training...")
        
        try:
            with SafeTraining(self.model_manager, self.model_name):
                
                # Prepare data
                train_loader, val_loader = self.prepare_data()
                if train_loader is None:
                    return False
                
                # Create model
                model = self.create_model()
                
                # Train model
                trained_model, history = self.train_model(model, train_loader, val_loader)
                
                # Evaluate model
                report, cm = self.evaluate_model(trained_model, val_loader)
                
                # Save model with metadata
                model_info = {
                    'model_type': 'ResNet18_health_classifier',
                    'classes': ['healthy', 'rotten'],
                    'epochs_trained': self.epochs,
                    'best_val_accuracy': history['best_val_acc'],
                    'training_samples': len(train_loader.dataset),
                    'validation_samples': len(val_loader.dataset)
                }
                
                success = self.model_manager.save_model_safely(
                    trained_model, 
                    self.model_name, 
                    model_info
                )
                
                if success:
                    # Save training history
                    history_path = self.models_dir / 'health_training_history.json'
                    with open(history_path, 'w') as f:
                        json.dump(history, f, indent=2)
                    
                    self.logger.info("✅ Health classifier training completed successfully!")
                    self.logger.info(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
                    return True
                else:
                    self.logger.error("❌ Failed to save trained model")
                    return False
        
        except Exception as e:
            self.logger.error(f"❌ Health classifier training failed: {e}")
            return False

def main():
    """Main function to train health classifier."""
    print("💚 Health Classifier Training")
    print("=" * 30)
    
    # Check if already training
    model_manager = ModelManager()
    if model_manager.is_model_training("health_classifier.pt"):
        print("⚠️ Health classifier is already being trained!")
        return
    
    # Create trainer
    trainer = HealthClassifierTrainer()
    
    # Train model
    success = trainer.train_health_classifier()
    
    if success:
        print("🎉 Health classifier training completed!")
    else:
        print("❌ Health classifier training failed!")
        
        # Check if rollback is needed
        if input("Would you like to rollback to previous version? (y/n): ").lower() == 'y':
            model_manager.rollback_model("health_classifier.pt")

if __name__ == "__main__":
    main()
