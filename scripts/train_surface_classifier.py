"""
Surface Classifier Training Script
==================================
Trains hybrid CNN + texture features model for waxed vs unwaxed classification.
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
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from model_manager import ModelManager, SafeTraining
import logging

class SurfaceDataset(Dataset):
    """Dataset for waxed vs unwaxed apple classification with texture features."""
    
    def __init__(self, data_dir, transform=None, split='train', include_texture=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.include_texture = include_texture
        self.classes = ['unwaxed', 'waxed']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load samples
        self.samples = []
        surface_dir = self.data_dir / 'surface'
        
        if split in ['train', 'val']:
            surface_dir = self.data_dir / f'surface_{split}'
        
        if not surface_dir.exists():
            surface_dir = self.data_dir / 'surface'  # Fallback to original
        
        for class_name in self.classes:
            class_dir = surface_dir / class_name
            if class_dir.exists():
                # Check for nested structure (variety subfolders)
                variety_dirs = [d for d in class_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                
                if variety_dirs:
                    # Nested structure: surface/waxed/Variety/image.jpg
                    for variety_dir in variety_dirs:
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                            for img_path in variety_dir.glob(ext):
                                self.samples.append((str(img_path), self.class_to_idx[class_name]))
                else:
                    # Flat structure: surface/waxed/image.jpg
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                        for img_path in class_dir.glob(ext):
                            self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def extract_texture_features(self, image_path):
        """Extract LBP and GLCM texture features."""
        try:
            # Load image in grayscale
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros(16)  # Return dummy features if image can't be loaded
            
            # Resize for consistency
            img = cv2.resize(img, (224, 224))
            
            # Local Binary Pattern (LBP)
            def local_binary_pattern(image, P=8, R=1):
                """Simple LBP implementation."""
                rows, cols = image.shape
                lbp = np.zeros_like(image)
                
                for i in range(R, rows - R):
                    for j in range(R, cols - R):
                        center = image[i, j]
                        code = 0
                        
                        # Compare with neighbors
                        for p in range(P):
                            angle = 2 * np.pi * p / P
                            x = int(i + R * np.cos(angle))
                            y = int(j + R * np.sin(angle))
                            
                            if 0 <= x < rows and 0 <= y < cols:
                                if image[x, y] >= center:
                                    code |= (1 << p)
                        
                        lbp[i, j] = code
                
                return lbp
            
            # Calculate LBP
            lbp = local_binary_pattern(img)
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
            
            # Take first 8 LBP features
            lbp_features = lbp_hist[:8]
            
            # Gray Level Co-occurrence Matrix (GLCM) features
            def glcm_features(image, distance=1, angle=0):
                """Simple GLCM features."""
                # Quantize image to reduce computation
                levels = 32
                image = (image / 256.0 * levels).astype(np.uint8)
                
                rows, cols = image.shape
                glcm = np.zeros((levels, levels))
                
                # Calculate co-occurrence matrix
                dx = int(distance * np.cos(angle))
                dy = int(distance * np.sin(angle))
                
                for i in range(rows):
                    for j in range(cols):
                        if 0 <= i + dx < rows and 0 <= j + dy < cols:
                            glcm[image[i, j], image[i + dx, j + dy]] += 1
                
                # Normalize
                glcm = glcm.astype(float)
                glcm /= (glcm.sum() + 1e-7)
                
                # Calculate features
                contrast = 0
                dissimilarity = 0
                homogeneity = 0
                energy = 0
                correlation = 0
                
                mean_i = np.sum(np.arange(levels) * glcm.sum(axis=1))
                mean_j = np.sum(np.arange(levels) * glcm.sum(axis=0))
                std_i = np.sqrt(np.sum((np.arange(levels) - mean_i)**2 * glcm.sum(axis=1)))
                std_j = np.sqrt(np.sum((np.arange(levels) - mean_j)**2 * glcm.sum(axis=0)))
                
                for i in range(levels):
                    for j in range(levels):
                        contrast += glcm[i, j] * (i - j)**2
                        dissimilarity += glcm[i, j] * abs(i - j)
                        homogeneity += glcm[i, j] / (1 + (i - j)**2)
                        energy += glcm[i, j]**2
                        
                        if std_i > 0 and std_j > 0:
                            correlation += glcm[i, j] * (i - mean_i) * (j - mean_j) / (std_i * std_j)
                
                return [contrast, dissimilarity, homogeneity, energy, correlation]
            
            # Calculate GLCM features for different angles
            glcm_0 = glcm_features(img, angle=0)
            glcm_90 = glcm_features(img, angle=np.pi/2)
            
            # Combine features
            texture_features = np.concatenate([lbp_features, glcm_0[:3]])  # 8 + 3 = 11 features
            
            # Add some basic statistical features
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            skewness = np.mean(((img - mean_intensity) / std_intensity)**3) if std_intensity > 0 else 0
            
            # Combine all features (11 + 3 = 14 features, pad to 16)
            all_features = np.concatenate([texture_features, [mean_intensity/255.0, std_intensity/255.0, skewness]])
            all_features = np.pad(all_features, (0, max(0, 16 - len(all_features))), 'constant')[:16]
            
            return all_features.astype(np.float32)
        
        except Exception as e:
            # Return dummy features if extraction fails
            return np.zeros(16, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Extract texture features if requested
            if self.include_texture:
                texture_features = self.extract_texture_features(img_path)
                return image, texture_features, label
            else:
                return image, label
        
        except Exception as e:
            # Return dummy data if loading fails
            dummy_image = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            
            if self.include_texture:
                dummy_texture = np.zeros(16, dtype=np.float32)
                return dummy_image, dummy_texture, label
            else:
                return dummy_image, label

class HybridSurfaceModel(nn.Module):
    """Hybrid model combining CNN features with texture features."""
    
    def __init__(self, num_classes=2, texture_features=16):
        super(HybridSurfaceModel, self).__init__()
        
        # CNN backbone (ResNet18)
        self.backbone = models.resnet18(pretrained=True)
        cnn_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final layer
        
        # Texture feature processor
        self.texture_processor = nn.Sequential(
            nn.Linear(texture_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combined classifier
        combined_features = cnn_features + 32
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image, texture_features):
        # CNN features
        cnn_feat = self.backbone(image)
        
        # Texture features
        texture_feat = self.texture_processor(texture_features)
        
        # Combine features
        combined = torch.cat([cnn_feat, texture_feat], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        
        return output

class SurfaceClassifierTrainer:
    """
    Isolated trainer for surface classification model.
    """
    
    def __init__(self, data_dir="data", models_dir="models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.model_manager = ModelManager(models_dir)
        self.logger = self._setup_logger()
        
        # Training configuration
        self.model_name = "surface_classifier.pt"
        self.epochs = 40
        self.batch_size = 16  # Smaller batch size due to texture feature extraction
        self.learning_rate = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _setup_logger(self):
        """Setup logger for surface classifier training."""
        logger = logging.getLogger('SurfaceClassifierTrainer')
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
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def prepare_data(self):
        """Prepare training and validation data loaders."""
        
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = SurfaceDataset(self.data_dir, transform=train_transform, split='train')
        val_dataset = SurfaceDataset(self.data_dir, transform=val_transform, split='val')
        
        # If no split datasets exist, create them from main dataset
        if len(train_dataset) == 0 and len(val_dataset) == 0:
            self.logger.info("No train/val split found. Using main dataset...")
            
            full_dataset = SurfaceDataset(self.data_dir, transform=train_transform)
            
            if len(full_dataset) == 0:
                self.logger.error("❌ No surface classification data found!")
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
        
        # Custom collate function for hybrid data
        def collate_fn(batch):
            images, textures, labels = zip(*batch)
            images = torch.stack(images)
            textures = torch.stack([torch.from_numpy(t) for t in textures])
            labels = torch.tensor(labels)
            return images, textures, labels
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def train_model(self, model, train_loader, val_loader):
        """Train the surface classification model."""
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
        
        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        self.logger.info(f"Training hybrid model on device: {self.device}")
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
            for images, textures, labels in train_pbar:
                images = images.to(self.device)
                textures = textures.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images, textures)
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
                for images, textures, labels in val_pbar:
                    images = images.to(self.device)
                    textures = textures.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(images, textures)
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
    
    def train_surface_classifier(self):
        """Main training function."""
        
        self.logger.info("🧴 Starting surface classifier training...")
        
        try:
            with SafeTraining(self.model_manager, self.model_name):
                
                # Prepare data
                train_loader, val_loader = self.prepare_data()
                if train_loader is None:
                    return False
                
                # Create hybrid model
                model = HybridSurfaceModel(num_classes=2, texture_features=16)
                
                # Train model
                trained_model, history = self.train_model(model, train_loader, val_loader)
                
                # Save model with metadata
                model_info = {
                    'model_type': 'Hybrid_CNN_Texture_surface_classifier',
                    'classes': ['unwaxed', 'waxed'],
                    'epochs_trained': self.epochs,
                    'best_val_accuracy': history['best_val_acc'],
                    'training_samples': len(train_loader.dataset),
                    'validation_samples': len(val_loader.dataset),
                    'features': 'CNN + LBP + GLCM texture features'
                }
                
                success = self.model_manager.save_model_safely(
                    trained_model, 
                    self.model_name, 
                    model_info
                )
                
                if success:
                    # Save training history
                    history_path = self.models_dir / 'surface_training_history.json'
                    with open(history_path, 'w') as f:
                        json.dump(history, f, indent=2)
                    
                    self.logger.info("✅ Surface classifier training completed successfully!")
                    self.logger.info(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
                    return True
                else:
                    self.logger.error("❌ Failed to save trained model")
                    return False
        
        except Exception as e:
            self.logger.error(f"❌ Surface classifier training failed: {e}")
            return False

def main():
    """Main function to train surface classifier."""
    print("🧴 Surface Classifier Training")
    print("=" * 30)
    
    # Check if already training
    model_manager = ModelManager()
    if model_manager.is_model_training("surface_classifier.pt"):
        print("⚠️ Surface classifier is already being trained!")
        return
    
    # Create trainer
    trainer = SurfaceClassifierTrainer()
    
    # Train model
    success = trainer.train_surface_classifier()
    
    if success:
        print("🎉 Surface classifier training completed!")
    else:
        print("❌ Surface classifier training failed!")
        
        # Check if rollback is needed
        if input("Would you like to rollback to previous version? (y/n): ").lower() == 'y':
            model_manager.rollback_model("surface_classifier.pt")

if __name__ == "__main__":
    main()
