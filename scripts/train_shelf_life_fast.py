"""
Fast Shelf Life Predictor Training
==================================
Trains robust shelf life predictor using regression approach.
Predicts remaining shelf life in days based on variety, health, and surface.
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
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import random
from pathlib import Path

class ShelfLifeDataset(Dataset):
    """Dataset for shelf life prediction with multi-modal inputs."""
    
    def __init__(self, data_dir, transform=None, max_samples_per_variety=200):
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples_per_variety = max_samples_per_variety
        
        # Apple varieties and their base shelf life (days)
        self.variety_to_days = {
            'Sharbati': 15,
            'Sunehari': 20,
            'Maharaji': 18,
            'Splendour': 22,
            'Himsona': 18,
            'Himkiran': 16
        }
        
        self.varieties = list(self.variety_to_days.keys())
        self.variety_to_idx = {v: i for i, v in enumerate(self.varieties)}
        
        # Health and surface factors
        self.health_factor = {'healthy': 1.0, 'rotten': 0.0}
        self.surface_factor = {'waxed': 1.3, 'unwaxed': 1.0}  # Waxed lasts 30% longer
        
        # Load samples
        self.samples = []
        self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic shelf life data based on variety, health, and surface."""
        
        print("📊 Generating synthetic shelf life dataset...")
        
        for variety in self.varieties:
            variety_samples = []
            base_days = self.variety_to_days[variety]
            
            # Generate samples for each combination
            for health in ['healthy', 'rotten']:
                for surface in ['waxed', 'unwaxed']:
                    # Calculate expected shelf life
                    health_mult = self.health_factor[health]
                    surface_mult = self.surface_factor[surface]
                    
                    if health == 'rotten':
                        expected_days = 0  # Rotten apples have 0 shelf life
                    else:
                        expected_days = int(base_days * surface_mult)
                    
                    # Generate multiple samples with variation
                    samples_per_combo = self.max_samples_per_variety // 8  # 8 combinations
                    
                    for _ in range(samples_per_combo):
                        # Add realistic variation
                        if expected_days > 0:
                            variation = random.randint(-3, 4)  # ±3 days variation
                            actual_days = max(0, expected_days + variation)
                        else:
                            actual_days = 0
                        
                        # Create sample
                        sample = {
                            'variety': variety,
                            'variety_idx': self.variety_to_idx[variety],
                            'health': health,
                            'health_idx': 0 if health == 'healthy' else 1,
                            'surface': surface,
                            'surface_idx': 0 if surface == 'waxed' else 1,
                            'shelf_life': actual_days,
                            'base_days': base_days
                        }
                        
                        variety_samples.append(sample)
            
            self.samples.extend(variety_samples)
            print(f"  {variety}: {len(variety_samples)} samples (base: {base_days} days)")
        
        print(f"Total synthetic samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Create feature vector: [variety_one_hot(6), health(1), surface(1), base_days(1)]
        variety_onehot = torch.zeros(6)
        variety_onehot[sample['variety_idx']] = 1.0
        
        health_val = torch.tensor([sample['health_idx']], dtype=torch.float32)
        surface_val = torch.tensor([sample['surface_idx']], dtype=torch.float32)
        base_days_val = torch.tensor([sample['base_days']], dtype=torch.float32)
        
        # Combine features
        features = torch.cat([variety_onehot, health_val, surface_val, base_days_val])
        
        # Target
        target = torch.tensor([sample['shelf_life']], dtype=torch.float32)
        
        return features, target

class ShelfLifePredictor(nn.Module):
    """Neural network for shelf life prediction."""
    
    def __init__(self, input_size=9):  # 6 variety + 1 health + 1 surface + 1 base_days
        super(ShelfLifePredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 1),
            nn.ReLU()  # Ensure positive output
        )
    
    def forward(self, x):
        return self.network(x)

def train_shelf_life_model(model, train_loader, val_loader, num_epochs=25, device='cpu'):
    """Train shelf life prediction model."""
    
    model = model.to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, verbose=True
    )
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_maes': [],
        'val_maes': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for features, targets in train_pbar:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
            
            # Calculate MAE
            mae = torch.mean(torch.abs(outputs - targets))
            running_mae += mae.item() * features.size(0)
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{mae.item():.2f} days'
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_mae = running_mae / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for features, targets in val_pbar:
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * features.size(0)
                mae = torch.mean(torch.abs(outputs - targets))
                val_mae += mae.item() * features.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_mae = val_mae / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        history['train_losses'].append(epoch_loss)
        history['val_losses'].append(val_loss)
        history['train_maes'].append(epoch_mae)
        history['val_maes'].append(val_mae)
        
        print(f'Train Loss: {epoch_loss:.4f} MAE: {epoch_mae:.2f} days')
        print(f'Val Loss: {val_loss:.4f} MAE: {val_mae:.2f} days')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_shelf_life_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(torch.load('models/best_shelf_life_model.pth'))
    
    return model, history

def evaluate_shelf_life_model(model, val_loader, device='cpu'):
    """Evaluate the shelf life model."""
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(all_targets, all_preds, alpha=0.6)
    plt.plot([0, max(all_targets)], [0, max(all_targets)], 'r--', lw=2)
    plt.xlabel('Actual Shelf Life (days)')
    plt.ylabel('Predicted Shelf Life (days)')
    plt.title('Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = np.array(all_preds) - np.array(all_targets)
    plt.scatter(all_targets, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Shelf Life (days)')
    plt.ylabel('Residuals (days)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/shelf_life_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }

def main():
    """Main training function."""
    
    print("⏰ Fast Shelf Life Predictor Training")
    print("=" * 44)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic dataset
    print("\n📊 Creating shelf life dataset...")
    full_dataset = ShelfLifeDataset("data/shelf_life", max_samples_per_variety=400)
    
    print(f"Total samples: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        print("❌ No training data generated.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # Create model
    model = ShelfLifePredictor(input_size=9)
    print("Model created: Neural Network for shelf life regression")
    
    # Train model
    print("\n🚀 Starting shelf life predictor training...")
    trained_model, history = train_shelf_life_model(model, train_loader, val_loader, 
                                                   num_epochs=25, device=device)
    
    # Evaluate model
    metrics = evaluate_shelf_life_model(trained_model, val_loader, device)
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save(trained_model, "models/shelf_life_predictor.pt")
    print(f"\n✅ Shelf life model saved: models/shelf_life_predictor.pt")
    
    # Save training history
    with open("models/shelf_life_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save evaluation metrics
    metrics_to_save = {k: v for k, v in metrics.items() if k not in ['predictions', 'targets']}
    with open("models/shelf_life_evaluation_report.json", 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Print final results
    best_val_mae = min(history['val_maes'])
    final_val_mae = history['val_maes'][-1]
    
    print(f"\n📊 Shelf Life Prediction Results:")
    print(f"Best Validation MAE: {best_val_mae:.2f} days")
    print(f"Final Validation MAE: {final_val_mae:.2f} days")
    print(f"RMSE: {metrics['rmse']:.2f} days")
    print(f"R² Score: {metrics['r2']:.3f}")
    
    if best_val_mae < 2.0:
        print("✅ Excellent shelf life prediction performance!")
    elif best_val_mae < 3.0:
        print("✅ Good shelf life prediction performance!")
    else:
        print("⚠️ Shelf life prediction needs improvement.")
    
    print("\n🎉 Shelf life predictor training completed!")
    print("The model can now predict remaining shelf life based on variety, health, and surface.")

if __name__ == "__main__":
    main()
