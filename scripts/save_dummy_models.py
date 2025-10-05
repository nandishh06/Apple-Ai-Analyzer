"""
Dummy Models Generator for Indian Apple Intelligence System
==========================================================
Creates placeholder model files for development and testing.
Replace with real trained models when available.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import numpy as np
from ultralytics import YOLO

def create_models_directory():
    """Create models directory if it doesn't exist."""
    models_dir = "../models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ Created directory: {models_dir}")
    return models_dir

def create_dummy_apple_detector(models_dir):
    """Create dummy YOLOv8 apple detector."""
    try:
        print("🔍 Creating dummy apple detector...")
        
        # Use YOLOv8 nano model as base
        model = YOLO('yolov8n.pt')  # This will download if not present
        
        # Save as apple detector
        detector_path = os.path.join(models_dir, 'apple_detector.pt')
        model.save(detector_path)
        
        print(f"✅ Dummy apple detector saved: {detector_path}")
        
    except Exception as e:
        print(f"⚠️ Could not create YOLOv8 detector: {e}")
        print("   This is normal if ultralytics is not installed yet.")

def create_dummy_variety_classifier(models_dir):
    """Create dummy EfficientNet-B3 variety classifier."""
    try:
        print("🍎 Creating dummy variety classifier...")
        
        # Create EfficientNet-B3 based model
        model = models.efficientnet_b3(pretrained=False)
        
        # Modify final layer for 6 apple varieties
        num_varieties = 6  # Sharbati, Sunehari, Maharaji, Splendour, Himsona, Himkiran
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, num_varieties)
        )
        
        # Set to evaluation mode
        model.eval()
        
        # Save model
        variety_path = os.path.join(models_dir, 'variety_classifier.pt')
        torch.save(model, variety_path)
        
        print(f"✅ Dummy variety classifier saved: {variety_path}")
        print(f"   Model: EfficientNet-B3 with {num_varieties} output classes")
        
    except Exception as e:
        print(f"❌ Error creating variety classifier: {e}")

def create_dummy_health_classifier(models_dir):
    """Create dummy ResNet18 health classifier."""
    try:
        print("💚 Creating dummy health classifier...")
        
        # Create ResNet18 based model
        model = models.resnet18(pretrained=False)
        
        # Modify final layer for 2 classes (Healthy, Rotten)
        num_classes = 2
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Set to evaluation mode
        model.eval()
        
        # Save model
        health_path = os.path.join(models_dir, 'health_classifier.pt')
        torch.save(model, health_path)
        
        print(f"✅ Dummy health classifier saved: {health_path}")
        print(f"   Model: ResNet18 with {num_classes} output classes (Healthy/Rotten)")
        
    except Exception as e:
        print(f"❌ Error creating health classifier: {e}")

def create_dummy_surface_classifier(models_dir):
    """Create dummy surface classifier (CNN + texture features)."""
    try:
        print("🧴 Creating dummy surface classifier...")
        
        # Create ResNet18 based model for surface classification
        model = models.resnet18(pretrained=False)
        
        # Modify final layer for 2 classes (Waxed, Unwaxed)
        num_classes = 2
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Set to evaluation mode
        model.eval()
        
        # Save model
        surface_path = os.path.join(models_dir, 'surface_classifier.pt')
        torch.save(model, surface_path)
        
        print(f"✅ Dummy surface classifier saved: {surface_path}")
        print(f"   Model: ResNet18 with {num_classes} output classes (Waxed/Unwaxed)")
        
    except Exception as e:
        print(f"❌ Error creating surface classifier: {e}")

def create_dummy_shelf_life_predictor(models_dir):
    """Create dummy Random Forest shelf life predictor."""
    try:
        print("⏳ Creating dummy shelf life predictor...")
        
        # Create dummy training data
        # Features: [variety_encoded, health_encoded, surface_encoded]
        # variety_encoded: 0-5 (6 varieties)
        # health_encoded: 0=Rotten, 1=Healthy
        # surface_encoded: 0=Unwaxed, 1=Waxed
        
        np.random.seed(42)  # For reproducible dummy data
        n_samples = 1000
        
        X = np.random.randint(0, 6, (n_samples, 1))  # Variety
        X = np.column_stack([
            X,
            np.random.randint(0, 2, n_samples),  # Health
            np.random.randint(0, 2, n_samples)   # Surface
        ])
        
        # Generate realistic shelf life targets
        base_shelf_life = [15, 20, 18, 22, 18, 16]  # Per variety
        y = []
        
        for i in range(n_samples):
            variety_idx = X[i, 0]
            is_healthy = X[i, 1]
            is_waxed = X[i, 2]
            
            if is_healthy == 0:  # Rotten
                shelf_life = 0
            else:
                base = base_shelf_life[variety_idx]
                if is_waxed:
                    shelf_life = base * 1.5  # Waxed apples last longer
                else:
                    shelf_life = base
                
                # Add some noise
                shelf_life += np.random.normal(0, 2)
                shelf_life = max(0, shelf_life)  # No negative shelf life
            
            y.append(shelf_life)
        
        y = np.array(y)
        
        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        model.fit(X, y)
        
        # Save model
        shelf_path = os.path.join(models_dir, 'shelf_life_model.pkl')
        joblib.dump(model, shelf_path)
        
        print(f"✅ Dummy shelf life predictor saved: {shelf_path}")
        print(f"   Model: Random Forest with {n_samples} training samples")
        print(f"   Features: variety_encoded, health_encoded, surface_encoded")
        
        # Test the model
        test_sample = np.array([[2, 1, 1]])  # Maharaji, Healthy, Waxed
        prediction = model.predict(test_sample)[0]
        print(f"   Test prediction: {prediction:.1f} days for Maharaji (Healthy, Waxed)")
        
    except Exception as e:
        print(f"❌ Error creating shelf life predictor: {e}")

def create_data_directories():
    """Create data directories for training datasets."""
    data_dirs = [
        "../data/varieties/Sharbati",
        "../data/varieties/Sunehari", 
        "../data/varieties/Maharaji",
        "../data/varieties/Splendour",
        "../data/varieties/Himsona",
        "../data/varieties/Himkiran",
        "../data/health/healthy",
        "../data/health/rotten",
        "../data/surface/waxed",
        "../data/surface/unwaxed"
    ]
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    print("✅ Created data directory structure")
    print("   Place your training images in the respective folders:")
    for data_dir in data_dirs:
        print(f"   - {data_dir}/")

def main():
    """Main function to create all dummy models."""
    print("🏗️ Creating Dummy Models for Indian Apple Intelligence System")
    print("=" * 60)
    
    # Create directories
    models_dir = create_models_directory()
    create_data_directories()
    
    print("\n📦 Generating Model Files...")
    print("-" * 30)
    
    # Create all dummy models
    create_dummy_apple_detector(models_dir)
    create_dummy_variety_classifier(models_dir)
    create_dummy_health_classifier(models_dir)
    create_dummy_surface_classifier(models_dir)
    create_dummy_shelf_life_predictor(models_dir)
    
    print("\n" + "=" * 60)
    print("✅ All dummy models created successfully!")
    print("\n📝 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the app: python app/app.py")
    print("3. Replace dummy models with real trained models when ready")
    print("\n🔧 Model Replacement Guide:")
    print("- apple_detector.pt: Train YOLOv8 on apple detection dataset")
    print("- variety_classifier.pt: Train EfficientNet-B3 on 6 apple varieties")
    print("- health_classifier.pt: Train ResNet18 on healthy/rotten apples")
    print("- surface_classifier.pt: Train CNN on waxed/unwaxed apples")
    print("- shelf_life_model.pkl: Train Random Forest on shelf life data")

if __name__ == "__main__":
    main()
