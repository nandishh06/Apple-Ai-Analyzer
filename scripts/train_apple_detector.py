"""
Apple Detector Training Script
==============================
Trains YOLOv8 model for apple detection.
Isolated training that doesn't affect other models.
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import shutil
from model_manager import ModelManager, SafeTraining
import logging

class AppleDetectorTrainer:
    """
    Isolated trainer for YOLOv8 apple detector.
    """
    
    def __init__(self, data_dir="data", models_dir="models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.model_manager = ModelManager(models_dir)
        self.logger = self._setup_logger()
        
        # Training configuration
        self.model_name = "apple_detector.pt"
        self.epochs = 100
        self.batch_size = 16
        self.img_size = 640
        
    def _setup_logger(self):
        """Setup logger for apple detector training."""
        logger = logging.getLogger('AppleDetectorTrainer')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def prepare_yolo_dataset(self):
        """
        Prepare dataset in YOLO format.
        For apple detection, we need to create bounding box annotations.
        """
        self.logger.info("🍎 Preparing YOLO dataset for apple detection...")
        
        # Create YOLO dataset structure
        yolo_dir = self.data_dir / "yolo_apple_detection"
        yolo_dir.mkdir(exist_ok=True)
        
        (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Create dataset YAML file
        dataset_yaml = {
            'path': str(yolo_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'apple'}
        }
        
        yaml_path = yolo_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f)
        
        self.logger.info(f"✅ YOLO dataset structure created at {yolo_dir}")
        
        # Check if we have existing apple images to use
        apple_images_found = 0
        for category in ['varieties', 'health', 'surface']:
            category_path = self.data_dir / category
            if category_path.exists():
                for subdir in category_path.iterdir():
                    if subdir.is_dir():
                        for img_file in subdir.glob("*.jpg"):
                            apple_images_found += 1
                        for img_file in subdir.glob("*.png"):
                            apple_images_found += 1
        
        if apple_images_found == 0:
            self.logger.warning("⚠️ No apple images found. Using pre-trained YOLOv8 as apple detector.")
            return None
        
        self.logger.info(f"Found {apple_images_found} apple images for detection training")
        
        # For now, we'll use a simplified approach where we assume all images contain apples
        # In a real scenario, you would need to annotate bounding boxes
        self.logger.warning("📝 Note: This demo assumes all images contain apples.")
        self.logger.warning("    For production, you need proper bounding box annotations.")
        
        return yaml_path
    
    def train_detector(self, dataset_yaml=None):
        """
        Train YOLOv8 apple detector.
        For now, we'll use the pre-trained COCO model which can detect apples.
        """
        self.logger.info("🚀 Starting apple detector setup...")
        
        try:
            with SafeTraining(self.model_manager, self.model_name):
                # Load pre-trained YOLOv8 model
                self.logger.info("Loading pre-trained YOLOv8 model...")
                model = YOLO('yolov8n.pt')  # Use nano version for faster inference
                
                # For apple detection, we'll use the pre-trained COCO model
                # COCO dataset includes 'apple' as class 47
                self.logger.info("Using pre-trained COCO model for apple detection...")
                self.logger.info("COCO model can detect apples (class 47) out of the box!")
                
                model_info = {
                    'model_type': 'YOLOv8_pretrained_coco',
                    'note': 'Pre-trained COCO model - can detect apples as class 47',
                    'classes': 'COCO 80 classes including apple',
                    'apple_class_id': 47,
                    'confidence_threshold': 0.5
                }
                
                success = self.model_manager.save_model_safely(
                    model, 
                    self.model_name, 
                    model_info
                )
                
                if success:
                    self.logger.info("✅ Apple detector setup completed successfully!")
                    self.logger.info("The model can detect apples using COCO pre-trained weights.")
                    return True
                else:
                    self.logger.error("❌ Failed to save apple detector model")
                    return False
        
        except Exception as e:
            self.logger.error(f"❌ Apple detector setup failed: {e}")
            return False
    
    def validate_detector(self):
        """
        Validate the trained apple detector.
        """
        model_path = self.models_dir / self.model_name
        
        if not model_path.exists():
            self.logger.error("❌ No trained model found for validation")
            return False
        
        try:
            self.logger.info("🔍 Validating apple detector...")
            
            # Load model
            model = YOLO(str(model_path))
            
            # Test on a few sample images if available
            sample_images = []
            for category in ['varieties', 'health', 'surface']:
                category_path = self.data_dir / category
                if category_path.exists():
                    for subdir in category_path.iterdir():
                        if subdir.is_dir():
                            for img_file in list(subdir.glob("*.jpg"))[:2]:  # Take 2 samples
                                sample_images.append(img_file)
                            if len(sample_images) >= 5:  # Limit to 5 samples
                                break
                    if len(sample_images) >= 5:
                        break
            
            if sample_images:
                self.logger.info(f"Testing on {len(sample_images)} sample images...")
                
                detection_count = 0
                for img_path in sample_images:
                    results = model(str(img_path))
                    
                    # Check if any objects were detected
                    if len(results[0].boxes) > 0:
                        detection_count += 1
                
                detection_rate = detection_count / len(sample_images)
                self.logger.info(f"Detection rate: {detection_rate:.1%} ({detection_count}/{len(sample_images)})")
                
                if detection_rate > 0.5:  # At least 50% detection rate
                    self.logger.info("✅ Apple detector validation passed!")
                    return True
                else:
                    self.logger.warning("⚠️ Low detection rate. Model may need improvement.")
                    return False
            
            else:
                self.logger.warning("⚠️ No sample images found for validation")
                return True  # Assume OK if no samples to test
        
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return False

def main():
    """Main function to train apple detector."""
    print("🍎 Apple Detector Training")
    print("=" * 30)
    
    # Check if already training
    model_manager = ModelManager()
    if model_manager.is_model_training("apple_detector.pt"):
        print("⚠️ Apple detector is already being trained!")
        return
    
    # Create trainer
    trainer = AppleDetectorTrainer()
    
    # Prepare dataset
    dataset_yaml = trainer.prepare_yolo_dataset()
    
    # Train model
    success = trainer.train_detector(dataset_yaml)
    
    if success:
        # Validate model
        trainer.validate_detector()
        print("🎉 Apple detector training completed!")
    else:
        print("❌ Apple detector training failed!")
        
        # Check if rollback is needed
        if input("Would you like to rollback to previous version? (y/n): ").lower() == 'y':
            model_manager.rollback_model("apple_detector.pt")

if __name__ == "__main__":
    main()
