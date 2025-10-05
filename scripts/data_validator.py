"""
Data Validation and Preprocessing Pipeline
==========================================
Validates datasets and prepares them for training.
Ensures data quality and consistency across all models.
"""

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from collections import Counter
import logging
from typing import Dict, List, Tuple
import shutil

class DataValidator:
    """
    Validates and preprocesses datasets for apple analysis models.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logger()
        
        # Expected directory structure
        self.expected_structure = {
            'varieties': ['Sharbati', 'Sunehari', 'Maharaji', 'Splendour', 'Himsona', 'Himkiran'],
            'health': ['healthy', 'rotten'],
            'surface': ['waxed', 'unwaxed']
        }
        
        # Image validation settings
        self.min_image_size = (100, 100)
        self.max_image_size = (4000, 4000)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
    def _setup_logger(self):
        """Setup logger for data validation."""
        logger = logging.getLogger('DataValidator')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def validate_directory_structure(self) -> Dict[str, bool]:
        """
        Validate that all required directories exist.
        """
        self.logger.info("🔍 Validating directory structure...")
        
        results = {}
        
        for category, subcategories in self.expected_structure.items():
            category_path = self.data_dir / category
            results[category] = {}
            
            if not category_path.exists():
                self.logger.warning(f"❌ Missing category directory: {category}")
                results[category]['exists'] = False
                continue
            
            results[category]['exists'] = True
            results[category]['subcategories'] = {}
            
            for subcat in subcategories:
                subcat_path = category_path / subcat
                if subcat_path.exists():
                    results[category]['subcategories'][subcat] = True
                    self.logger.info(f"✅ Found: {category}/{subcat}")
                else:
                    results[category]['subcategories'][subcat] = False
                    self.logger.warning(f"❌ Missing: {category}/{subcat}")
        
        return results
    
    def validate_images(self, category: str) -> Dict[str, any]:
        """
        Validate images in a specific category.
        """
        self.logger.info(f"🖼️ Validating images in {category}...")
        
        category_path = self.data_dir / category
        if not category_path.exists():
            return {'error': f'Category {category} does not exist'}
        
        results = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'issues': [],
            'class_distribution': {},
            'image_stats': {
                'min_width': float('inf'),
                'max_width': 0,
                'min_height': float('inf'),
                'max_height': 0,
                'avg_width': 0,
                'avg_height': 0
            }
        }
        
        widths, heights = [], []
        
        for subcat in self.expected_structure[category]:
            subcat_path = category_path / subcat
            if not subcat_path.exists():
                continue
            
            image_files = []
            for ext in self.supported_formats:
                image_files.extend(subcat_path.glob(f"*{ext}"))
                image_files.extend(subcat_path.glob(f"*{ext.upper()}"))
            
            valid_count = 0
            
            for img_path in image_files:
                results['total_images'] += 1
                
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        
                        # Check image size
                        if (width < self.min_image_size[0] or 
                            height < self.min_image_size[1]):
                            results['issues'].append(f"Too small: {img_path.name} ({width}x{height})")
                            results['invalid_images'] += 1
                            continue
                        
                        if (width > self.max_image_size[0] or 
                            height > self.max_image_size[1]):
                            results['issues'].append(f"Too large: {img_path.name} ({width}x{height})")
                            results['invalid_images'] += 1
                            continue
                        
                        # Check if image can be loaded properly
                        img.verify()
                        
                        # Image is valid
                        valid_count += 1
                        results['valid_images'] += 1
                        
                        # Collect statistics
                        widths.append(width)
                        heights.append(height)
                        
                        # Update min/max
                        results['image_stats']['min_width'] = min(results['image_stats']['min_width'], width)
                        results['image_stats']['max_width'] = max(results['image_stats']['max_width'], width)
                        results['image_stats']['min_height'] = min(results['image_stats']['min_height'], height)
                        results['image_stats']['max_height'] = max(results['image_stats']['max_height'], height)
                
                except Exception as e:
                    results['issues'].append(f"Corrupted: {img_path.name} - {str(e)}")
                    results['invalid_images'] += 1
            
            results['class_distribution'][subcat] = valid_count
            self.logger.info(f"  {subcat}: {valid_count} valid images")
        
        # Calculate averages
        if widths:
            results['image_stats']['avg_width'] = np.mean(widths)
            results['image_stats']['avg_height'] = np.mean(heights)
        else:
            results['image_stats'] = {k: 0 for k in results['image_stats']}
        
        return results
    
    def check_class_balance(self, category: str) -> Dict[str, any]:
        """
        Check class balance and recommend data augmentation if needed.
        """
        validation_results = self.validate_images(category)
        
        if 'class_distribution' not in validation_results:
            return {'error': 'Could not analyze class distribution'}
        
        distribution = validation_results['class_distribution']
        total_images = sum(distribution.values())
        
        if total_images == 0:
            return {'error': 'No valid images found'}
        
        # Calculate balance metrics
        class_percentages = {k: (v/total_images)*100 for k, v in distribution.items()}
        
        # Check for imbalance (if any class has < 15% or > 60% of data)
        imbalanced_classes = []
        for class_name, percentage in class_percentages.items():
            if percentage < 15 or percentage > 60:
                imbalanced_classes.append((class_name, percentage))
        
        recommendations = []
        
        if imbalanced_classes:
            recommendations.append("⚠️ Class imbalance detected!")
            for class_name, percentage in imbalanced_classes:
                if percentage < 15:
                    recommendations.append(f"  - {class_name}: Only {percentage:.1f}% of data. Consider data augmentation.")
                else:
                    recommendations.append(f"  - {class_name}: {percentage:.1f}% of data. Consider reducing samples.")
        else:
            recommendations.append("✅ Classes are reasonably balanced.")
        
        return {
            'distribution': distribution,
            'percentages': class_percentages,
            'total_images': total_images,
            'is_balanced': len(imbalanced_classes) == 0,
            'recommendations': recommendations
        }
    
    def create_train_val_split(self, category: str, train_ratio: float = 0.8, 
                              val_ratio: float = 0.2) -> bool:
        """
        Create train/validation split for a category.
        """
        if abs(train_ratio + val_ratio - 1.0) > 0.001:
            self.logger.error("Train and validation ratios must sum to 1.0")
            return False
        
        self.logger.info(f"📊 Creating train/val split for {category} ({train_ratio:.1%}/{val_ratio:.1%})")
        
        category_path = self.data_dir / category
        train_path = self.data_dir / f"{category}_train"
        val_path = self.data_dir / f"{category}_val"
        
        # Create split directories
        train_path.mkdir(exist_ok=True)
        val_path.mkdir(exist_ok=True)
        
        for subcat in self.expected_structure[category]:
            subcat_path = category_path / subcat
            if not subcat_path.exists():
                continue
            
            # Create subdirectories
            (train_path / subcat).mkdir(exist_ok=True)
            (val_path / subcat).mkdir(exist_ok=True)
            
            # Get all valid images
            image_files = []
            for ext in self.supported_formats:
                image_files.extend(subcat_path.glob(f"*{ext}"))
                image_files.extend(subcat_path.glob(f"*{ext.upper()}"))
            
            # Shuffle and split
            np.random.shuffle(image_files)
            n_train = int(len(image_files) * train_ratio)
            
            train_files = image_files[:n_train]
            val_files = image_files[n_train:]
            
            # Copy files
            for img_file in train_files:
                shutil.copy2(img_file, train_path / subcat / img_file.name)
            
            for img_file in val_files:
                shutil.copy2(img_file, val_path / subcat / img_file.name)
            
            self.logger.info(f"  {subcat}: {len(train_files)} train, {len(val_files)} val")
        
        return True
    
    def generate_data_report(self) -> Dict[str, any]:
        """
        Generate comprehensive data validation report.
        """
        self.logger.info("📋 Generating comprehensive data report...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'directory_structure': self.validate_directory_structure(),
            'categories': {}
        }
        
        for category in self.expected_structure.keys():
            self.logger.info(f"Analyzing {category}...")
            
            report['categories'][category] = {
                'validation': self.validate_images(category),
                'balance': self.check_class_balance(category)
            }
        
        # Overall statistics
        total_images = sum(
            report['categories'][cat]['validation']['valid_images'] 
            for cat in report['categories']
        )
        
        report['summary'] = {
            'total_valid_images': total_images,
            'categories_analyzed': len(report['categories']),
            'ready_for_training': total_images > 0
        }
        
        return report
    
    def save_report(self, report: Dict, filename: str = "data_validation_report.json"):
        """Save validation report to file."""
        report_path = self.data_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📄 Report saved to {report_path}")
    
    def create_data_directories(self):
        """Create all required data directories."""
        self.logger.info("📁 Creating data directory structure...")
        
        for category, subcategories in self.expected_structure.items():
            category_path = self.data_dir / category
            category_path.mkdir(exist_ok=True)
            
            for subcat in subcategories:
                subcat_path = category_path / subcat
                subcat_path.mkdir(exist_ok=True)
                
                # Create placeholder README
                readme_path = subcat_path / "README.md"
                if not readme_path.exists():
                    with open(readme_path, 'w') as f:
                        f.write(f"# {category.title()} - {subcat.title()}\n\n")
                        f.write(f"Place {subcat} {category} images here.\n\n")
                        f.write("Supported formats: JPG, JPEG, PNG, BMP, TIFF\n")
                        f.write(f"Recommended size: 224x224 to 1024x1024 pixels\n")
        
        self.logger.info("✅ Data directories created successfully!")

def main():
    """Main function to run data validation."""
    print("🔍 Apple Dataset Validation")
    print("=" * 40)
    
    # Create validator
    validator = DataValidator()
    
    # Create directories if they don't exist
    validator.create_data_directories()
    
    # Generate and save report
    report = validator.generate_data_report()
    validator.save_report(report)
    
    # Print summary
    print("\n📊 Validation Summary:")
    print("-" * 20)
    
    for category, data in report['categories'].items():
        validation = data['validation']
        balance = data['balance']
        
        print(f"\n{category.upper()}:")
        print(f"  Valid images: {validation['valid_images']}")
        print(f"  Invalid images: {validation['invalid_images']}")
        
        if 'distribution' in balance:
            print(f"  Class distribution:")
            for class_name, count in balance['distribution'].items():
                percentage = balance['percentages'][class_name]
                print(f"    {class_name}: {count} images ({percentage:.1f}%)")
        
        if validation['issues']:
            print(f"  Issues found: {len(validation['issues'])}")
            for issue in validation['issues'][:3]:  # Show first 3 issues
                print(f"    - {issue}")
            if len(validation['issues']) > 3:
                print(f"    ... and {len(validation['issues']) - 3} more")
    
    print(f"\n📄 Detailed report saved to: data/data_validation_report.json")
    print(f"Total valid images: {report['summary']['total_valid_images']}")
    
    if report['summary']['ready_for_training']:
        print("✅ Dataset is ready for training!")
    else:
        print("❌ Dataset needs attention before training.")

if __name__ == "__main__":
    main()
