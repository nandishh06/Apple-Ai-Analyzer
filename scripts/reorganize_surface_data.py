"""
Surface Data Reorganizer
========================
Reorganizes nested surface data structure for training compatibility.
Moves images from variety subfolders to main waxed/unwaxed folders.
"""

import os
import shutil
from pathlib import Path
import logging

def reorganize_surface_data(data_dir="data", backup=True):
    """
    Reorganize surface data from nested structure to flat structure.
    
    From: surface/waxed/Variety/image.jpg
    To:   surface/waxed/image.jpg
    """
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    surface_dir = Path(data_dir) / "surface"
    
    if not surface_dir.exists():
        logger.error(f"Surface directory not found: {surface_dir}")
        return False
    
    # Create backup if requested
    if backup:
        backup_dir = Path(data_dir) / "surface_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        logger.info("📦 Creating backup...")
        shutil.copytree(surface_dir, backup_dir)
        logger.info(f"✅ Backup created at: {backup_dir}")
    
    # Process waxed and unwaxed folders
    for surface_type in ['waxed', 'unwaxed']:
        surface_type_dir = surface_dir / surface_type
        
        if not surface_type_dir.exists():
            logger.warning(f"Directory not found: {surface_type_dir}")
            continue
        
        logger.info(f"🔄 Processing {surface_type} images...")
        
        # Get all variety subdirectories
        variety_dirs = [d for d in surface_type_dir.iterdir() 
                       if d.is_dir() and d.name != '__pycache__']
        
        moved_count = 0
        
        for variety_dir in variety_dirs:
            logger.info(f"  📁 Processing {variety_dir.name}...")
            
            # Get all image files in this variety directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(variety_dir.glob(f"*{ext}"))
            
            # Move images to parent directory with variety prefix
            for img_file in image_files:
                # Create new filename with variety prefix
                new_name = f"{variety_dir.name}_{img_file.name}"
                new_path = surface_type_dir / new_name
                
                # Handle duplicate names
                counter = 1
                while new_path.exists():
                    name_parts = img_file.stem, counter, img_file.suffix
                    new_name = f"{variety_dir.name}_{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                    new_path = surface_type_dir / new_name
                    counter += 1
                
                # Move the file
                try:
                    shutil.move(str(img_file), str(new_path))
                    moved_count += 1
                except Exception as e:
                    logger.error(f"Failed to move {img_file}: {e}")
            
            # Remove empty variety directory
            try:
                if variety_dir.exists() and not any(variety_dir.iterdir()):
                    variety_dir.rmdir()
                    logger.info(f"  🗑️ Removed empty directory: {variety_dir.name}")
                elif variety_dir.exists():
                    # Directory not empty, list remaining files
                    remaining = list(variety_dir.iterdir())
                    logger.warning(f"  ⚠️ Directory not empty: {variety_dir.name} ({len(remaining)} items)")
            except Exception as e:
                logger.error(f"Failed to remove directory {variety_dir}: {e}")
        
        logger.info(f"✅ Moved {moved_count} {surface_type} images")
    
    # Verify final structure
    logger.info("\n📊 Final structure verification:")
    for surface_type in ['waxed', 'unwaxed']:
        surface_type_dir = surface_dir / surface_type
        if surface_type_dir.exists():
            # Count image files directly in the directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.tiff']
            image_count = 0
            
            for ext in image_extensions:
                image_count += len(list(surface_type_dir.glob(f"*{ext}")))
            
            logger.info(f"  {surface_type}: {image_count} images")
    
    logger.info("🎉 Surface data reorganization completed!")
    return True

def check_surface_structure(data_dir="data"):
    """Check current surface data structure."""
    
    surface_dir = Path(data_dir) / "surface"
    
    if not surface_dir.exists():
        print(f"❌ Surface directory not found: {surface_dir}")
        return
    
    print("🔍 Current Surface Data Structure:")
    print("=" * 40)
    
    for surface_type in ['waxed', 'unwaxed']:
        surface_type_dir = surface_dir / surface_type
        
        if not surface_type_dir.exists():
            print(f"❌ {surface_type} directory not found")
            continue
        
        print(f"\n📁 {surface_type.upper()}:")
        
        # Check if there are subdirectories (nested structure)
        subdirs = [d for d in surface_type_dir.iterdir() 
                  if d.is_dir() and d.name not in ['__pycache__', '.DS_Store']]
        
        # Check if there are direct image files (flat structure)
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.tiff']
        direct_images = 0
        for ext in image_extensions:
            direct_images += len(list(surface_type_dir.glob(f"*{ext}")))
        
        if subdirs:
            print(f"  📂 Nested structure detected:")
            for subdir in subdirs:
                if subdir.name.endswith('.md'):
                    continue
                item_count = len(list(subdir.iterdir()))
                print(f"    {subdir.name}/: {item_count} items")
        
        if direct_images > 0:
            print(f"  🖼️ Direct images: {direct_images}")
        
        if not subdirs and direct_images == 0:
            print(f"  ❌ No images found")
    
    # Recommendation
    if any((surface_dir / st).exists() and 
           [d for d in (surface_dir / st).iterdir() if d.is_dir() and d.name not in ['__pycache__', '.DS_Store']]
           for st in ['waxed', 'unwaxed']):
        print(f"\n💡 RECOMMENDATION:")
        print(f"Your data has nested structure. Run reorganization for training compatibility:")
        print(f"python scripts/reorganize_surface_data.py")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reorganize surface data structure')
    parser.add_argument('--check', action='store_true', help='Only check current structure')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup')
    
    args = parser.parse_args()
    
    if args.check:
        check_surface_structure()
    else:
        print("🔄 Surface Data Reorganization")
        print("=" * 35)
        
        # Check current structure first
        check_surface_structure()
        
        # Ask for confirmation
        response = input("\nProceed with reorganization? (y/n): ")
        if response.lower() == 'y':
            success = reorganize_surface_data(backup=not args.no_backup)
            if success:
                print("\n✅ Reorganization completed successfully!")
                print("Your surface data is now ready for training.")
            else:
                print("\n❌ Reorganization failed. Check the logs.")
        else:
            print("Operation cancelled.")

if __name__ == "__main__":
    main()
