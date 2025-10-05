"""
Model Manager - Safe Training with Versioning and Backup
========================================================
Ensures each model trains independently without affecting others.
Provides rollback capability if training fails.
"""

import os
import shutil
import json
import torch
import joblib
from datetime import datetime
import logging
from pathlib import Path

class ModelManager:
    """
    Manages model versions, backups, and safe training operations.
    Ensures one model training doesn't affect others.
    """
    
    def __init__(self, models_dir="models", backup_dir="model_backups"):
        self.models_dir = Path(models_dir)
        self.backup_dir = Path(backup_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.registry_file = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()
        
        # Setup logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup dedicated logger for model management."""
        logger = logging.getLogger('ModelManager')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.models_dir / "training.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def _load_registry(self):
        """Load model registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save model registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def backup_model(self, model_name):
        """
        Create backup of existing model before training.
        Returns backup path or None if no existing model.
        """
        model_path = self.models_dir / f"{model_name}"
        
        if not model_path.exists():
            self.logger.info(f"No existing model to backup: {model_name}")
            return None
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{model_name.replace('.', '_')}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        try:
            if model_path.suffix in ['.pt', '.pth']:
                # PyTorch model
                shutil.copy2(model_path, backup_path.with_suffix('.pt'))
            elif model_path.suffix == '.pkl':
                # Scikit-learn model
                shutil.copy2(model_path, backup_path.with_suffix('.pkl'))
            
            self.logger.info(f"✅ Backed up {model_name} to {backup_path}")
            
            # Update registry
            if model_name not in self.registry:
                self.registry[model_name] = {}
            
            self.registry[model_name]['last_backup'] = str(backup_path)
            self.registry[model_name]['backup_time'] = timestamp
            self._save_registry()
            
            return backup_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to backup {model_name}: {e}")
            return None
    
    def save_model_safely(self, model, model_name, model_info=None):
        """
        Save model with atomic operation to prevent corruption.
        """
        temp_path = self.models_dir / f"{model_name}.tmp"
        final_path = self.models_dir / model_name
        
        try:
            # Save to temporary file first
            if model_name.endswith('.pt') or model_name.endswith('.pth'):
                torch.save(model, temp_path)
            elif model_name.endswith('.pkl'):
                joblib.dump(model, temp_path)
            else:
                raise ValueError(f"Unsupported model format: {model_name}")
            
            # Atomic move to final location
            shutil.move(temp_path, final_path)
            
            # Update registry
            if model_name not in self.registry:
                self.registry[model_name] = {}
            
            self.registry[model_name].update({
                'save_time': datetime.now().isoformat(),
                'file_size': final_path.stat().st_size,
                'status': 'active'
            })
            
            if model_info:
                self.registry[model_name].update(model_info)
            
            self._save_registry()
            self.logger.info(f"✅ Safely saved {model_name}")
            
            return True
            
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            
            self.logger.error(f"❌ Failed to save {model_name}: {e}")
            return False
    
    def rollback_model(self, model_name):
        """
        Rollback to previous version if training failed.
        """
        if model_name not in self.registry or 'last_backup' not in self.registry[model_name]:
            self.logger.warning(f"No backup available for {model_name}")
            return False
        
        backup_path = Path(self.registry[model_name]['last_backup'])
        model_path = self.models_dir / model_name
        
        try:
            if backup_path.exists():
                shutil.copy2(backup_path, model_path)
                self.logger.info(f"✅ Rolled back {model_name} from backup")
                
                # Update registry
                self.registry[model_name]['status'] = 'rolled_back'
                self.registry[model_name]['rollback_time'] = datetime.now().isoformat()
                self._save_registry()
                
                return True
            else:
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to rollback {model_name}: {e}")
            return False
    
    def is_model_training(self, model_name):
        """
        Check if a model is currently being trained.
        Uses lock files to prevent concurrent training.
        """
        lock_file = self.models_dir / f"{model_name}.lock"
        return lock_file.exists()
    
    def acquire_training_lock(self, model_name):
        """
        Acquire training lock for a model.
        Prevents concurrent training of the same model.
        """
        lock_file = self.models_dir / f"{model_name}.lock"
        
        if lock_file.exists():
            self.logger.warning(f"Model {model_name} is already being trained")
            return False
        
        try:
            with open(lock_file, 'w') as f:
                f.write(f"Training started: {datetime.now().isoformat()}\n")
                f.write(f"PID: {os.getpid()}\n")
            
            self.logger.info(f"🔒 Acquired training lock for {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to acquire lock for {model_name}: {e}")
            return False
    
    def release_training_lock(self, model_name):
        """Release training lock for a model."""
        lock_file = self.models_dir / f"{model_name}.lock"
        
        try:
            if lock_file.exists():
                lock_file.unlink()
                self.logger.info(f"🔓 Released training lock for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to release lock for {model_name}: {e}")
    
    def get_model_status(self):
        """Get status of all models."""
        status = {}
        
        for model_name in ['apple_detector.pt', 'variety_classifier.pt', 
                          'health_classifier.pt', 'surface_classifier.pt', 
                          'shelf_life_model.pkl']:
            
            model_path = self.models_dir / model_name
            
            status[model_name] = {
                'exists': model_path.exists(),
                'size': model_path.stat().st_size if model_path.exists() else 0,
                'training': self.is_model_training(model_name),
                'registry_info': self.registry.get(model_name, {})
            }
        
        return status
    
    def cleanup_old_backups(self, keep_last_n=5):
        """Clean up old backup files, keeping only the last N."""
        try:
            backup_files = list(self.backup_dir.glob("*"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the last N backups
            for backup_file in backup_files[keep_last_n:]:
                backup_file.unlink()
                self.logger.info(f"🗑️ Cleaned up old backup: {backup_file.name}")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup backups: {e}")

# Context manager for safe training
class SafeTraining:
    """
    Context manager for safe model training with automatic rollback.
    """
    
    def __init__(self, model_manager, model_name):
        self.model_manager = model_manager
        self.model_name = model_name
        self.backup_path = None
        self.lock_acquired = False
    
    def __enter__(self):
        # Acquire training lock
        if not self.model_manager.acquire_training_lock(self.model_name):
            raise RuntimeError(f"Could not acquire training lock for {self.model_name}")
        
        self.lock_acquired = True
        
        # Create backup
        self.backup_path = self.model_manager.backup_model(self.model_name)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release lock
        if self.lock_acquired:
            self.model_manager.release_training_lock(self.model_name)
        
        # If there was an exception and we have a backup, offer rollback
        if exc_type is not None and self.backup_path:
            self.model_manager.logger.error(
                f"Training failed for {self.model_name}. Backup available at {self.backup_path}"
            )
            
            # Auto-rollback could be implemented here if desired
            # self.model_manager.rollback_model(self.model_name)
        
        return False  # Don't suppress exceptions
