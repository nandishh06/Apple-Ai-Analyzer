"""
Training Orchestrator - Train All Models Safely
===============================================
Orchestrates training of all models with proper isolation and error handling.
Each model trains independently without affecting others.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import json
from datetime import datetime
import logging
from model_manager import ModelManager
from data_validator import DataValidator

class TrainingOrchestrator:
    """
    Orchestrates training of all models with proper isolation.
    """
    
    def __init__(self, data_dir="data", models_dir="models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.model_manager = ModelManager(models_dir)
        self.data_validator = DataValidator(data_dir)
        self.logger = self._setup_logger()
        
        # Training scripts mapping
        self.training_scripts = {
            'apple_detector.pt': 'train_apple_detector.py',
            'variety_classifier.pt': 'train_variety.py',
            'health_classifier.pt': 'train_health_classifier.py',
            'surface_classifier.pt': 'train_surface_classifier.py',
            'shelf_life_model.pkl': 'train_shelf_life.py'
        }
        
        # Training order (some models can train in parallel)
        self.training_groups = [
            # Group 1: Independent models that can train in parallel
            ['apple_detector.pt', 'shelf_life_model.pkl'],
            # Group 2: Classification models (can train in parallel)
            ['variety_classifier.pt', 'health_classifier.pt', 'surface_classifier.pt']
        ]
        
    def _setup_logger(self):
        """Setup logger for training orchestrator."""
        logger = logging.getLogger('TrainingOrchestrator')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.models_dir / "training_orchestrator.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def validate_data_before_training(self):
        """Validate all datasets before starting training."""
        
        self.logger.info("🔍 Validating datasets before training...")
        
        # Generate validation report
        report = self.data_validator.generate_data_report()
        
        # Check if data is ready for training
        if not report['summary']['ready_for_training']:
            self.logger.error("❌ Data validation failed. Cannot proceed with training.")
            return False, report
        
        # Check individual categories
        issues = []
        for category, data in report['categories'].items():
            validation = data['validation']
            
            if validation['valid_images'] == 0:
                issues.append(f"No valid images found for {category}")
            elif validation['valid_images'] < 10:
                issues.append(f"Very few images for {category}: {validation['valid_images']}")
            
            if validation['invalid_images'] > validation['valid_images'] * 0.2:
                issues.append(f"High number of invalid images in {category}: {validation['invalid_images']}")
        
        if issues:
            self.logger.warning("⚠️ Data quality issues detected:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
            
            # Ask user if they want to continue
            response = input("Continue with training despite data issues? (y/n): ")
            if response.lower() != 'y':
                return False, report
        
        self.logger.info("✅ Data validation passed. Ready for training.")
        return True, report
    
    def check_training_prerequisites(self):
        """Check if all prerequisites for training are met."""
        
        self.logger.info("🔧 Checking training prerequisites...")
        
        # Check Python packages
        required_packages = [
            'torch', 'torchvision', 'ultralytics', 'sklearn', 
            'opencv-python', 'gradio', 'matplotlib', 'seaborn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"❌ Missing required packages: {missing_packages}")
            self.logger.error("Install with: pip install " + " ".join(missing_packages))
            return False
        
        # Check directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Check available disk space (at least 1GB)
        import shutil
        free_space = shutil.disk_usage(self.models_dir).free / (1024**3)  # GB
        if free_space < 1.0:
            self.logger.warning(f"⚠️ Low disk space: {free_space:.1f}GB available")
        
        self.logger.info("✅ Prerequisites check passed.")
        return True
    
    def run_training_script(self, script_name, model_name):
        """Run a training script in a separate process."""
        
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            self.logger.error(f"❌ Training script not found: {script_name}")
            return False, f"Script not found: {script_name}"
        
        self.logger.info(f"🚀 Starting training: {model_name}")
        
        try:
            # Run training script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Training completed successfully: {model_name}")
                return True, result.stdout
            else:
                self.logger.error(f"❌ Training failed: {model_name}")
                self.logger.error(f"Error output: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ Training timeout: {model_name}")
            return False, "Training timeout (1 hour)"
        
        except Exception as e:
            self.logger.error(f"❌ Training error: {model_name} - {e}")
            return False, str(e)
    
    def train_model_group(self, model_names, parallel=False):
        """Train a group of models (parallel or sequential)."""
        
        if parallel and len(model_names) > 1:
            self.logger.info(f"🔄 Training models in parallel: {model_names}")
            
            # For simplicity, we'll train sequentially but could use threading
            # In a production system, you might use multiprocessing
            results = {}
            
            for model_name in model_names:
                if model_name in self.training_scripts:
                    script_name = self.training_scripts[model_name]
                    success, output = self.run_training_script(script_name, model_name)
                    results[model_name] = {'success': success, 'output': output}
                else:
                    results[model_name] = {'success': False, 'output': 'No training script found'}
            
            return results
        
        else:
            # Sequential training
            self.logger.info(f"🔄 Training models sequentially: {model_names}")
            
            results = {}
            
            for model_name in model_names:
                if model_name in self.training_scripts:
                    script_name = self.training_scripts[model_name]
                    success, output = self.run_training_script(script_name, model_name)
                    results[model_name] = {'success': success, 'output': output}
                    
                    if not success:
                        self.logger.warning(f"⚠️ Training failed for {model_name}, continuing with others...")
                else:
                    results[model_name] = {'success': False, 'output': 'No training script found'}
            
            return results
    
    def generate_training_report(self, all_results, start_time, end_time):
        """Generate comprehensive training report."""
        
        duration = end_time - start_time
        
        report = {
            'training_session': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'duration_formatted': str(duration)
            },
            'models': {},
            'summary': {
                'total_models': len(all_results),
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0
            }
        }
        
        # Process results
        for model_name, result in all_results.items():
            report['models'][model_name] = {
                'success': result['success'],
                'script': self.training_scripts.get(model_name, 'Unknown'),
                'output_length': len(result['output']),
                'has_error': not result['success']
            }
            
            if result['success']:
                report['summary']['successful'] += 1
            else:
                report['summary']['failed'] += 1
        
        # Calculate success rate
        if report['summary']['total_models'] > 0:
            report['summary']['success_rate'] = (
                report['summary']['successful'] / report['summary']['total_models']
            )
        
        # Get model status
        report['model_status'] = self.model_manager.get_model_status()
        
        # Save report
        report_path = self.models_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📄 Training report saved to {report_path}")
        
        return report
    
    def print_training_summary(self, report):
        """Print a nice summary of training results."""
        
        print("\n" + "="*60)
        print("🎯 TRAINING SUMMARY")
        print("="*60)
        
        print(f"📅 Duration: {report['training_session']['duration_formatted']}")
        print(f"📊 Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"✅ Successful: {report['summary']['successful']}")
        print(f"❌ Failed: {report['summary']['failed']}")
        
        print("\n📋 Model Status:")
        print("-" * 30)
        
        for model_name, result in report['models'].items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"{model_name:<25} {status}")
        
        print("\n💾 Model Files:")
        print("-" * 30)
        
        for model_name, status in report['model_status'].items():
            exists = "✅" if status['exists'] else "❌"
            size_mb = status['size'] / (1024*1024) if status['size'] > 0 else 0
            print(f"{model_name:<25} {exists} ({size_mb:.1f} MB)")
        
        if report['summary']['success_rate'] == 1.0:
            print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
            print("Your Apple Intelligence System is ready to use!")
        elif report['summary']['success_rate'] > 0.5:
            print("\n✅ MAJORITY OF MODELS TRAINED SUCCESSFULLY!")
            print("Your system is functional with some limitations.")
        else:
            print("\n⚠️ MULTIPLE TRAINING FAILURES DETECTED")
            print("Please check the logs and fix issues before proceeding.")
        
        print("="*60)
    
    def train_all_models(self, skip_validation=False, parallel_training=True):
        """Main function to train all models."""
        
        start_time = datetime.now()
        self.logger.info("🚀 Starting comprehensive model training...")
        
        # Check prerequisites
        if not self.check_training_prerequisites():
            return False
        
        # Validate data
        if not skip_validation:
            data_valid, validation_report = self.validate_data_before_training()
            if not data_valid:
                return False
        
        # Train models in groups
        all_results = {}
        
        for group_idx, model_group in enumerate(self.training_groups):
            self.logger.info(f"\n🔄 Training Group {group_idx + 1}: {model_group}")
            
            # Check if any models in this group are already being trained
            busy_models = []
            for model_name in model_group:
                if self.model_manager.is_model_training(model_name):
                    busy_models.append(model_name)
            
            if busy_models:
                self.logger.warning(f"⚠️ Some models are already training: {busy_models}")
                # Skip busy models
                model_group = [m for m in model_group if m not in busy_models]
            
            if model_group:  # If there are models to train
                group_results = self.train_model_group(model_group, parallel=parallel_training)
                all_results.update(group_results)
                
                # Brief pause between groups
                time.sleep(2)
        
        end_time = datetime.now()
        
        # Generate and display report
        report = self.generate_training_report(all_results, start_time, end_time)
        self.print_training_summary(report)
        
        # Cleanup old backups
        self.model_manager.cleanup_old_backups()
        
        return report['summary']['success_rate'] > 0.5

def main():
    """Main function to orchestrate training."""
    
    print("🏗️ Apple Intelligence System - Model Training Orchestrator")
    print("=" * 65)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train all Apple Intelligence models')
    parser.add_argument('--skip-validation', action='store_true', 
                       help='Skip data validation before training')
    parser.add_argument('--sequential', action='store_true',
                       help='Train models sequentially instead of in parallel')
    parser.add_argument('--models', nargs='+', 
                       help='Train specific models only',
                       choices=['apple_detector.pt', 'variety_classifier.pt', 
                               'health_classifier.pt', 'surface_classifier.pt', 
                               'shelf_life_model.pkl'])
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator()
    
    # Train specific models if requested
    if args.models:
        print(f"🎯 Training specific models: {args.models}")
        
        all_results = {}
        for model_name in args.models:
            if model_name in orchestrator.training_scripts:
                script_name = orchestrator.training_scripts[model_name]
                success, output = orchestrator.run_training_script(script_name, model_name)
                all_results[model_name] = {'success': success, 'output': output}
        
        # Generate report for specific models
        start_time = datetime.now()
        end_time = datetime.now()
        report = orchestrator.generate_training_report(all_results, start_time, end_time)
        orchestrator.print_training_summary(report)
        
    else:
        # Train all models
        success = orchestrator.train_all_models(
            skip_validation=args.skip_validation,
            parallel_training=not args.sequential
        )
        
        if success:
            print("\n🎉 Training orchestration completed successfully!")
            print("Run 'python app/app.py' to test your trained models.")
        else:
            print("\n❌ Training orchestration completed with issues.")
            print("Check the logs for details and retry failed models.")

if __name__ == "__main__":
    main()
