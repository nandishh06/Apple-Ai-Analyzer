"""
Shelf Life Predictor Training Script
====================================
Trains Random Forest model for shelf life prediction.
Isolated training that doesn't affect other models.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from model_manager import ModelManager, SafeTraining
import logging

class ShelfLifeTrainer:
    """
    Isolated trainer for shelf life prediction model.
    """
    
    def __init__(self, data_dir="data", models_dir="models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.model_manager = ModelManager(models_dir)
        self.logger = self._setup_logger()
        
        # Training configuration
        self.model_name = "shelf_life_model.pkl"
        self.varieties = ['Sharbati', 'Sunehari', 'Maharaji', 'Splendour', 'Himsona', 'Himkiran']
        
        # Base shelf life per variety (days) - research-based values
        self.base_shelf_life = {
            'Sharbati': 15,
            'Sunehari': 20,
            'Maharaji': 18,
            'Splendour': 22,
            'Himsona': 18,
            'Himkiran': 16
        }
        
    def _setup_logger(self):
        """Setup logger for shelf life training."""
        logger = logging.getLogger('ShelfLifeTrainer')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def generate_synthetic_data(self, n_samples=5000):
        """
        Generate synthetic training data based on domain knowledge.
        In a real scenario, this would be replaced with actual collected data.
        """
        self.logger.info(f"📊 Generating {n_samples} synthetic training samples...")
        
        np.random.seed(42)  # For reproducible results
        
        data = []
        
        for _ in range(n_samples):
            # Random variety
            variety = np.random.choice(self.varieties)
            variety_encoded = self.varieties.index(variety)
            
            # Random health (80% healthy, 20% rotten)
            is_healthy = np.random.choice([0, 1], p=[0.2, 0.8])
            
            # Random surface (60% unwaxed, 40% waxed)
            is_waxed = np.random.choice([0, 1], p=[0.6, 0.4])
            
            # Calculate shelf life based on rules
            if is_healthy == 0:  # Rotten
                shelf_life = 0
            else:
                base_days = self.base_shelf_life[variety]
                
                # Apply waxing effect
                if is_waxed:
                    shelf_life = base_days * 1.5  # Waxed apples last 50% longer
                else:
                    shelf_life = base_days
                
                # Add environmental factors (temperature, humidity effects)
                temp_factor = np.random.normal(1.0, 0.15)  # Temperature variation
                humidity_factor = np.random.normal(1.0, 0.1)  # Humidity variation
                
                shelf_life *= temp_factor * humidity_factor
                
                # Add some random noise
                noise = np.random.normal(0, 1.5)
                shelf_life += noise
                
                # Ensure non-negative
                shelf_life = max(0, shelf_life)
                
                # Cap at reasonable maximum (45 days)
                shelf_life = min(shelf_life, 45)
            
            # Add additional features that might affect shelf life
            storage_temp = np.random.normal(4, 2)  # Storage temperature (°C)
            humidity = np.random.normal(85, 10)  # Relative humidity (%)
            days_since_harvest = np.random.randint(0, 7)  # Days since harvest
            
            data.append({
                'variety_encoded': variety_encoded,
                'variety_name': variety,
                'is_healthy': is_healthy,
                'is_waxed': is_waxed,
                'storage_temp': storage_temp,
                'humidity': humidity,
                'days_since_harvest': days_since_harvest,
                'shelf_life_days': shelf_life
            })
        
        df = pd.DataFrame(data)
        
        # Log statistics
        self.logger.info("📈 Dataset statistics:")
        self.logger.info(f"  Healthy samples: {(df['is_healthy'] == 1).sum()} ({(df['is_healthy'] == 1).mean():.1%})")
        self.logger.info(f"  Waxed samples: {(df['is_waxed'] == 1).sum()} ({(df['is_waxed'] == 1).mean():.1%})")
        self.logger.info(f"  Average shelf life: {df['shelf_life_days'].mean():.1f} days")
        self.logger.info(f"  Shelf life range: {df['shelf_life_days'].min():.1f} - {df['shelf_life_days'].max():.1f} days")
        
        # Variety distribution
        variety_counts = df['variety_name'].value_counts()
        self.logger.info("  Variety distribution:")
        for variety, count in variety_counts.items():
            self.logger.info(f"    {variety}: {count} samples")
        
        return df
    
    def load_real_data(self):
        """
        Load real shelf life data if available.
        Returns None if no real data exists.
        """
        data_file = self.data_dir / "shelf_life_data.csv"
        
        if data_file.exists():
            self.logger.info(f"📂 Loading real data from {data_file}")
            try:
                df = pd.read_csv(data_file)
                
                # Validate required columns
                required_cols = ['variety_encoded', 'is_healthy', 'is_waxed', 'shelf_life_days']
                if all(col in df.columns for col in required_cols):
                    self.logger.info(f"✅ Loaded {len(df)} real samples")
                    return df
                else:
                    self.logger.warning(f"⚠️ Missing required columns in real data: {required_cols}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"❌ Error loading real data: {e}")
                return None
        
        return None
    
    def prepare_features(self, df):
        """Prepare feature matrix and target vector."""
        
        # Basic features used by the pipeline
        basic_features = ['variety_encoded', 'is_healthy', 'is_waxed']
        
        # Extended features if available
        extended_features = ['storage_temp', 'humidity', 'days_since_harvest']
        
        # Use extended features if available
        feature_cols = basic_features.copy()
        for feat in extended_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        X = df[feature_cols].values
        y = df['shelf_life_days'].values
        
        self.logger.info(f"🔧 Using features: {feature_cols}")
        self.logger.info(f"📊 Feature matrix shape: {X.shape}")
        
        return X, y, feature_cols
    
    def train_model(self, X, y):
        """Train Random Forest model."""
        
        self.logger.info("🌲 Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Create model with optimized parameters
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        # Log results
        self.logger.info("📊 Model Performance:")
        self.logger.info(f"  Training MAE: {train_mae:.2f} days")
        self.logger.info(f"  Test MAE: {test_mae:.2f} days")
        self.logger.info(f"  Training RMSE: {train_rmse:.2f} days")
        self.logger.info(f"  Test RMSE: {test_rmse:.2f} days")
        self.logger.info(f"  Training R²: {train_r2:.3f}")
        self.logger.info(f"  Test R²: {test_r2:.3f}")
        self.logger.info(f"  Cross-validation MAE: {cv_mae:.2f} days")
        
        # Feature importance
        feature_importance = model.feature_importances_
        self.logger.info("🔍 Feature Importance:")
        for i, importance in enumerate(feature_importance):
            self.logger.info(f"  Feature {i}: {importance:.3f}")
        
        # Create performance metrics dictionary
        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'feature_importance': feature_importance.tolist()
        }
        
        return model, metrics, (X_test, y_test, test_pred)
    
    def plot_results(self, test_data, metrics):
        """Plot training results and save visualizations."""
        
        X_test, y_test, y_pred = test_data
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Shelf Life (days)')
        axes[0, 0].set_ylabel('Predicted Shelf Life (days)')
        axes[0, 0].set_title(f'Actual vs Predicted\nR² = {metrics["test_r2"]:.3f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Shelf Life (days)')
        axes[0, 1].set_ylabel('Residuals (days)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature Importance
        feature_names = ['Variety', 'Health', 'Surface', 'Storage Temp', 'Humidity', 'Days Since Harvest']
        importance = metrics['feature_importance']
        
        # Pad feature names if needed
        while len(feature_names) < len(importance):
            feature_names.append(f'Feature {len(feature_names)}')
        
        feature_names = feature_names[:len(importance)]
        
        axes[1, 0].barh(feature_names, importance, color='orange')
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Feature Importance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Distribution of predictions
        axes[1, 1].hist(y_test, bins=30, alpha=0.7, label='Actual', color='blue')
        axes[1, 1].hist(y_pred, bins=30, alpha=0.7, label='Predicted', color='red')
        axes[1, 1].set_xlabel('Shelf Life (days)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.models_dir / 'shelf_life_training_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Training results saved to {plot_path}")
    
    def test_model_predictions(self, model, feature_cols):
        """Test model with example predictions."""
        
        self.logger.info("🧪 Testing model with example predictions...")
        
        # Test cases
        test_cases = [
            {'variety': 'Himsona', 'health': 'Healthy', 'surface': 'Waxed'},
            {'variety': 'Sunehari', 'health': 'Healthy', 'surface': 'Unwaxed'},
            {'variety': 'Sharbati', 'health': 'Rotten', 'surface': 'Waxed'},
            {'variety': 'Splendour', 'health': 'Healthy', 'surface': 'Waxed'},
            {'variety': 'Maharaji', 'health': 'Healthy', 'surface': 'Unwaxed'}
        ]
        
        for i, case in enumerate(test_cases):
            # Encode features
            variety_encoded = self.varieties.index(case['variety'])
            health_encoded = 1 if case['health'] == 'Healthy' else 0
            surface_encoded = 1 if case['surface'] == 'Waxed' else 0
            
            # Create feature vector (basic features only for compatibility)
            features = [variety_encoded, health_encoded, surface_encoded]
            
            # Pad with default values if model expects more features
            while len(features) < len(feature_cols):
                if 'storage_temp' in feature_cols[len(features)]:
                    features.append(4.0)  # Default storage temp
                elif 'humidity' in feature_cols[len(features)]:
                    features.append(85.0)  # Default humidity
                elif 'days_since_harvest' in feature_cols[len(features)]:
                    features.append(2)  # Default days since harvest
                else:
                    features.append(0.0)  # Default value
            
            features = np.array([features])
            
            # Predict
            prediction = model.predict(features)[0]
            
            self.logger.info(f"  Test {i+1}: {case['variety']} ({case['health']}, {case['surface']}) → {prediction:.1f} days")
    
    def train_shelf_life_predictor(self):
        """Main training function."""
        
        self.logger.info("⏳ Starting shelf life predictor training...")
        
        try:
            with SafeTraining(self.model_manager, self.model_name):
                
                # Try to load real data first
                df = self.load_real_data()
                
                # If no real data, generate synthetic data
                if df is None:
                    df = self.generate_synthetic_data()
                
                # Prepare features
                X, y, feature_cols = self.prepare_features(df)
                
                # Train model
                model, metrics, test_data = self.train_model(X, y)
                
                # Plot results
                self.plot_results(test_data, metrics)
                
                # Test model
                self.test_model_predictions(model, feature_cols)
                
                # Save model with metadata
                model_info = {
                    'model_type': 'RandomForest_shelf_life_predictor',
                    'n_estimators': 200,
                    'features': feature_cols,
                    'target': 'shelf_life_days',
                    'training_samples': len(X),
                    'test_mae': metrics['test_mae'],
                    'test_r2': metrics['test_r2'],
                    'cv_mae': metrics['cv_mae'],
                    'varieties': self.varieties,
                    'base_shelf_life': self.base_shelf_life
                }
                
                success = self.model_manager.save_model_safely(
                    model, 
                    self.model_name, 
                    model_info
                )
                
                if success:
                    # Save training metrics
                    metrics_path = self.models_dir / 'shelf_life_training_metrics.json'
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    
                    # Save feature columns for future use
                    features_path = self.models_dir / 'shelf_life_features.json'
                    with open(features_path, 'w') as f:
                        json.dump({'features': feature_cols}, f, indent=2)
                    
                    self.logger.info("✅ Shelf life predictor training completed successfully!")
                    self.logger.info(f"Test MAE: {metrics['test_mae']:.2f} days")
                    self.logger.info(f"Test R²: {metrics['test_r2']:.3f}")
                    return True
                else:
                    self.logger.error("❌ Failed to save trained model")
                    return False
        
        except Exception as e:
            self.logger.error(f"❌ Shelf life predictor training failed: {e}")
            return False

def main():
    """Main function to train shelf life predictor."""
    print("⏳ Shelf Life Predictor Training")
    print("=" * 35)
    
    # Check if already training
    model_manager = ModelManager()
    if model_manager.is_model_training("shelf_life_model.pkl"):
        print("⚠️ Shelf life predictor is already being trained!")
        return
    
    # Create trainer
    trainer = ShelfLifeTrainer()
    
    # Train model
    success = trainer.train_shelf_life_predictor()
    
    if success:
        print("🎉 Shelf life predictor training completed!")
    else:
        print("❌ Shelf life predictor training failed!")
        
        # Check if rollback is needed
        if input("Would you like to rollback to previous version? (y/n): ").lower() == 'y':
            model_manager.rollback_model("shelf_life_model.pkl")

if __name__ == "__main__":
    main()
