"""
Indian Apple Intelligence System - Core Pipeline
===============================================
Modular, independent functions for apple analysis.
Each objective runs independently with no cross-dependencies.
"""

import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import joblib
import os
from typing import Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppleAnalysisPipeline:
    """
    Core pipeline for Indian Apple Intelligence System.
    All functions are independent and modular.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Indian apple varieties supported
        self.varieties = ['Sharbati', 'Sunehari', 'Maharaji', 'Splendour', 'Himsona', 'Himkiran']
        
        # Base shelf life per variety (days)
        self.base_shelf_life = {
            'Sharbati': 15,
            'Sunehari': 20,
            'Maharaji': 18,
            'Splendour': 22,
            'Himsona': 18,
            'Himkiran': 16
        }
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all models with error handling."""
        try:
            # Load Apple Detector (YOLOv8)
            detector_path = os.path.join(self.models_dir, "apple_detector.pt")
            if os.path.exists(detector_path):
                # Load the saved YOLO model
                import joblib
                saved_model = joblib.load(detector_path)
                if hasattr(saved_model, 'model'):
                    self.apple_detector = saved_model
                else:
                    # If it's the raw YOLO model, use it directly
                    self.apple_detector = saved_model
                logger.info("✅ Apple detector loaded successfully")
            else:
                # Try loading pre-trained YOLO as fallback
                try:
                    self.apple_detector = YOLO('yolov8n.pt')
                    logger.info("✅ Using pre-trained YOLOv8 for apple detection")
                except:
                    logger.warning("⚠️ Apple detector not found, using dummy")
                    self.apple_detector = None
        except Exception as e:
            logger.error(f"❌ Failed to load apple detector: {e}")
            # Try loading pre-trained YOLO as fallback
            try:
                self.apple_detector = YOLO('yolov8n.pt')
                logger.info("✅ Using pre-trained YOLOv8 for apple detection")
            except:
                logger.warning("⚠️ Apple detector not found, using dummy")
                self.apple_detector = None
        
        try:
            # Variety classifier - try best model first
            best_variety_path = os.path.join(self.models_dir, 'best_variety_model.pth')
            variety_path = os.path.join(self.models_dir, 'variety_classifier.pt')
            
            if os.path.exists(best_variety_path):
                # Load the best model with proper architecture
                import torchvision.models as models
                self.variety_model = models.efficientnet_b0(pretrained=False)
                self.variety_model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(self.variety_model.classifier[1].in_features, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(128, 6)
                )
                
                # Load the state dict
                state_dict = torch.load(best_variety_path, map_location=self.device)
                self.variety_model.load_state_dict(state_dict)
                self.variety_model.eval()
                logger.info("✅ Best variety classifier loaded successfully")
                
            elif os.path.exists(variety_path):
                self.variety_model = torch.load(variety_path, map_location=self.device)
                self.variety_model.eval()
                logger.info("✅ Variety classifier loaded successfully")
            else:
                logger.warning("⚠️ Variety classifier not found, using dummy")
                self.variety_model = None
                
        except Exception as e:
            logger.error(f"❌ Error loading variety classifier: {e}")
            self.variety_model = None
        
        try:
            # Health classifier
            health_path = os.path.join(self.models_dir, 'health_classifier.pt')
            if os.path.exists(health_path):
                self.health_model = torch.load(health_path, map_location=self.device)
                self.health_model.eval()
                logger.info("✅ Health classifier loaded successfully")
            else:
                logger.warning("⚠️ Health classifier not found, using dummy")
                self.health_model = None
                
        except Exception as e:
            logger.error(f"❌ Error loading health classifier: {e}")
            self.health_model = None
        
        try:
            # Surface classifier
            surface_path = os.path.join(self.models_dir, 'surface_classifier.pt')
            if os.path.exists(surface_path):
                self.surface_model = torch.load(surface_path, map_location=self.device)
                self.surface_model.eval()
                logger.info("✅ Surface classifier loaded successfully")
            else:
                logger.warning("⚠️ Surface classifier not found, using dummy")
                self.surface_model = None
                
        except Exception as e:
            logger.error(f"❌ Error loading surface classifier: {e}")
            self.surface_model = None
        
        try:
            # Shelf life predictor
            shelf_path = os.path.join(self.models_dir, 'shelf_life_model.pkl')
            if os.path.exists(shelf_path):
                self.shelf_life_model = joblib.load(shelf_path)
                logger.info("✅ Shelf life predictor loaded successfully")
            else:
                logger.warning("⚠️ Shelf life predictor not found, using dummy")
                self.shelf_life_model = None
                
        except Exception as e:
            logger.error(f"❌ Error loading shelf life predictor: {e}")
            self.shelf_life_model = None
    
    def _preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """Preprocess image for model inference."""
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return transform(image).unsqueeze(0).to(self.device)
    
    def _extract_simple_texture_features(self, image: Image.Image) -> np.ndarray:
        """Extract simplified texture features for surface classification."""
        try:
            # Convert to grayscale
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # Resize for consistency
            img_array = cv2.resize(img_array, (224, 224))
            
            # Basic statistical features
            mean_intensity = np.mean(img_array) / 255.0
            std_intensity = np.std(img_array) / 255.0
            
            # Simple gradient features
            grad_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
            
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            mean_gradient = np.mean(grad_magnitude) / 255.0
            std_gradient = np.std(grad_magnitude) / 255.0
            
            # Simple texture measures
            # Local variance (measure of texture roughness)
            kernel = np.ones((5,5)) / 25
            local_mean = cv2.filter2D(img_array.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((img_array.astype(np.float32) - local_mean)**2, -1, kernel)
            mean_local_variance = np.mean(local_variance) / (255**2)
            
            # Edge density
            edges = cv2.Canny(img_array, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Combine features (16 features total, padded if needed)
            features = np.array([
                mean_intensity, std_intensity, mean_gradient, std_gradient,
                mean_local_variance, edge_density,
                # Add more basic features
                np.percentile(img_array, 25) / 255.0,  # 25th percentile
                np.percentile(img_array, 75) / 255.0,  # 75th percentile
                np.mean(grad_x) / 255.0,  # Mean horizontal gradient
                np.mean(grad_y) / 255.0,  # Mean vertical gradient
                # Pad to 16 features
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)
            
            return features[:16]  # Ensure exactly 16 features
            
        except Exception as e:
            logger.warning(f"Failed to extract texture features: {e}")
            # Return dummy features if extraction fails
            return np.zeros(16, dtype=np.float32)
    
    def detect_apple(self, image: Image.Image) -> Dict[str, any]:
        """
        🍎 OBJECTIVE 1: Apple Detection (MANDATORY FIRST STEP)
        Uses YOLOv8 to detect if image contains an apple.
        Returns: {"detected": bool, "confidence": float, "message": str}
        """
        try:
            if self.apple_detector is None:
                # More realistic dummy logic - not all images are apples
                import random
                # 30% chance of detecting apple in dummy mode (more realistic)
                is_apple = random.random() < 0.3
                if is_apple:
                    confidence = random.uniform(0.85, 0.95)
                    return {
                        "detected": True,
                        "confidence": confidence,
                        "message": f"🍎 Apple detected (Confidence: {confidence:.1%}) [Dummy Mode]"
                    }
                else:
                    return {
                        "detected": False,
                        "confidence": 0.0,
                        "message": "❌ No apple detected in image [Dummy Mode]"
                    }
            
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Run YOLO detection
            results = self.apple_detector(cv_image)
            
            # Check if apple is detected (COCO class 47 is apple)
            apple_detected = False
            max_confidence = 0.0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detected class is apple (class 47 in COCO) and confidence > threshold
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()
                        
                        if class_id == 47 and confidence > 0.5:  # Apple class with 50% confidence
                            apple_detected = True
                            max_confidence = max(max_confidence, confidence)
            
            if apple_detected:
                return {
                    "detected": True,
                    "confidence": max_confidence,
                    "message": f"🍎 Apple detected successfully (Confidence: {max_confidence:.1%})"
                }
            else:
                return {
                    "detected": False,
                    "confidence": 0.0,
                    "message": "❌ No apple detected in the image. Please upload a clear image of a single apple."
                }
                
        except Exception as e:
            logger.error(f"Error in apple detection: {e}")
            return {
                "detected": False,
                "confidence": 0.0,
                "message": f"❌ Error during apple detection: {str(e)}"
            }
    
    def classify_variety(self, image: Image.Image) -> Dict[str, any]:
        """
        🍎 OBJECTIVE 2: Variety Classification
        Classifies apple variety using EfficientNet-B3.
        Returns: {"variety": str, "confidence": float, "message": str}
        """
        try:
            if self.variety_model is None:
                # Dummy logic - simulate variety prediction
                import random
                variety = random.choice(self.varieties)
                confidence = random.uniform(0.85, 0.98)
                
                return {
                    "variety": variety,
                    "confidence": confidence,
                    "message": f"🍎 Predicted Variety: {variety} (Confidence: {confidence:.1%}) [Dummy Mode]"
                }
            
            logger.info("Using trained variety model for prediction")
            
            # Preprocess image
            input_tensor = self._preprocess_image(image)
            
            # Model inference
            with torch.no_grad():
                outputs = self.variety_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                variety = self.varieties[predicted_idx.item()]
                conf_score = confidence.item()
                
                # Debug logging
                logger.info(f"Variety prediction - Index: {predicted_idx.item()}, Variety: {variety}, Confidence: {conf_score:.3f}")
                
                # Get top 3 predictions for better insight
                top3_probs, top3_indices = torch.topk(probabilities, 3)
                top3_varieties = [self.varieties[idx.item()] for idx in top3_indices[0]]
                top3_scores = [prob.item() for prob in top3_probs[0]]
                
                logger.info(f"Top 3: {list(zip(top3_varieties, top3_scores))}")
                
                if conf_score > 0.60:  # Lowered threshold to see more predictions
                    # Include top 3 in message for transparency
                    top3_str = ", ".join([f"{var}({score:.1%})" for var, score in zip(top3_varieties[:2], top3_scores[:2])])
                    return {
                        "variety": variety,
                        "confidence": conf_score,
                        "message": f"🍎 Predicted Variety: {variety} (Confidence: {conf_score:.1%}) | Also considered: {top3_str}"
                    }
                else:
                    return {
                        "variety": "Unknown",
                        "confidence": conf_score,
                        "message": f"⚠️ Low confidence variety prediction: {variety} ({conf_score:.1%})"
                    }
                    
        except Exception as e:
            logger.error(f"Error in variety classification: {e}")
            return {
                "variety": "Error",
                "confidence": 0.0,
                "message": f"❌ Error during variety classification: {str(e)}"
            }
    
    def predict_health(self, image: Image.Image) -> Dict[str, any]:
        """
        ✅❌ OBJECTIVE 3: Health/Freshness Detection
        Predicts if apple is healthy or rotten using ResNet18.
        Returns: {"health": str, "confidence": float, "message": str}
        """
        try:
            if self.health_model is None:
                # Dummy logic - simulate health prediction
                import random
                is_healthy = random.choice([True, False])
                confidence = random.uniform(0.85, 0.98)
                
                health_status = "Healthy" if is_healthy else "Rotten"
                emoji = "✅" if is_healthy else "⚠️"
                
                return {
                    "health": health_status,
                    "confidence": confidence,
                    "message": f"{emoji} Condition: {health_status} (Confidence: {confidence:.1%}) [Dummy Mode]"
                }
            
            # Preprocess image
            input_tensor = self._preprocess_image(image)
            
            # Model inference
            with torch.no_grad():
                outputs = self.health_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Assuming 0=Healthy, 1=Rotten
                is_healthy = predicted_idx.item() == 0
                health_status = "Healthy" if is_healthy else "Rotten"
                emoji = "✅" if is_healthy else "⚠️"
                conf_score = confidence.item()
                
                return {
                    "health": health_status,
                    "confidence": conf_score,
                    "message": f"{emoji} Condition: {health_status} (Confidence: {conf_score:.1%})"
                }
                
        except Exception as e:
            logger.error(f"Error in health prediction: {e}")
            return {
                "health": "Error",
                "confidence": 0.0,
                "message": f"❌ Error during health prediction: {str(e)}"
            }
    
    def predict_surface(self, image: Image.Image) -> Dict[str, any]:
        """
        🧼🧴 OBJECTIVE 4: Surface Classification (Waxed vs Unwaxed)
        Uses hybrid CNN + texture features approach.
        Returns: {"surface": str, "confidence": float, "message": str}
        """
        try:
            if self.surface_model is None:
                # Dummy logic - simulate surface prediction
                import random
                is_waxed = random.choice([True, False])
                confidence = random.uniform(0.85, 0.98)
                
                surface_type = "Waxed" if is_waxed else "Unwaxed"
                emoji = "🧴" if is_waxed else "🧼"
                
                return {
                    "surface": surface_type,
                    "confidence": confidence,
                    "message": f"{emoji} Surface: {surface_type} (Confidence: {confidence:.1%}) [Dummy Mode]"
                }
            
            # Check if this is a hybrid model (expects texture features)
            try:
                # Try hybrid model inference first
                input_tensor = self._preprocess_image(image)
                
                # Extract texture features (simplified version)
                texture_features = self._extract_simple_texture_features(image)
                texture_tensor = torch.from_numpy(texture_features).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.surface_model(input_tensor, texture_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    # Assuming 0=Unwaxed, 1=Waxed
                    is_waxed = predicted_idx.item() == 1
                    surface_type = "Waxed" if is_waxed else "Unwaxed"
                    emoji = "🧴" if is_waxed else "🧼"
                    conf_score = confidence.item()
                    
                    return {
                        "surface": surface_type,
                        "confidence": conf_score,
                        "message": f"{emoji} Surface: {surface_type} (Confidence: {conf_score:.1%})"
                    }
            
            except Exception:
                # Fallback to simple CNN model
                input_tensor = self._preprocess_image(image)
                
                with torch.no_grad():
                    outputs = self.surface_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    # Assuming 0=Unwaxed, 1=Waxed
                    is_waxed = predicted_idx.item() == 1
                    surface_type = "Waxed" if is_waxed else "Unwaxed"
                    emoji = "🧴" if is_waxed else "🧼"
                    conf_score = confidence.item()
                    
                    return {
                        "surface": surface_type,
                        "confidence": conf_score,
                        "message": f"{emoji} Surface: {surface_type} (Confidence: {conf_score:.1%})"
                    }
                
        except Exception as e:
            logger.error(f"Error in surface prediction: {e}")
            return {
                "surface": "Error",
                "confidence": 0.0,
                "message": f"❌ Error during surface prediction: {str(e)}"
            }
    
    def predict_shelf_life(self, variety: str, health: str, surface: str) -> Dict[str, any]:
        """
        ⏳ OBJECTIVE 5: Shelf Life Prediction
        Predicts remaining shelf life based on variety, health, and surface.
        This function is READ-ONLY and doesn't modify other objectives.
        Returns: {"days": int, "message": str}
        """
        try:
            # Handle error cases
            if variety == "Error" or health == "Error" or surface == "Error":
                return {
                    "days": 0,
                    "message": "❌ Cannot predict shelf life due to classification errors"
                }
            
            # If apple is rotten, shelf life is 0
            if health == "Rotten":
                return {
                    "days": 0,
                    "message": "⏳ Estimated Shelf Life: 0 days (Apple is rotten)"
                }
            
            # Get base shelf life for variety
            base_days = self.base_shelf_life.get(variety, 18)  # Default 18 days
            
            # Apply waxed multiplier
            if surface == "Waxed":
                final_days = int(base_days * 1.5)  # 50% increase for waxed apples
            else:
                final_days = base_days
            
            # Use ML model if available
            if self.shelf_life_model is not None:
                try:
                    # Encode features for ML model
                    variety_encoded = self.varieties.index(variety) if variety in self.varieties else 0
                    health_encoded = 1 if health == "Healthy" else 0
                    surface_encoded = 1 if surface == "Waxed" else 0
                    
                    features = np.array([[variety_encoded, health_encoded, surface_encoded]])
                    ml_prediction = self.shelf_life_model.predict(features)[0]
                    
                    # Use ML prediction if reasonable, otherwise fallback to rule-based
                    if 0 <= ml_prediction <= 50:  # Sanity check
                        final_days = int(ml_prediction)
                        
                except Exception as e:
                    logger.warning(f"ML model prediction failed, using rule-based: {e}")
            
            return {
                "days": final_days,
                "message": f"⏳ Estimated Shelf Life: {final_days} days"
            }
            
        except Exception as e:
            logger.error(f"Error in shelf life prediction: {e}")
            return {
                "days": 0,
                "message": f"❌ Error during shelf life prediction: {str(e)}"
            }
    
    def analyze_apple(self, image: Image.Image) -> Dict[str, any]:
        """
        Complete apple analysis pipeline.
        First checks for apple detection, then runs all objectives independently.
        """
        # Step 1: Mandatory apple detection
        detection_result = self.detect_apple(image)
        
        if not detection_result["detected"]:
            return {
                "apple_detected": False,
                "detection": detection_result,
                "variety": None,
                "health": None,
                "surface": None,
                "shelf_life": None,
                "error_message": detection_result["message"]
            }
        
        # Step 2: Run all objectives independently and in parallel
        try:
            variety_result = self.classify_variety(image)
            health_result = self.predict_health(image)
            surface_result = self.predict_surface(image)
            
            # Step 3: Shelf life prediction (read-only, uses results from above)
            shelf_life_result = self.predict_shelf_life(
                variety_result["variety"],
                health_result["health"],
                surface_result["surface"]
            )
            
            return {
                "apple_detected": True,
                "detection": detection_result,
                "variety": variety_result,
                "health": health_result,
                "surface": surface_result,
                "shelf_life": shelf_life_result,
                "error_message": None
            }
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            return {
                "apple_detected": True,
                "detection": detection_result,
                "variety": None,
                "health": None,
                "surface": None,
                "shelf_life": None,
                "error_message": f"Analysis error: {str(e)}"
            }
