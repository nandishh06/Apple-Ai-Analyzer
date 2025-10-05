"""
Fixed Indian Apple Intelligence System - Core Pipeline
====================================================
Fixed version with proper model loading for variety classification.
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
import torchvision.models as models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedAppleAnalysisPipeline:
    """
    Fixed core pipeline for Indian Apple Intelligence System.
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
        
        # Load Apple Detector (YOLOv8)
        try:
            self.apple_detector = YOLO('yolov8n.pt')
            logger.info("✅ Using pre-trained YOLOv8 for apple detection")
        except Exception as e:
            logger.warning("⚠️ Apple detector not found, using dummy")
            self.apple_detector = None
        
        # Load Variety Classifier - FORCE LOAD THE BEST MODEL
        try:
            best_variety_path = os.path.join(self.models_dir, 'best_variety_model.pth')
            logger.info(f"Looking for best model at: {best_variety_path}")
            logger.info(f"File exists: {os.path.exists(best_variety_path)}")
            
            if os.path.exists(best_variety_path):
                logger.info("🔄 Loading BEST variety classifier...")
                
                # Create the exact model architecture
                self.variety_model = models.efficientnet_b0(pretrained=False)
                self.variety_model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(1280, 128),  # EfficientNet-B0 has 1280 features
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(128, 6)
                )
                
                # Load the state dict
                state_dict = torch.load(best_variety_path, map_location=self.device)
                self.variety_model.load_state_dict(state_dict)
                self.variety_model.eval()
                self.variety_model = self.variety_model.to(self.device)
                
                logger.info("✅ BEST variety classifier loaded successfully!")
                
                # Test the model immediately
                self._test_variety_model()
                
            else:
                logger.warning("⚠️ Best variety classifier not found, using dummy")
                self.variety_model = None
                
        except Exception as e:
            logger.error(f"❌ Error loading variety classifier: {e}")
            self.variety_model = None
        
        # Load Health Classifier
        try:
            best_health_path = os.path.join(self.models_dir, 'best_health_model.pth')
            health_path = os.path.join(self.models_dir, 'health_classifier.pt')
            
            if os.path.exists(best_health_path):
                logger.info("🔄 Loading BEST health classifier...")
                
                # Create the exact model architecture
                self.health_model = models.resnet18(pretrained=False)
                self.health_model.fc = torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(self.health_model.fc.in_features, 64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(64, 2)
                )
                
                # Load the state dict
                state_dict = torch.load(best_health_path, map_location=self.device)
                self.health_model.load_state_dict(state_dict)
                self.health_model.eval()
                self.health_model = self.health_model.to(self.device)
                
                logger.info("✅ BEST health classifier loaded successfully!")
                
            elif os.path.exists(health_path):
                self.health_model = torch.load(health_path, map_location=self.device)
                self.health_model.eval()
                logger.info("✅ Health classifier loaded successfully")
            else:
                logger.warning("⚠️ Health classifier not found, using dummy")
                self.health_model = None
                
        except Exception as e:
            logger.error(f"❌ Error loading health classifier: {e}")
            self.health_model = None
        
        # Load Surface Classifier
        try:
            best_surface_path = os.path.join(self.models_dir, 'best_surface_model.pth')
            surface_path = os.path.join(self.models_dir, 'surface_classifier.pt')
            
            if os.path.exists(best_surface_path):
                logger.info("🔄 Loading BEST surface classifier...")
                
                # Create the exact model architecture
                self.surface_model = models.efficientnet_b1(pretrained=False)
                self.surface_model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(1280, 256),  # EfficientNet-B1 has 1280 features
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(256),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(256, 64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(64, 2)
                )
                
                # Load the state dict
                state_dict = torch.load(best_surface_path, map_location=self.device)
                self.surface_model.load_state_dict(state_dict)
                self.surface_model.eval()
                self.surface_model = self.surface_model.to(self.device)
                
                logger.info("✅ BEST surface classifier loaded successfully!")
                
            elif os.path.exists(surface_path):
                self.surface_model = torch.load(surface_path, map_location=self.device)
                self.surface_model.eval()
                logger.info("✅ Surface classifier loaded successfully")
            else:
                logger.warning("⚠️ Surface classifier not found, using dummy")
                self.surface_model = None
                
        except Exception as e:
            logger.error(f"❌ Error loading surface classifier: {e}")
            self.surface_model = None
        
        # Load Shelf Life Predictor
        try:
            best_shelf_life_path = os.path.join(self.models_dir, 'best_shelf_life_model.pth')
            shelf_life_path = os.path.join(self.models_dir, 'shelf_life_predictor.pt')
            
            if os.path.exists(best_shelf_life_path):
                logger.info("🔄 Loading BEST shelf life predictor...")
                
                # Create the exact model architecture
                class ShelfLifePredictor(torch.nn.Module):
                    def __init__(self, input_size=9):
                        super(ShelfLifePredictor, self).__init__()
                        self.network = torch.nn.Sequential(
                            torch.nn.Linear(input_size, 64),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm1d(64),
                            torch.nn.Dropout(0.2),
                            torch.nn.Linear(64, 32),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm1d(32),
                            torch.nn.Dropout(0.1),
                            torch.nn.Linear(32, 16),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.1),
                            torch.nn.Linear(16, 1),
                            torch.nn.ReLU()
                        )
                    def forward(self, x):
                        return self.network(x)
                
                self.shelf_life_model = ShelfLifePredictor(input_size=9)
                
                # Load the state dict
                state_dict = torch.load(best_shelf_life_path, map_location=self.device)
                self.shelf_life_model.load_state_dict(state_dict)
                self.shelf_life_model.eval()
                self.shelf_life_model = self.shelf_life_model.to(self.device)
                
                logger.info("✅ BEST shelf life predictor loaded successfully!")
                
            elif os.path.exists(shelf_life_path):
                self.shelf_life_model = torch.load(shelf_life_path, map_location=self.device)
                self.shelf_life_model.eval()
                logger.info("✅ Shelf life predictor loaded successfully")
            else:
                logger.warning("⚠️ Shelf life predictor not found, using dummy")
                self.shelf_life_model = None
                
        except Exception as e:
            logger.error(f"❌ Error loading shelf life predictor: {e}")
            self.shelf_life_model = None
    
    def _test_variety_model(self):
        """Test the variety model immediately after loading."""
        if self.variety_model is None:
            return
            
        try:
            logger.info("🧪 Testing variety model...")
            
            # Create a test image
            test_image = Image.new('RGB', (224, 224), color=(255, 0, 0))  # Red image
            input_tensor = self._preprocess_image(test_image)
            
            with torch.no_grad():
                outputs = self.variety_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                variety = self.varieties[predicted_idx.item()]
                conf_score = confidence.item()
                
                logger.info(f"✅ Test prediction: {variety} (confidence: {conf_score:.3f})")
                
                # Check if model is working properly
                probs = probabilities[0].tolist()
                max_prob = max(probs)
                min_prob = min(probs)
                
                if max_prob - min_prob > 0.1:
                    logger.info("✅ Model appears to be working correctly (good probability distribution)")
                else:
                    logger.warning("⚠️ Model may be overfitted (similar probabilities)")
                    
        except Exception as e:
            logger.error(f"❌ Error testing variety model: {e}")
    
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
    
    def detect_apple(self, image: Image.Image) -> Dict[str, any]:
        """
        🍎 OBJECTIVE 1: Apple Detection (MANDATORY FIRST STEP)
        Uses YOLOv8 to detect if image contains an apple.
        Returns: {"detected": bool, "confidence": float, "message": str}
        """
        try:
            if self.apple_detector is None:
                # More realistic dummy logic
                import random
                is_apple = random.random() < 0.7  # 70% chance for testing
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
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()
                        
                        # Log all detected objects for debugging
                        detected_objects.append(f"Class {class_id} (conf: {confidence:.3f})")
                        
                        if class_id == 47 and confidence > 0.3:  # Apple class with 30% confidence (lower for rotten apples)
                            apple_detected = True
                            max_confidence = max(max_confidence, confidence)
            
            # Debug logging
            if detected_objects:
                logger.info(f"🔍 YOLO detected objects: {', '.join(detected_objects)}")
            else:
                logger.info("🔍 YOLO detected no objects")
            
            if apple_detected:
                return {
                    "detected": True,
                    "confidence": max_confidence,
                    "message": f"🍎 Apple detected successfully (Confidence: {max_confidence:.1%})"
                }
            else:
                # Fallback: Check if any fruit-like objects were detected
                fruit_classes = [47, 52, 53, 54, 55, 56]  # apple, banana, orange, etc.
                fruit_detected = any(int(obj.split()[1]) in fruit_classes for obj in detected_objects if len(obj.split()) > 1)
                
                if fruit_detected:
                    logger.info("🍎 Fallback: Fruit-like object detected, assuming degraded apple")
                    return {
                        "detected": True,
                        "confidence": 0.5,  # Medium confidence for fallback detection
                        "message": "🍎 Apple detected (degraded/rotten apple detected via fallback)"
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
        Classifies apple variety using the BEST trained model.
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
            
            logger.info("🔍 Using BEST trained variety model for prediction")
            
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
                logger.info(f"🎯 FIXED Variety prediction - Index: {predicted_idx.item()}, Variety: {variety}, Confidence: {conf_score:.3f}")
                
                # Get top 3 predictions for better insight
                top3_probs, top3_indices = torch.topk(probabilities, 3)
                top3_varieties = [self.varieties[idx.item()] for idx in top3_indices[0]]
                top3_scores = [prob.item() for prob in top3_probs[0]]
                
                logger.info(f"🏆 Top 3: {list(zip(top3_varieties, top3_scores))}")
                
                if conf_score > 0.60:  # Reasonable threshold
                    # Include top 2 in message for transparency
                    top2_str = ", ".join([f"{var}({score:.1%})" for var, score in zip(top3_varieties[:2], top3_scores[:2])])
                    return {
                        "variety": variety,
                        "confidence": conf_score,
                        "message": f"🍎 Predicted Variety: {variety} (Confidence: {conf_score:.1%}) | Top 2: {top2_str}"
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
        💚❤️ OBJECTIVE 3: Health Classification (Healthy vs Rotten)
        Returns: {"health": str, "confidence": float, "message": str}
        """
        try:
            if self.health_model is None:
                # Dummy implementation
                import random
                is_healthy = random.choice([True, False])
                confidence = random.uniform(0.85, 0.98)
                
                health_status = "Healthy" if is_healthy else "Rotten"
                emoji = "💚" if is_healthy else "❤️"
                
                return {
                    "health": health_status,
                    "confidence": confidence,
                    "message": f"{emoji} Condition: {health_status} (Confidence: {confidence:.1%}) [Dummy Mode]"
                }
            
            logger.info("🔍 Using BEST trained health model for prediction")
            
            # Preprocess image
            input_tensor = self._preprocess_image(image)
            
            # Model inference
            with torch.no_grad():
                outputs = self.health_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Classes: 0=healthy, 1=rotten
                health_classes = ['Healthy', 'Rotten']
                health_status = health_classes[predicted_idx.item()]
                conf_score = confidence.item()
                
                # Debug logging
                logger.info(f"🏥 Health prediction - Index: {predicted_idx.item()}, Health: {health_status}, Confidence: {conf_score:.3f}")
                
                # Get probabilities for both classes
                healthy_prob = probabilities[0][0].item()
                rotten_prob = probabilities[0][1].item()
                
                logger.info(f"🏥 Probabilities - Healthy: {healthy_prob:.3f}, Rotten: {rotten_prob:.3f}")
                
                emoji = "💚" if health_status == "Healthy" else "❤️"
                
                if conf_score > 0.70:  # Good confidence threshold
                    return {
                        "health": health_status,
                        "confidence": conf_score,
                        "message": f"{emoji} Condition: {health_status} (Confidence: {conf_score:.1%})"
                    }
                else:
                    return {
                        "health": "Uncertain",
                        "confidence": conf_score,
                        "message": f"⚠️ Uncertain health status: {health_status} ({conf_score:.1%})"
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
        Returns: {"surface": str, "confidence": float, "message": str}
        """
        try:
            if self.surface_model is None:
                # Dummy implementation
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
            
            logger.info("🔍 Using BEST trained surface model for prediction")
            
            # Preprocess image
            input_tensor = self._preprocess_image(image)
            
            # Model inference
            with torch.no_grad():
                outputs = self.surface_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Classes: 0=waxed, 1=unwaxed
                surface_classes = ['Waxed', 'Unwaxed']
                surface_type = surface_classes[predicted_idx.item()]
                conf_score = confidence.item()
                
                # Debug logging
                logger.info(f"🧴 Surface prediction - Index: {predicted_idx.item()}, Surface: {surface_type}, Confidence: {conf_score:.3f}")
                
                # Get probabilities for both classes
                waxed_prob = probabilities[0][0].item()
                unwaxed_prob = probabilities[0][1].item()
                
                logger.info(f"🧴 Probabilities - Waxed: {waxed_prob:.3f}, Unwaxed: {unwaxed_prob:.3f}")
                
                emoji = "🧴" if surface_type == "Waxed" else "🧼"
                
                if conf_score > 0.75:  # Good confidence threshold
                    return {
                        "surface": surface_type,
                        "confidence": conf_score,
                        "message": f"{emoji} Surface: {surface_type} (Confidence: {conf_score:.1%})"
                    }
                else:
                    return {
                        "surface": "Uncertain",
                        "confidence": conf_score,
                        "message": f"⚠️ Uncertain surface type: {surface_type} ({conf_score:.1%})"
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
        ⏰ OBJECTIVE 5: Shelf Life Prediction
        Returns: {"shelf_life": int, "message": str}
        """
        try:
            if self.shelf_life_model is None:
                # Dummy implementation
                import random
                
                if health == "Rotten":
                    shelf_life = 0
                else:
                    base_days = self.base_shelf_life.get(variety, 18)
                    if surface == "Waxed":
                        shelf_life = int(base_days * 1.5)  # Waxed lasts longer
                    else:
                        shelf_life = base_days
                    
                    # Add some randomness
                    shelf_life += random.randint(-2, 3)
                    shelf_life = max(0, shelf_life)
                
                return {
                    "shelf_life": shelf_life,
                    "message": f"⏰ Estimated Shelf Life: {shelf_life} days [Dummy Mode]"
                }
            
            logger.info("🔍 Using BEST trained shelf life predictor")
            
            # Prepare input features
            variety_to_idx = {v: i for i, v in enumerate(self.varieties)}
            variety_idx = variety_to_idx.get(variety, 0)
            
            # Create feature vector: [variety_one_hot(6), health(1), surface(1), base_days(1)]
            variety_onehot = torch.zeros(6)
            variety_onehot[variety_idx] = 1.0
            
            health_val = torch.tensor([1.0 if health.lower() == "rotten" else 0.0])
            surface_val = torch.tensor([0.0 if surface.lower() == "waxed" else 1.0])
            base_days_val = torch.tensor([float(self.base_shelf_life.get(variety, 18))])
            
            # Combine features
            features = torch.cat([variety_onehot, health_val, surface_val, base_days_val])
            features = features.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Model inference
            with torch.no_grad():
                output = self.shelf_life_model(features)
                predicted_days = output.item()
                
                # Round to nearest integer and ensure non-negative
                shelf_life = max(0, round(predicted_days))
                
                # Debug logging
                logger.info(f"⏰ Shelf life prediction - Variety: {variety}, Health: {health}, Surface: {surface}")
                logger.info(f"⏰ Predicted: {predicted_days:.2f} days, Rounded: {shelf_life} days")
                
                # Create appropriate message
                if shelf_life == 0:
                    message = "⏰ Shelf Life: 0 days (consume immediately or discard)"
                elif shelf_life <= 3:
                    message = f"⏰ Shelf Life: {shelf_life} days (consume soon)"
                elif shelf_life <= 7:
                    message = f"⏰ Shelf Life: {shelf_life} days (good for this week)"
                else:
                    message = f"⏰ Shelf Life: {shelf_life} days (excellent longevity)"
                
                return {
                    "shelf_life": shelf_life,
                    "message": message
                }
                    
        except Exception as e:
            logger.error(f"Error in shelf life prediction: {e}")
            return {
                "shelf_life": 0,
                "message": f"❌ Error during shelf life prediction: {str(e)}"
            }
    
    def analyze_apple(self, image: Image.Image) -> Dict[str, any]:
        """
        🍎 COMPLETE APPLE ANALYSIS PIPELINE
        Runs all objectives in sequence with proper error handling.
        """
        results = {
            "apple_detection": {},
            "variety_classification": {},
            "health_prediction": {},
            "surface_prediction": {},
            "shelf_life_prediction": {},
            "overall_success": False
        }
        
        try:
            # Step 1: Apple Detection (MANDATORY)
            apple_result = self.detect_apple(image)
            results["apple_detection"] = apple_result
            
            if not apple_result["detected"]:
                # If no apple detected, stop here
                results["variety_classification"] = {"variety": "N/A", "confidence": 0.0, "message": "⚠️ Skipped - No apple detected"}
                results["health_prediction"] = {"health": "N/A", "confidence": 0.0, "message": "⚠️ Skipped - No apple detected"}
                results["surface_prediction"] = {"surface": "N/A", "confidence": 0.0, "message": "⚠️ Skipped - No apple detected"}
                results["shelf_life_prediction"] = {"shelf_life": 0, "message": "⚠️ Skipped - No apple detected"}
                return results
            
            # Step 2: Variety Classification
            variety_result = self.classify_variety(image)
            results["variety_classification"] = variety_result
            
            # Step 3: Health Prediction
            health_result = self.predict_health(image)
            results["health_prediction"] = health_result
            
            # Step 4: Surface Prediction
            surface_result = self.predict_surface(image)
            results["surface_prediction"] = surface_result
            
            # Step 5: Shelf Life Prediction
            shelf_life_result = self.predict_shelf_life(
                variety_result["variety"],
                health_result["health"],
                surface_result["surface"]
            )
            results["shelf_life_prediction"] = shelf_life_result
            
            results["overall_success"] = True
            
        except Exception as e:
            logger.error(f"Error in complete apple analysis: {e}")
            results["error"] = str(e)
        
        return results
