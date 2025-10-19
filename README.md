# 🍏 Indian Apple Intelligence System

**AI-Powered Variety, Freshness, Surface & Shelf-Life Analysis**

A professional, modular, and highly accurate Apple Analysis System built for final-year computer science projects using only free tools and local resources. The system runs entirely offline with no premium dependencies.

## 🎯 Project Overview

This system analyzes Indian apple varieties using state-of-the-art AI models to provide:

- **🍎 Variety Classification**: Identifies 6 Indian apple varieties (Sharbati, Sunehari, Maharaji, Splendour, Himsona)
- **💚 Health Assessment**: Detects healthy vs rotten apples with explainable AI
- **🧴 Surface Analysis**: Classifies waxed vs unwaxed surfaces using hybrid CNN + texture features
- **⏳ Shelf Life Prediction**: Estimates remaining shelf life based on variety, health, and surface

## ✨ Key Features

- **🔒 100% Offline**: No internet required after installation
- **🆓 Free & Open Source**: Uses only free tools and libraries
- **🧩 Modular Design**: Independent objectives that don't interfere with each other
- **🎨 Professional UI**: Startup-grade interface with green agri-tech theme
- **🎓 Academic Ready**: Perfect for final-year projects and presentations
- **⚡ CPU Optimized**: Runs on any computer without GPU requirements

## 🏗️ Architecture

### Core Principles
- **Apple Detection First**: Every image is validated for apple presence using YOLOv8
- **Modular Independence**: All four objectives run independently and in parallel
- **High Precision**: Uses confidence thresholds (>85%) and comprehensive error handling
- **Local Resources**: All models and data stored locally

### Technical Stack
- **Backend**: Python 3.9+, PyTorch, scikit-learn
- **Computer Vision**: YOLOv8, EfficientNet-B3, ResNet18
- **Frontend**: Gradio (professional web interface)
- **ML Models**: CNN, Random Forest, Texture Analysis

## 📁 Project Structure

```
indian-apple-ai/
├── data/                          # Training datasets
│   ├── varieties/                 # Apple variety images
│   │   ├── Sharbati/
│   │   ├── Sunehari/
│   │   ├── Maharaji/
│   │   ├── Splendour/
│   │   ├── Himsona/
│   │   └── Himkiran/
│   ├── health/                    # Health classification images
│   │   ├── healthy/
│   │   └── rotten/
│   └── surface/                   # Surface classification images
│       ├── waxed/
│       └── unwaxed/
├── models/                        # Trained AI models
│   ├── apple_detector.pt          # YOLOv8 apple detector
│   ├── variety_classifier.pt      # EfficientNet-B3 variety classifier
│   ├── health_classifier.pt       # ResNet18 health classifier
│   ├── surface_classifier.pt      # CNN surface classifier
│   └── shelf_life_model.pkl       # Random Forest shelf life predictor
├── app/                           # Main application
│   ├── core/
│   │   └── pipeline.py            # Core AI pipeline
│   └── app.py                     # Gradio web interface
├── scripts/                       # Utility scripts
│   ├── train_variety.py           # Training scripts (for future use)
│   └── save_dummy_models.py       # Generate dummy models for testing
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd indian-apple-ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dummy Models (for testing)

```bash
# Create dummy models for development
python scripts/save_dummy_models.py
```

### 3. Launch Application

```bash
# Start the web interface
python app/app.py
```

The application will open in your browser at `http://127.0.0.1:7860`

## 🖥️ User Interface

### Professional Features
- **🎨 Modern Design**: Green-themed agri-tech interface
- **📱 Responsive**: Works on desktop, tablet, and mobile
- **⚡ Real-time Analysis**: Progress tracking with visual feedback
- **📊 Detailed Results**: Confidence scores and explanations
- **🔄 Easy Reset**: One-click interface reset
- **📄 Report Generation**: PDF export capability (coming soon)

### Analysis Workflow
1. **Upload Image**: Drag & drop or click to upload apple image
2. **Apple Detection**: System validates apple presence (mandatory first step)
3. **Parallel Analysis**: Four independent AI models analyze the apple
4. **Results Display**: Professional cards showing all predictions
5. **Export Options**: Download detailed analysis report

## 🧠 AI Models & Accuracy

### 1. Apple Detector (YOLOv8)
- **Purpose**: Validates apple presence in uploaded images
- **Architecture**: YOLOv8 nano for fast inference
- **Threshold**: 85% confidence for apple detection
- **Fallback**: Clear error message if no apple detected

### 2. Variety Classifier (EfficientNet-B3)
- **Purpose**: Identifies specific Indian apple variety
- **Classes**: 5 varieties (Sharbati, Sunehari, Maharaji, Splendour, Himsona)
- **Architecture**: Fine-tuned EfficientNet-B3
- **Expected Accuracy**: >90% on test set

### 3. Health Classifier (ResNet18)
- **Purpose**: Determines apple freshness (Healthy vs Rotten)
- **Architecture**: ResNet18 with texture analysis
- **Features**: Grad-CAM explainability for decision visualization
- **Expected Accuracy**: >95% on balanced dataset

### 4. Surface Classifier (Hybrid CNN)
- **Purpose**: Detects waxed vs unwaxed apple surfaces
- **Architecture**: CNN + Local Binary Pattern (LBP) + GLCM texture features
- **Innovation**: Combines deep learning with traditional texture analysis
- **Expected Accuracy**: >88% on surface texture dataset

### 5. Shelf Life Predictor (Random Forest)
- **Purpose**: Estimates remaining shelf life in days
- **Input Features**: Variety, health status, surface type
- **Logic**: 
  - Base days per variety (e.g., Himsona = 18 days)
  - Rotten apples = 0 days
  - Waxed apples = 1.5x multiplier
- **Model**: Random Forest Regressor with 100 trees

## 📊 Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Apple Detection | >95% | YOLOv8 with confidence threshold |
| Variety Classification | >90% | 5-class classification |
| Health Assessment | >95% | Binary classification (Healthy/Rotten) |
| Surface Analysis | >88% | Binary classification (Waxed/Unwaxed) |
| Shelf Life MAE | <2 days | Mean Absolute Error |
| Processing Time | <5 seconds | Per image on CPU |

## 🔧 Development & Training

### Replace Dummy Models

1. **Collect Training Data**: Place images in respective `data/` folders
2. **Train Models**: Use provided training scripts or your own
3. **Replace Models**: Save trained models in `models/` directory
4. **Test System**: Verify accuracy with real models

### Training Tips
- **Data Quality**: Use high-resolution, well-lit images
- **Data Balance**: Ensure equal samples per class
- **Augmentation**: Apply rotation, scaling, color adjustments
- **Validation**: Use separate test set for unbiased evaluation

## 🎓 Academic Excellence

### Project Highlights
- **Innovation**: Hybrid CNN + texture analysis for surface classification
- **Modularity**: Independent objectives with no cross-dependencies
- **Reliability**: Comprehensive error handling and fallbacks
- **Scalability**: Easy to extend with new varieties or features
- **Documentation**: Professional code documentation and comments

### Presentation Points
- **Problem Solving**: Addresses real agricultural challenges
- **Technical Depth**: Multiple AI/ML techniques integrated
- **User Experience**: Professional, intuitive interface
- **Practical Impact**: Reduces food waste through shelf life prediction
- **Cost Effective**: 100% free solution for farmers and retailers

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Models not loading
```bash
# Solution: Generate dummy models first
python scripts/save_dummy_models.py
```

**Issue**: Gradio interface not opening
```bash
# Solution: Check if port 7860 is available
# Try different port in app.py: server_port=7861
```

**Issue**: CUDA/GPU errors
```bash
# Solution: System is designed for CPU-first operation
# Models automatically fallback to CPU
```

**Issue**: Import errors
```bash
# Solution: Install all requirements
pip install -r requirements.txt
```

## 🤝 Contributing

This is an academic project, but improvements are welcome:

1. **Data Collection**: Contribute apple images for training
2. **Model Improvement**: Enhance accuracy with better architectures
3. **Feature Addition**: Add new analysis capabilities
4. **UI Enhancement**: Improve user interface and experience
5. **Documentation**: Expand guides and tutorials

## 📜 License

This project is open-source and available under the MIT License. Perfect for academic use, learning, and further development.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **EfficientNet**: Google for efficient CNN architecture
- **ResNet**: Microsoft Research for residual networks
- **Gradio**: Hugging Face for the amazing web interface framework
- **PyTorch**: Facebook AI Research for the deep learning framework

## 📞 Support

For questions, issues, or contributions:

1. **Check Documentation**: Review this README and code comments
2. **Test with Dummy Models**: Ensure basic functionality works
3. **Check Requirements**: Verify all dependencies are installed
4. **Review Logs**: Check console output for error details

---

**🎯 Built for Excellence | 🎓 Academic Ready | 🆓 100% Free | 🔒 Fully Offline**

*This project demonstrates the power of free, open-source AI tools in solving real-world agricultural challenges. Perfect for final-year computer science students looking to make an impact!*
