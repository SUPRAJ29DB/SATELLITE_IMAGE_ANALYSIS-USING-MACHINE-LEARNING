# 🛰️ Satellite Image Classification System

An AI-based Satellite Image Classification System developed using Deep Learning and Transfer Learning techniques.

This project classifies satellite images into different land cover categories using the EuroSAT RGB dataset and MobileNetV2 architecture.

---

# 📌 Project Overview

Satellite image classification is an important application of remote sensing and computer vision.

This project uses:
- TensorFlow
- MobileNetV2
- Streamlit
- OpenCV

to classify satellite images into different land use and land cover classes.

---

# 🚀 Features

✅ Satellite Image Upload  
✅ Deep Learning Based Classification  
✅ MobileNetV2 Transfer Learning  
✅ Confidence Score Prediction  
✅ Grad-CAM Heatmap Visualization  
✅ Streamlit Web Application  
✅ Accuracy and Loss Graphs  
✅ Confusion Matrix  
✅ Professional UI Design  

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| TensorFlow/Keras | Deep Learning |
| OpenCV | Image Processing |
| Streamlit | Web Application |
| Matplotlib | Visualization |
| Seaborn | Confusion Matrix |
| NumPy | Numerical Computation |

---

# 📂 Dataset

Dataset Used:
## EuroSAT RGB Dataset

The dataset contains 10 classes:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

---

# 🏗️ Project Architecture

```text
Satellite Image
        ↓
Image Preprocessing
        ↓
MobileNetV2 Model
        ↓
Feature Extraction
        ↓
Classification Layer
        ↓
Prediction Result
        ↓
Grad-CAM Visualization
📊 Model Features
Transfer Learning using MobileNetV2
Data Augmentation
Early Stopping
Accuracy Evaluation
Confusion Matrix
Explainable AI using Grad-CAM
📈 Evaluation Metrics

The model is evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
🖥️ Streamlit Dashboard Features
Upload satellite image
Predict land cover class
Show confidence score
Display Grad-CAM heatmap
Interactive visualization
⚙️ Installation
Clone Repository
git clone https://github.com/your-username/satellite-image-classification.git
Navigate to Project Folder
cd satellite-image-classification
Install Dependencies
pip install -r requirements.txt
▶️ Run Project
Train Model
python train.py
Run Streamlit App
streamlit run app.py
📁 Project Structure
satellite-image-classification/
│
├── app.py
├── train.py
├── satellite_model.h5
├── requirements.txt
├── README.md
├── EuroSAT/
└── screenshots/
🧪 Grad-CAM Visualization

Grad-CAM is used for Explainable AI (XAI).

It highlights the important regions of the satellite image used by the CNN for prediction.

🎯 Future Scope
Multispectral Satellite Analysis
GeoTIFF Support
Real-Time Satellite Monitoring
Flood Detection
Forest Fire Detection
Crop Health Monitoring
Object Detection using YOLO
Semantic Segmentation using U-Net
🌍 Applications
Remote Sensing
Smart Agriculture
Disaster Management
Urban Planning
Environmental Monitoring
Land Cover Mapping
