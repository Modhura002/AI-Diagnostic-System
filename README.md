# 🧠 AI Diagnostic System

An AI-powered medical image diagnosis web application built using **Flask**, **TensorFlow/Keras**, and **Transfer Learning**.

This project aims to provide an easy-to-use interface for diagnosing diseases from medical images using deep learning models along with Grad-CAM visualizations for model explainability.

> 🚧 **Project Status:** Under Development

Currently Completed:
- ✅ Brain Tumor Diagnosis
- ✅ Flask Web Application
- ✅ Grad-CAM Visualization

Upcoming Modules:
- 🔄 Pneumonia Detection
- 🔄 Skin Disease Detection

---

## Features

- Upload MRI images through a web interface
- Brain tumor classification using ResNet50
- Confidence score prediction
- Grad-CAM heatmap visualization
- Modular Flask application structure
- Easy to extend with additional AI models

---

## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- ResNet50 (Transfer Learning)
- OpenCV
- NumPy
- HTML

---

## Project Structure

```
AI-Diagnostic-System/

├── app.py
├── models/
│   └── brain_model.keras
├── static/
│   ├── uploads/
│   └── results/
├── templates/
│   ├── index.html
│   └── result.html
├── utils/
│   ├── brain_predict.py
│   └── gradcam.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Brain Tumor Classes

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

---

## Current Workflow

```
Upload MRI
      ↓
Flask Web App
      ↓
Brain Tumor Model (ResNet50)
      ↓
Prediction + Confidence
      ↓
Grad-CAM Generation
      ↓
Display Results
```

---

## Future Improvements

- Pneumonia Diagnosis Module
- Skin Disease Diagnosis Module
- Modern Responsive UI
- Disease Information
- Downloadable Prediction Reports
- Docker Deployment

---

## Screenshots

Screenshots will be added after completion of all modules.

---

## Author

**Modhura Banerjee**