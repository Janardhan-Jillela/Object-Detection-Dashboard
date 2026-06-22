# 🎯 Object Detection Dashboard

> A real-time AI-powered object detection web app built with **Streamlit**, **YOLOv8**, **PyTorch**, and **OpenCV** — featuring image, video, and live webcam detection modes.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=flat-square&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-green?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📷 **Image Detection** | Upload JPG/PNG images and get annotated results instantly |
| 🎞️ **Video Detection** | Process full videos frame-by-frame with progress tracking |
| 📹 **Live Webcam** | Real-time detection directly from your webcam |
| 🔍 **Object Tracking** | YOLOv8 tracking assigns unique IDs to objects across frames |
| 📊 **Detection Summary** | Object counts, confidence scores, inference time |
| 💾 **Download Results** | Save annotated images and processed videos |
| 🎨 **Premium Dark UI** | Modern glassmorphism design with gradient accents |

---

## 🤖 Supported Models

| Model | Type | Best For | Speed |
|---|---|---|---|
| **YOLOv8** (n/s/m/l/x) | One-stage detector | General purpose, real-time | ⚡ Fastest |
| **SSD** (MobileNetV3) | One-stage detector | Lightweight inference | 🚀 Fast |
| **Faster R-CNN** | Two-stage detector | High accuracy | 🎯 Accurate |
| **Haar Cascade** | Classical CV | Face detection only | ✅ CPU-friendly |

All models detect from **80 COCO classes** (person, car, dog, etc.) — Haar detects faces.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit, Custom CSS |
| **Detection Engine** | Ultralytics YOLOv8, PyTorch, TorchVision |
| **Image Processing** | OpenCV (cv2), NumPy |
| **Deep Learning** | PyTorch 2.0+, CUDA support |
| **Language** | Python 3.8+ |

---

## 📁 Project Structure

```
Object-Detection-Dashboard/
│
├── app.py                           # 🚀 Main Streamlit application
├── style.css                        # 🎨 Custom dark theme styling
├── requirements.txt                 # 📦 Python dependencies
├── Object_Detection_Dashboard.ipynb # 📓 Jupyter Notebook version
├── jana pyhton.py                   # 🔧 Flask version (alternative)
│
├── PROJECT DOCUMENTATION.pdf        # 📄 Full project documentation
├── PROJECT DOCUMENTATION.docx       # 📝 Editable documentation
├── IEEE PAPER(48,G7).pdf            # 📰 IEEE Conference Paper
├── IEEE_Conference_Template.pdf     # 📋 IEEE Template
├── 48,G7.pptx                       # 📊 Project Presentation
│
├── Screenshot *.png                 # 🖼️ Demo screenshots
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Janardhan-Jillela/Object-Detection-Dashboard.git
cd Object-Detection-Dashboard
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

### 5. Open in Browser
```
http://localhost:8501
```

---

## 🚀 Usage Guide

### Image Detection
1. Select a model from the sidebar
2. Adjust the confidence threshold
3. Go to **📷 Image Detection** tab
4. Upload an image (JPG, JPEG, PNG)
5. View annotated results and download

### Video Detection
1. Select a model and configure frame skip
2. Go to **🎞️ Video Detection** tab
3. Upload a video (MP4, AVI, MOV)
4. Watch real-time frame-by-frame processing
5. Download the processed video

### Live Webcam
1. Select a model and enable tracking (YOLOv8)
2. Go to **📹 Live Webcam** tab
3. Click **▶️ Start Webcam**
4. View live detections with FPS counter
5. Click **⏹️ Stop Webcam** when done

---

## 📸 Screenshots

![Dashboard Screenshot](Screenshot%202025-10-09%20102802.png)

---

## 📄 Documentation & Research

- 📘 [Project Documentation (PDF)](PROJECT%20DOCUMENTATION.pdf)
- 📝 [Project Documentation (DOCX)](PROJECT%20DOCUMENTATION.docx)
- 📰 [IEEE Conference Paper](IEEE%20PAPER(48%2CG7).pdf)
- 📓 [Jupyter Notebook Version](Object_Detection_Dashboard.ipynb)
- 📊 [Project Presentation](48%2CG7.pptx)

---

## 🧑‍💻 Alternative: Flask Version

A lightweight Flask-based version is also available:

```bash
python "jana pyhton.py"
# Open: http://localhost:5000
```

---

## 👤 Author


**Janardhan Jillela**
GitHub: [@Janardhan-Jillela](https://github.com/Janardhan-Jillela)
**Pujala Venkata Swathi**
GitHub: [@Swathi1596Srinivas](https://github.com/Swathi1596Srinivas).
---

## 📜 License

This project is developed for **academic and research purposes** as part of an IEEE conference submission.
