# 🎯 Object Detection Dashboard

A web-based Object Detection Dashboard built with **Flask**, **OpenCV**, **PyTorch**, and **Ultralytics YOLO**. Upload an image or video and run real-time object detection using multiple state-of-the-art models — all from your browser.

---

## 🚀 Features

- 📷 **Image Detection** — Upload `.jpg`, `.jpeg`, or `.png` files
- 🎥 **Video Detection** — Upload `.mp4`, `.avi`, or `.mov` files
- 🤖 **Multiple Models Supported:**
  - **YOLOv5** (Ultralytics) — Fast & accurate
  - **SSD** (SSDLite320 + MobileNetV3) — Lightweight
  - **Faster R-CNN** (ResNet50 FPN) — High accuracy
  - **Haar Cascade** — Face detection (OpenCV classic)
- 📊 **Object Count Summary** — Shows detected object labels and counts
- 🖼️ **Annotated Output** — Bounding boxes drawn directly on the result

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Detection | Ultralytics YOLO, PyTorch, TorchVision |
| Image Processing | OpenCV (cv2) |
| Frontend | HTML, Jinja2 Templates |
| Models | YOLOv5, SSD, Faster R-CNN, Haar Cascade |

---

## 📁 Project Structure

```
Object-Detection-Dashboard/
│
├── jana pyhton.py                   # Main Flask application
├── Object_Detection_Dashboard.ipynb # Jupyter Notebook demo
├── PROJECT DOCUMENTATION.pdf        # Full project documentation
├── IEEE PAPER(48,G7).pdf            # IEEE Conference Paper
├── IEEE_Conference_Template.pdf     # IEEE Template used
├── 48,G7.pptx                       # Project Presentation
├── Screenshot *.png                 # Demo screenshots
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Janardhan-Jillela/Object-Detection-Dashboard.git
cd Object-Detection-Dashboard
```

### 2. Install dependencies
```bash
pip install flask opencv-python torch torchvision ultralytics numpy
```

### 3. Run the application
```bash
python "jana pyhton.py"
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 🧠 Supported Models

### YOLOv5 (Ultralytics)
- Real-time detection on 80 COCO classes
- Model file: `yolov5su.pt` (auto-downloaded on first run)

### SSD — SSDLite320 MobileNetV3
- Lightweight, fast inference
- Pre-trained on COCO dataset

### Faster R-CNN — ResNet50 FPN
- High accuracy two-stage detector
- Pre-trained on COCO dataset

### Haar Cascade (OpenCV)
- Classic face detection
- No GPU required

---

## 📸 Demo Screenshots

![Screenshot](Screenshot%202025-10-09%20102802.png)

---

## 📄 Documentation

- 📘 [Project Documentation (PDF)](PROJECT%20DOCUMENTATION.pdf)
- 📝 [IEEE Conference Paper](IEEE%20PAPER(48%2CG7).pdf)
- 📓 [Jupyter Notebook](Object_Detection_Dashboard.ipynb)

---

## 👤 Author

**Janardhan Jillela**  
GitHub: [@Janardhan-Jillela](https://github.com/Janardhan-Jillela)

---

## 📜 License

This project is for academic and educational purposes.
