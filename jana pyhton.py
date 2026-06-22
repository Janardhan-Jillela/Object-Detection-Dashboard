from flask import Flask, render_template, request
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from ultralytics import YOLO
from collections import Counter
from io import BytesIO
import base64
import tempfile
import time

app = Flask(__name__)

# --------------------------
# Helper functions
# --------------------------
def get_coco_labels():
    return [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

def load_model(model_name):
    try:
        if model_name == "YOLO":
            return YOLO("yolov5su.pt")
        elif model_name == "SSD":
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
            model.eval()
            return model
        elif model_name == "Faster R-CNN":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            model.eval()
            return model
        elif model_name == "Haar Cascade":
            return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None

# --------------------------
# Detection functions
# --------------------------
def detect_yolo(img, model):
    objects = []
    results = model(img, verbose=False)
    output = results[0].plot()
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        objects.append(label)
    return output, objects

def detect_torchvision(img, model):
    objects = []
    labels_list = get_coco_labels()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor_img = transforms.ToTensor()(img_rgb)
    with torch.no_grad():
        preds = model([tensor_img])[0]
    for i, score in enumerate(preds["scores"]):
        if score > 0.5:
            box = preds["boxes"][i].cpu().numpy().astype(int)
            label = labels_list[preds["labels"][i]]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {score:.2f}", (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            objects.append(label)
    return img, objects

def detect_haar(img, model):
    objects = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        objects.append("face")
    return img, objects

def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    detected_objects = None
    result_video = None
    video_path = None
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        model_name = request.form.get("model")
        if not uploaded_file or not model_name:
            return "File and model required", 400

        model = load_model(model_name)
        if not model:
            return f"Failed to load model {model_name}", 500

        filename = uploaded_file.filename.lower()

        # --- Image ---
        if filename.endswith((".jpg", ".jpeg", ".png")):
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if model_name == "YOLO":
                output, objects = detect_yolo(img, model)
            elif model_name in ["SSD", "Faster R-CNN"]:
                output, objects = detect_torchvision(img, model)
            elif model_name == "Haar Cascade":
                output, objects = detect_haar(img, model)
            result_image = encode_image_to_base64(output)
            detected_objects = Counter(objects)

        # --- Video ---
        elif filename.endswith((".mp4", ".avi", ".mov")):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out = cv2.VideoWriter(out_temp.name, fourcc, fps, (width, height))
            all_objects = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if model_name == "YOLO":
                    output, objects = detect_yolo(frame, model)
                elif model_name in ["SSD", "Faster R-CNN"]:
                    output, objects = detect_torchvision(frame, model)
                elif model_name == "Haar Cascade":
                    output, objects = detect_haar(frame, model)
                all_objects.extend(objects)
                out.write(output)
            cap.release()
            out.release()
            result_video = out_temp.name
            detected_objects = Counter(all_objects)

    return render_template("index.html",
                           result_image=result_image,
                           detected_objects=detected_objects,
                           result_video=result_video)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)