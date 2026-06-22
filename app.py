import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
import torchvision
from torchvision import transforms
from ultralytics import YOLO
from collections import Counter
import time
import os
import base64

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AI Object Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# CSS Styling
# ---------------------------
def load_css():
    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------------------
# COCO Labels
# ---------------------------
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

# ---------------------------
# Model Loading (Cached)
# ---------------------------
@st.cache_resource
def load_model(model_name, yolo_size="n", custom_path=None):
    """Loads a detection model with caching to avoid reloading."""
    try:
        if model_name == "YOLOv8":
            if custom_path and os.path.exists(custom_path):
                return YOLO(custom_path)
            return YOLO(f"yolov8{yolo_size}.pt")
        elif model_name == "SSD":
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
            model.eval()
            return model
        elif model_name == "Faster R-CNN":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            model.eval()
            return model
        elif model_name == "Haar Cascade (Face)":
            return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception as e:
        st.error(f"❌ Failed to load {model_name}: {e}")
        return None

# ---------------------------
# Detection Functions
# ---------------------------
def process_frame(frame, model, model_name, conf_thresh, use_tracking=False):
    """Processes a single frame — returns annotated frame, detected objects, scores, tracked_items."""
    output_frame = frame.copy()

    if model_name == "YOLOv8" and use_tracking:
        results = model.track(output_frame, verbose=False, conf=conf_thresh, persist=True)
        tracked_items = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = results[0].names
            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = box
                label = class_names[cls_id]
                tracked_items.append((label, track_id))
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
                cv2.putText(output_frame, f"ID {track_id}: {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)
        current_objects = [item[0] for item in tracked_items]
        return output_frame, current_objects, [], tracked_items

    elif model_name == "YOLOv8":
        results = model(output_frame, verbose=False, conf=conf_thresh)
        objects, scores = [], []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id, score = int(box.cls[0]), float(box.conf[0])
                label = r.names[cls_id]
                objects.append(label)
                scores.append(score)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
                cv2.putText(output_frame, f"{label} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)
        return output_frame, objects, scores, []

    elif model_name in ["SSD", "Faster R-CNN"]:
        labels_list = get_coco_labels()
        img_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        tensor_img = transforms.ToTensor()(img_rgb)
        with torch.no_grad():
            preds = model([tensor_img])[0]
        objects, scores = [], []
        for i, score_val in enumerate(preds["scores"]):
            if score_val > conf_thresh:
                box = preds["boxes"][i].cpu().numpy().astype(int)
                label = labels_list[preds["labels"][i]]
                objects.append(label)
                scores.append(float(score_val))
                cv2.rectangle(output_frame, (box[0], box[1]), (box[2], box[3]), (0, 200, 255), 2)
                cv2.putText(output_frame, f"{label} {score_val:.2f}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        return output_frame, objects, scores, []

    elif model_name == "Haar Cascade (Face)":
        gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
        faces = model.detectMultiScale(gray, 1.1, 4)
        objects = []
        for (x, y, w, h) in faces:
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            cv2.putText(output_frame, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            objects.append("face")
        return output_frame, objects, [], []

    return output_frame, [], [], []

# ---------------------------
# Sidebar
# ---------------------------
def display_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")
    st.sidebar.markdown("---")

    model_name = st.sidebar.selectbox(
        "🤖 Detection Model",
        ["YOLOv8", "SSD", "Faster R-CNN", "Haar Cascade (Face)"],
        help="Choose the object detection model"
    )

    conf_thresh = st.sidebar.slider(
        "🎯 Confidence Threshold",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Only show detections above this confidence score"
    )

    frame_skip = st.sidebar.slider(
        "⏭️ Frame Skip (Video/Webcam)",
        min_value=1, max_value=10, value=1,
        help="Process every Nth frame to improve speed"
    )

    custom_model_path, use_tracking, yolo_size = None, False, "n"

    if model_name == "YOLOv8":
        st.sidebar.markdown("---")
        use_tracking = st.sidebar.checkbox("🔍 Enable Object Tracking", value=True,
            help="Assigns unique IDs to objects across video frames")
        use_custom = st.sidebar.checkbox("📂 Use Custom YOLO Model")
        if use_custom:
            custom_model_file = st.sidebar.file_uploader("Upload YOLO Weights (.pt)", type=["pt"])
            if custom_model_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_pt:
                    temp_pt.write(custom_model_file.read())
                    custom_model_path = temp_pt.name
        else:
            size_option = st.sidebar.selectbox(
                "📏 YOLOv8 Model Size",
                ["n — Nano (Fastest)", "s — Small", "m — Medium", "l — Large", "x — Extra Large (Best Accuracy)"]
            )
            yolo_size = size_option[0]

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Lower confidence threshold = more detections. Use Frame Skip for faster video processing.")

    return model_name, conf_thresh, frame_skip, yolo_size, custom_model_path, use_tracking

# ---------------------------
# Image Tab
# ---------------------------
def render_image_tab(model, model_name, conf_thresh):
    st.markdown("### 📷 Upload an Image for Detection")

    uploaded_file = st.file_uploader(
        "Drag & drop or browse", type=["jpg", "jpeg", "png"], key="image_uploader",
        help="Supported: JPG, JPEG, PNG"
    )

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_orig, col_result = st.columns(2)
        with col_orig:
            st.markdown("**📁 Original Image**")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("🔍 Detecting objects..."):
            start = time.time()
            output_frame, objects, scores, _ = process_frame(img, model, model_name, conf_thresh, use_tracking=False)
            elapsed = time.time() - start

        with col_result:
            st.markdown("**✅ Detection Result**")
            st.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("🏷️ Objects Detected", len(objects))
        m2.metric("⏱️ Inference Time", f"{elapsed:.2f}s")
        m3.metric("📊 Avg Confidence", f"{np.mean(scores):.2f}" if scores else "N/A")

        if objects:
            st.markdown("#### 📊 Detection Summary")
            counts = Counter(objects)
            cols = st.columns(min(len(counts), 5))
            for i, (name, count) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
                cols[i % len(cols)].metric(label=name.title(), value=count)

        _, buf = cv2.imencode(".jpg", output_frame)
        st.download_button(
            "💾 Download Result Image", buf.tobytes(),
            file_name="detection_result.jpg", mime="image/jpeg"
        )

# ---------------------------
# Video Tab
# ---------------------------
def render_video_tab(model, model_name, conf_thresh, frame_skip, use_tracking):
    st.markdown("### 🎞️ Upload a Video for Detection")

    uploaded_file = st.file_uploader(
        "Drag & drop or browse", type=["mp4", "avi", "mov"], key="video_uploader",
        help="Supported: MP4, AVI, MOV"
    )

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        stframe = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_counts = Counter()
        counted_track_ids = set()
        prev_time = time.time()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                output_frame, current_objs, _, tracked_items = process_frame(
                    frame, model, model_name, conf_thresh, use_tracking
                )

                if use_tracking and model_name == "YOLOv8":
                    for label, track_id in tracked_items:
                        if track_id not in counted_track_ids:
                            total_counts[label] += 1
                            counted_track_ids.add(track_id)
                else:
                    total_counts.update(current_objs)

                curr_time = time.time()
                fps_live = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                cv2.putText(output_frame, f"FPS: {fps_live:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                progress = min(frame_count / max(total_frames, 1), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"⏳ Processing frame {frame_count} / {total_frames}")
                stframe.image(output_frame, channels="BGR", use_container_width=True)
                out.write(output_frame)

            frame_count += 1

        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Video Processing Complete!")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("💾 Download Processed Video", f.read(),
                                   file_name="detection_output.mp4", mime="video/mp4")
        with col2:
            st.markdown("#### 📊 Total Objects Detected")
            if total_counts:
                for name, count in sorted(total_counts.items(), key=lambda x: -x[1]):
                    st.metric(label=name.title(), value=count)
            else:
                st.warning("No objects detected in this video.")

# ---------------------------
# Webcam Tab
# ---------------------------
def render_webcam_tab(model, model_name, conf_thresh, frame_skip, use_tracking):
    st.markdown("### 📹 Live Webcam Detection")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("▶️ Start Webcam", disabled=st.session_state.webcam_running, use_container_width=True):
            st.session_state.webcam_running = True
            st.session_state.total_object_counts = Counter()
            st.session_state.counted_track_ids = set()
            st.rerun()
    with col_btn2:
        if st.button("⏹️ Stop Webcam", disabled=not st.session_state.webcam_running, use_container_width=True):
            st.session_state.webcam_running = False
            st.rerun()

    if st.session_state.webcam_running:
        col1, col2 = st.columns([3, 1])
        with col1:
            frame_window = st.image([])
        with col2:
            st.markdown("#### 📊 Live Counts")
            counts_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check your camera.")
            st.session_state.webcam_running = False
            return

        prev_time = time.time()
        frame_count = 0

        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture from webcam.")
                st.session_state.webcam_running = False
                break

            if frame_count % frame_skip == 0:
                output_frame, current_objs, _, tracked_items = process_frame(
                    frame, model, model_name, conf_thresh, use_tracking
                )

                if use_tracking and model_name == "YOLOv8":
                    for label, track_id in tracked_items:
                        if track_id not in st.session_state.counted_track_ids:
                            st.session_state.total_object_counts[label] += 1
                            st.session_state.counted_track_ids.add(track_id)
                else:
                    st.session_state.total_object_counts.update(current_objs)

                curr_time = time.time()
                fps_live = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                cv2.putText(output_frame, f"FPS: {fps_live:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                frame_window.image(output_frame, channels="BGR", use_container_width=True)

                with counts_placeholder.container():
                    for name, count in sorted(st.session_state.total_object_counts.items()):
                        st.metric(label=name.title(), value=count)

            frame_count += 1

        cap.release()
        st.info("📷 Webcam stopped.")

# ---------------------------
# Main Application
# ---------------------------
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Object Detection Dashboard</h1>
        <p>Real-time AI-powered object detection with YOLOv8, SSD, Faster R-CNN & Haar Cascade</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    model_name, conf_thresh, frame_skip, yolo_size, custom_model_path, use_tracking = display_sidebar()

    # Load Model
    with st.spinner(f"🔄 Loading {model_name} model..."):
        model = load_model(model_name, yolo_size, custom_model_path)

    if model is None:
        st.error("❌ Model failed to load. Please try a different model.")
        st.stop()

    st.sidebar.success(f"✅ {model_name} model ready!")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📷  Image Detection", "🎞️  Video Detection", "📹  Live Webcam"])

    with tab1:
        render_image_tab(model, model_name, conf_thresh)

    with tab2:
        render_video_tab(model, model_name, conf_thresh, frame_skip, use_tracking)

    with tab3:
        render_webcam_tab(model, model_name, conf_thresh, frame_skip, use_tracking)

if __name__ == "__main__":
    main()
