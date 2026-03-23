import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from collections import deque

# ============================
# CONFIG
# ============================
IMG_SIZE = 48
EMOTIONS = ["Angry", "Happy", "Sad"]
FRAME_DELAY = 0.04  # ~25 FPS

PROTO = "deploy.prototxt"
MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# ============================
# STREAMLIT SETUP
# ============================
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# ============================
# LOAD TFLITE MODEL
# ============================
@st.cache_resource
def load_tflite():
    interpreter = tf.lite.Interpreter(model_path="emotion_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ============================
# LOAD FACE DETECTOR (OpenCV DNN)
# ============================
@st.cache_resource
def load_face_net():
    return cv2.dnn.readNetFromCaffe(PROTO, MODEL)

face_net = load_face_net()

# ============================
# SIDEBAR
# ============================
st.sidebar.title("⚙️ Controls")
run = st.sidebar.toggle("Start Webcam", False)
show_history = st.sidebar.checkbox("Show Emotion History", True)

# ============================
# UI
# ============================
st.markdown(
    "<h1 style='text-align:center;'>😊 Human Emotion Recognition</h1>",
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])
frame_box = col1.image([])

# ============================
# SESSION STATE
# ============================
if "camera" not in st.session_state:
    st.session_state.camera = None

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=10)

# ============================
# MAIN LOOP
# ============================
if run:
    if st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    ret, frame = st.session_state.camera.read()
    if not ret:
        col1.error("❌ Camera not accessible")
    else:
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        face_net.setInput(blob)
        detections = face_net.forward()

        emotion = "No face"
        confidence = 0.0

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                face = face / 255.0
                face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

                interpreter.set_tensor(input_details[0]["index"], face)
                interpreter.invoke()
                preds = interpreter.get_tensor(output_details[0]["index"])

                idx = np.argmax(preds)
                emotion = EMOTIONS[idx]
                confidence = float(np.max(preds))

                st.session_state.history.appendleft(emotion)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    emotion,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
                break

        frame_box.image(frame, channels="BGR")

        col2.metric("Emotion", emotion)
        col2.progress(confidence)

        if show_history:
            col2.markdown("### 🕒 Recent Emotions")
            for e in st.session_state.history:
                col2.markdown(f"- {e}")

        time.sleep(FRAME_DELAY)

else:
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    col1.info("📷 Webcam OFF")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:12px;'>Runs locally. No data stored.</p>",
    unsafe_allow_html=True
)
