import streamlit as st
import cv2
import numpy as np
import os
from mtcnn.mtcnn import MTCNN
from utils import load_facenet_model, get_embedding

# Initialize model and face detector
MODEL_PATH = "facenet_model/20180402-114759.pb"

st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("🎥 Real-time Face Attendance System")

try:
    graph = load_facenet_model(MODEL_PATH)
    detector = MTCNN()
    st.success("✅ FaceNet model and MTCNN loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Load registered face embeddings from the known_faces folder
@st.cache_resource
def load_known_faces():
    known_embeddings = []
    known_names = []
    for file in os.listdir("known_faces"):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join("known_faces", file))
            results = detector.detect_faces(img)
            if results:
                x, y, w, h = results[0]['box']
                face = img[y:y+h, x:x+w]
                emb = get_embedding(face, graph)
                known_embeddings.append(emb)
                known_names.append(file.split('.')[0])
    return known_embeddings, known_names

known_embeddings, known_names = load_known_faces()

# Start webcam stream
stframe = st.empty()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Failed to access the webcam.")
    st.stop()

st.info("📸 Please look directly at the camera. Your identity will be recognized in real-time.")

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Failed to read from webcam.")
        break

    results = detector.detect_faces(frame)
    if results:
        x, y, w, h = results[0]['box']
        face_crop = frame[y:y+h, x:x+w]
        emb = get_embedding(face_crop, graph)
        distances = [np.linalg.norm(emb - e) for e in known_embeddings]
        min_idx = np.argmin(distances)

        if distances[min_idx] < 0.7:
            name = known_names[min_idx]
            text = f"✅ {name} ({distances[min_idx]:.4f})"
            color = (0, 255, 0)
        else:
            text = "❌ Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    stframe.image(frame, channels="BGR")

cap.release()
