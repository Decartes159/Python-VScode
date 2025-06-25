import streamlit as st
import numpy as np
import cv2
import joblib
from utils import load_facenet_model, get_embedding
from PIL import Image

# Configure page settings
st.set_page_config(page_title="Real-Time Face Recognition", layout="wide")
st.title("📸 Real-Time Face Recognition")

# Load models with caching to avoid reloading
@st.cache_resource
def load_models():
    try:
        graph = load_facenet_model("facenet_model/20180402-114759.pb")
        classifier = joblib.load("trained_model/classifier.joblib")
        encoder = joblib.load("trained_model/label_encoder.joblib")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return graph, classifier, encoder, face_cascade
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None, None

graph, classifier, encoder, face_cascade = load_models()

# Only proceed if models loaded successfully
if graph and classifier and encoder and face_cascade:
    # Create two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("1. Capture Image")
        img_file_buffer = st.camera_input("Look at the camera")
        
    with col2:
        st.header("2. Recognition Results")
        
        if img_file_buffer is not None:
            # Convert image to OpenCV format
            image = Image.open(img_file_buffer)
            frame = np.array(image)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Extract face region
                    face = frame[y:y+h, x:x+w]
                    face = cv2.resize(face, (160, 160))
                    
                    # Get embedding and make prediction
                    with st.spinner("Recognizing..."):
                        emb = get_embedding(face, graph)
                        prediction = classifier.predict([emb])
                        prob = classifier.predict_proba([emb]).max()
                        name = encoder.inverse_transform(prediction)[0]
                    
                    # Display results
                    st.image(face, caption="Detected Face", width=200)
                    
                    if prob > 0.7:
                        st.success(f"✅ Recognized: {name} (Confidence: {prob:.2%})")
                    else:
                        st.warning(f"⚠️ Low confidence recognition: {name} (Confidence: {prob:.2%})")
            else:
                st.error("No face detected, please try again")
                
            # Display original image with detection box
            st.image(frame, caption="Detection Result", use_column_width=True)
else:
    st.error("Failed to load required models, please check model file paths")