import streamlit as st
import pandas as pd
import os
import datetime
from geopy.distance import geodesic
import geocoder
import numpy as np
import cv2
import joblib
from PIL import Image
import threading
import av  # The PyAV library is a dependency for streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# You will need a utils.py file for the next line, or place the functions directly in this script
# from utils import load_facenet_model, get_embedding 

# --- FAKE HELPER FUNCTIONS (for demonstration purposes) ---
# In your actual project, REMOVE these and use your 'from utils import ...' line
@st.cache_resource
def load_facenet_model(path):
    st.info("INFO: Using placeholder `load_facenet_model`. In production, this loads the real model.")
    class DummyGraph:
        def get_tensor_by_name(self, name):
            return None
    return DummyGraph()

def get_embedding(face_pixels, graph):
    st.info("INFO: Using placeholder `get_embedding`. In production, this generates real embeddings.")
    return np.random.rand(1, 128)
# --- END OF FAKE HELPER FUNCTIONS ---


# --- App Configuration & State ---
CREDENTIALS = {
    "student1": {"password": "studpass1", "role": "Student"},
    "student2": {"password": "studpass2", "role": "Student"},
    "lecturer1": {"password": "lecpass1", "role": "Lecturer"},
    "lecturer2": {"password": "lecpass2", "role": "Lecturer"},
}

if "page" not in st.session_state:
    st.session_state.page = "login"

# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_models():
    """Loads all necessary models for face recognition."""
    try:
        model_path = "facenet_model/20180402-114759.pb"
        classifier_path = "trained_model/classifier.joblib"
        encoder_path = "trained_model/label_encoder.joblib"
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # This check is commented out to allow the app to run with placeholder models.
        # Uncomment it in your real project.
        # if not all(os.path.exists(p) for p in [model_path, classifier_path, encoder_path]):
        #      st.error("Model files not found! Please ensure 'facenet_model' and 'trained_model' directories are correct.")
        #      return None, None, None, None

        graph = load_facenet_model(model_path)
        classifier = joblib.load(classifier_path)
        encoder = joblib.load(encoder_path)
        face_cascade = cv2.CascadeClassifier(cascade_path)
        return graph, classifier, encoder, face_cascade
    except Exception as e:
        # For demonstration, we'll create dummy models if the files are not found.
        # In your real project, you should handle this error more gracefully.
        st.warning(f"Could not load real models (Error: {e}). Creating dummy models for demonstration.")
        class DummyClassifier:
            def predict(self, emb): return ["student1"]
            def predict_proba(self, emb): return np.array([[0.95]])
        class DummyEncoder:
            def inverse_transform(self, pred): return pred
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return DummyGraph(), DummyClassifier(), DummyEncoder(), face_cascade


# --- Sidebar ---
if st.session_state.page != "login":
    with st.sidebar:
        st.markdown("## Account Info")
        st.markdown(f"**Username:** `{st.session_state.get('username', 'N/A')}`")
        st.markdown(f"**Role:** `{st.session_state.get('role', 'N/A')}`")
        st.divider()
        if st.button("Log Out"):
            for key in list(st.session_state.keys()):
                if key not in ['page']:
                    del st.session_state[key]
            st.session_state.page = "login"
            st.rerun()


# --- Page Definitions (login_page, student_page, etc. are unchanged) ---
def login_page():
    st.set_page_config(page_title="Attendance Recorder")
    st.title("Attendance Recorder")
    st.divider()
    st.subheader("Welcome!")

    role = st.radio("I am a:", ["Student", "Lecturer"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = CREDENTIALS.get(username)
        if user and user["password"] == password and user["role"] == role:
            st.session_state.username = username
            st.session_state.role = role
            st.success(f"Logged in as {role}")

            if role == "Student":
                st.session_state.page = "student"
                st.rerun()
            elif role == "Lecturer":
                st.session_state.page = "lecturer"
                st.rerun()
        else:
            st.error("Incorrect username, password, or role.")

def student_page():
    username = st.session_state.get("username", "Student")
    st.title(f"Welcome, {username}!")
    st.divider()

    if st.session_state.get("attendance_submitted"):
        st.success("Attendance has been submitted successfully.")
        st.session_state.attendance_submitted = False # Reset flag

    classes = ["Select a class", "BSC124 - Algebra", "BSC121 - Calculus", "AIT102 - Python"]
    selected_class = st.selectbox("Select your class", classes)

    if selected_class != "Select a class":
        st.session_state.selected_class = selected_class
        st.info(f"Class selected: **{selected_class}**")

    if st.button("Start Attendance"):
        if selected_class != "Select a class":
            st.session_state.page = "verify"
            st.rerun()
        else:
            st.error("Please select a class.")

def lecturer_page():
    username = st.session_state.get("username", "Lecturer")
    st.title(f"Welcome, {username}!")
    st.divider()

    lecturer_classes = ["Select a class", "BSC124 - Algebra", "BSC121 - Calculus", "AIT102 - Python"]
    selected_class = st.selectbox("View Attendance for:", lecturer_classes)

    if selected_class != "Select a class":
        class_code = selected_class.split()[0]
        file_name = f"data/attendance_{class_code}.csv"

        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            st.subheader(f"Attendance for {selected_class}")
            st.dataframe(df)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"attendance_{class_code}.csv",
                mime='text/csv'
            )
            st.info(f"{len(df)} records found.")
        else:
            st.warning("No attendance records for this class yet.")

def calculate_distance(user_loc, target_loc):
    if user_loc and target_loc:
        return geodesic(user_loc, target_loc).meters
    return float('inf')

def save_attendance(username, selected_class):
    class_code = selected_class.split()[0]
    file_name = f"data/attendance_{class_code}.csv"
    os.makedirs("data", exist_ok=True)
    new_entry = {"Name": username, "Date": datetime.date.today().strftime("%Y-%m-%d"), "Time": datetime.datetime.now().strftime("%H:%M:%S")}
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
    df.to_csv(file_name, index=False)

def verify_page():
    st.title("Attendance Process")
    st.divider()
    st.session_state.setdefault('face_verified', False)
    st.session_state.setdefault('location_verified', False)

    st.markdown(f"**Class:** {st.session_state.get('selected_class', 'N/A')}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Step 1: Face Verification")
        if st.button("Begin Face Verification"):
            st.session_state.page = "face_verification"
            st.rerun()
    with col2:
        st.subheader("Step 2: Location Verification")
        if st.button("Verify My Location (IP)"):
            with st.spinner("Getting location based on IP..."):
                g = geocoder.ip('me')
                if g.ok:
                    user_location = g.latlng
                    TARGET_LOCATION = (2.8136, 101.7583) # Sepang
                    ALLOWED_DISTANCE_METERS = 20000  # 20km
                    distance = calculate_distance(user_location, TARGET_LOCATION)
                    if distance <= ALLOWED_DISTANCE_METERS:
                        st.session_state['location_verified'] = True
                        st.success(f"Location Verified within {ALLOWED_DISTANCE_METERS / 1000}km.")
                    else:
                        st.session_state['location_verified'] = False
                        st.error("Location Check Failed: Too far from target.")
                else:
                    st.error("Could not determine location from IP.")
    
    st.divider()
    st.markdown("### Verification Checklist")
    st.checkbox("Face Verified", value=st.session_state.get('face_verified', False), disabled=True)
    st.checkbox("Location Verified", value=st.session_state.get('location_verified', False), disabled=True)

    if st.session_state.get('face_verified') and st.session_state.get('location_verified'):
        if st.button("Submit Attendance", type="primary"):
            save_attendance(st.session_state.get("username"), st.session_state.get("selected_class"))
            st.session_state.attendance_submitted = True
            st.session_state.face_verified = False
            st.session_state.location_verified = False
            st.session_state.page = "student"
            st.rerun()
    
    if st.button("Back to Class Selection"):
        st.session_state.page = "student"
        st.rerun()


# ==============================================================================
#                      FACE VERIFICATION PAGE (WEBRTC IMPLEMENTATION)
# ==============================================================================
def face_verification_page():
    st.title("Face Verification")
    st.divider()

    # Load models
    graph, classifier, encoder, face_cascade = load_models()
    if not all([graph, classifier, encoder, face_cascade]):
        st.error("Face recognition system is currently unavailable.")
        if st.button("Return to Verification"):
            st.session_state.page = "verify"
            st.rerun()
        return

    # This class handles the video frames from the webcam.
    class FaceVerificationTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_lock = threading.Lock()
            self.latest_frame = None

        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            # Convert the frame to a NumPy array
            img = frame.to_ndarray(format="bgr24")

            # Store the latest frame for later processing
            with self.frame_lock:
                self.latest_frame = img.copy()
            
            # Draw a rectangle on the live feed for user feedback
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    st.info("Position your face in the center of the frame and click 'Verify Face'.")

    # RTCConfiguration is needed for deployment on Streamlit Cloud
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="face-verification",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=FaceVerificationTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if st.button("Verify Face", type="primary"):
        if ctx.video_processor:
            with ctx.video_processor.frame_lock:
                frame = ctx.video_processor.latest_frame
            
            if frame is None:
                st.warning("No frame captured. Please ensure your camera is active and try again.")
            else:
                with st.spinner("Analyzing image..."):
                    # Process the captured frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                    if len(faces) == 0:
                        st.error("No face detected. Please try again.")
                    elif len(faces) > 1:
                        st.warning("Multiple faces detected. Please ensure only you are in the frame.")
                    else:
                        (x, y, w, h) = faces[0]
                        face = frame[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (160, 160))
                        
                        embedding = get_embedding(face_resized, graph)
                        prediction = classifier.predict([embedding])
                        probability = classifier.predict_proba([embedding]).max()
                        recognized_name = encoder.inverse_transform(prediction)[0]
                        logged_in_user = st.session_state.get("username")

                        st.image(face, channels="BGR", caption="Captured Face for Verification")
                        CONFIDENCE_THRESHOLD = 0.80

                        st.write(f"Detected: `{recognized_name}` with {probability:.2%} confidence.")
                        st.write(f"Logged In As: `{logged_in_user}`")

                        if recognized_name == logged_in_user and probability > CONFIDENCE_THRESHOLD:
                            st.success(f"✅ Welcome, {recognized_name}! Your face has been verified.")
                            st.session_state.face_verified = True
                            st.balloons()
                        elif recognized_name != logged_in_user:
                            st.error(f"❌ Verification Failed. Face recognized as `{recognized_name}`, but you are logged in as `{logged_in_user}`.")
                            st.session_state.face_verified = False
                        else:
                            st.warning(f"⚠️ Low confidence ({probability:.2%}). Please try again with better lighting.")
                            st.session_state.face_verified = False
        else:
            st.error("Camera not ready. Please wait for the video stream to start.")

    st.divider()
    if st.button("Return to Verification Steps"):
        st.session_state.page = "verify"
        st.rerun()

# --- Main Page Router ---
if __name__ == "__main__":
    page_name = st.session_state.get("page", "login")
    page_map = {
        "login": login_page,
        "student": student_page,
        "lecturer": lecturer_page,
        "verify": verify_page,
        "face_verification": face_verification_page
    }
    page_function = page_map.get(page_name, login_page)
    page_function()