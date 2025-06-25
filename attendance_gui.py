import streamlit as st
import pandas as pd
import os
import datetime
from geopy.distance import geodesic
import cv2
import numpy as np
from utils import load_facenet_model, get_embedding
from PIL import Image
import joblib
from streamlit_geolocation import streamlit_geolocation
CREDENTIALS = {
    "Josiah": {"password": "Josiah1", "role": "Student"},
    "Bethany": {"password": "Bethany1", "role": "Student"},
    "Huiming": {"password": "Huiming1", "role": "Student"},
    "Hongshen": {"password": "Hongshen1", "role": "Student"},
    "Yuxuan": {"password": "Yuxuan1", "role": "Student"},
    "lecturer1": {"password": "lecpass1", "role": "Lecturer"},
    "lecturer2": {"password": "lecpass2", "role": "Lecturer"},
}

@st.cache_resource
def load_models():
    """Loads all necessary models for face recognition."""
    try:
        # NOTE: Ensure these paths are correct in your project structure
        model_path = "facenet_model/20180402-114759.pb"
        classifier_path = "trained_model/classifier.joblib"
        encoder_path = "trained_model/label_encoder.joblib"
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        graph = load_facenet_model(model_path)
        classifier = joblib.load(classifier_path)
        encoder = joblib.load(encoder_path)
        face_cascade = cv2.CascadeClassifier(cascade_path)
        return graph, classifier, encoder, face_cascade
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

if "page" not in st.session_state:
    st.session_state.page = "login"

# sidebar info
if st.session_state.page != "login":
    with st.sidebar:
        st.markdown("## Account Info")
        st.markdown(f"**Username:** `{st.session_state.get('username', 'N/A')}`")
        st.markdown(f"**Role:** `{st.session_state.get('role', 'N/A')}`")
        st.divider()
        if st.button("Log Out"):
            st.session_state.page = "login"
            st.rerun()

# Login Page 
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

# Student Page
def student_page():
    username = st.session_state.get("username", "Student")
    st.title(f"Welcome, {username}!")
    st.divider()

    if st.session_state.get("attendance_submitted"):
        st.success("Attendance has been submitted successfully.")
        st.session_state.attendance_submitted = False

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

# Lecturer Page
def lecturer_page():
    username = st.session_state.get("username", "Lecturer")
    st.title(f"Welcome, {username}!")
    st.divider()

    lecturer_classes = ["Select a class", "BSC124 - Algebra", "BSC121 - Calculus", "AIT102 - Python"]
    selected_class = st.selectbox("View Attendance for:", lecturer_classes)

    # 
    if selected_class:
        class_code = selected_class.split()[0]
        file_name = f"data/attendance_{class_code}.csv"

        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            st.subheader(f"Attendance for {selected_class}")
            st.dataframe(df)

            st.download_button(
                label = "Download CSV",
                data = df.to_csv(index = False),
                file_name = file_name,
                mime = 'text/csv'
            )

            st.info(f"{len(df)} records found.")
        else:
            st.warning("No attendance records for this class yet.")

# helper function to calculate distance
def calculate_distance(user_loc, target_loc):
    """Calculates distance in meters between two lat/lon points."""
    if user_loc and target_loc:
        return geodesic(user_loc, target_loc).meters
    return float('inf') # Return a very large number if a location is missing

# save to csv file
def save_attendance(username, selected_class):
    class_code = selected_class.split()[0]
    file_name = f"data/attendance_{class_code}.csv"

    os.makedirs("data", exist_ok=True)

    new_entry = {
        "Name": username,
        "Date": datetime.date.today().strftime("%Y-%m-%d"),
        "Time": datetime.datetime.now().strftime("%H:%M:%S")
    }

    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])

    df.to_csv(file_name, index=False)

# Verification Page (students)
def verify_page():
    st.title("Attendance Process")
    st.divider()

    # --- Initialize session state variables ---
    if 'face_verified' not in st.session_state:
        st.session_state['face_verified'] = False # You will need to set this to True from your face_verification_page
    if 'location_verified' not in st.session_state:
        st.session_state['location_verified'] = False

    selected_class = st.session_state.get("selected_class", "N/A")
    st.markdown(f"Class: **{selected_class}**")

    # --- Step 1: Face Verification ---
    st.subheader("Step 1: Face Verification")
    if st.button("Begin Face Verification"):
        st.session_state.page = "face_verification"
        st.rerun()

    # --- Step 2: Location Verification ---
    st.subheader("Step 2: Location Verification")
    st.warning(
        "Please ensure you are on campus and have enabled location services in your browser."
    )
    location = st.session_state.get('location', None)
    if st.button("Verify My Location"):
        location = streamlit_geolocation() # This widget will ask for permission
        st.write("Location data received:", location)

    if location and 'latitude' in location and 'longitude' in location:
        st.write(f"Your coordinates are {location['latitude']}, {location['longitude']}")
        user_location = (location['latitude'], location['longitude'])
        TARGET_LOCATION = (2.830973,101.703846) # XMUM Campus A3
        ALLOWED_DISTANCE_METERS = 500 # Can be much smaller now!
        distance = calculate_distance(user_location, TARGET_LOCATION)
        if distance <= ALLOWED_DISTANCE_METERS:
            st.success("Location Verified! You are on campus.")
            st.session_state['location_verified'] = True
        else:
            st.error(f"Location Check Failed. You are {distance:.2f} meters away from campus.")
    else:
        st.error("Location verification failed. Please ensure you have allowed location access in your browser settings.")
##########
    st.divider()
    st.markdown("### Verification Checklist")

    st.checkbox("Face Verified", value=st.session_state.get('face_verified', False), disabled=True)
    st.checkbox("Location Verified", value=st.session_state.get('location_verified', False), disabled=True)

    if st.session_state.get('face_verified') and st.session_state.get('location_verified'):
        if st.button("Submit Attendance"):
            save_attendance(
                username=st.session_state.get("username", "Unknown"),
                selected_class=st.session_state.get("selected_class", "Unknown"),
            )
            st.session_state.attendance_submitted = True
            st.session_state.face_verified = False
            st.session_state.location_verified = False
            st.session_state.page = "student"
            st.rerun()
    else:
        st.warning("Please complete all verification steps.")

    if st.button("Back to Class Selection"):
        st.session_state.page = "student"
        st.rerun()

def face_verification_page():
    st.title("Face Verification")
    st.markdown("Please capture your face for verification.")
    st.divider()

    # --- Load Models ---
    graph, classifier, encoder, face_cascade = load_models()

    if not all([graph, classifier, encoder, face_cascade]):
        st.error("Face recognition system is currently unavailable. Please contact an administrator.")
        if st.button("Return to Verification"):
            st.session_state.page = "verify"
            st.rerun()
        return # Stop execution if models aren't loaded

    # --- Capture Image ---
    st.info("Please look directly at the camera and capture your image.")
    img_file_buffer = st.camera_input("Capture your face for verification")
    
    logged_in_user = st.session_state.get("username")

    if img_file_buffer is not None:
        # Convert image buffer to OpenCV format
        image = Image.open(img_file_buffer)
        frame = np.array(image)
        # Convert RGB (from PIL) to BGR (for OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            st.error("No face detected. Please try again, ensuring good lighting and a clear view.")
        elif len(faces) > 1:
            st.warning("Multiple faces detected. Please ensure only your face is visible.")
        else: # Exactly one face was detected
            (x, y, w, h) = faces[0]
            
            # Extract face region, resize for the model, and get embedding
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (160, 160))
            
            with st.spinner("Verifying..."):
                embedding = get_embedding(face_resized, graph)
                prediction = classifier.predict([embedding])
                probability = classifier.predict_proba([embedding]).max()
                recognized_name = encoder.inverse_transform(prediction)[0]

            # For visualization, draw rectangle on the original frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Convert back to RGB for Streamlit display
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Verification Image")

            # --- THE CRITICAL VERIFICATION CHECK ---
            CONFIDENCE_THRESHOLD = 0.80 # 80% confidence required

            st.write(f"Detected: `{recognized_name}` with {probability:.2%} confidence.")
            st.write(f"Logged In As: `{logged_in_user}`")

            if recognized_name == logged_in_user and probability > CONFIDENCE_THRESHOLD:
                st.success(f"✅ Welcome, {recognized_name}! Your face has been verified.")
                st.session_state['face_verified'] = True
                st.balloons()
            elif recognized_name != logged_in_user:
                 st.error(f"❌ Verification Failed. Face recognized as `{recognized_name}`, but you are logged in as `{logged_in_user}`.")
                 st.session_state['face_verified'] = False
            else: # Correct person, but low confidence
                 st.warning(f"⚠️ Low confidence ({probability:.2%}). Please try again with better lighting and a clearer view of your face.")
                 st.session_state['face_verified'] = False

    st.divider()
    if st.button("Return to Verification Steps"):
        st.session_state.page = "verify"
        st.rerun()


# Switching Pages
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "student":
    student_page()
elif st.session_state.page == "lecturer":
    lecturer_page()
elif st.session_state.page == "verify":
    verify_page()
elif st.session_state.page == "face_verification":
    face_verification_page()