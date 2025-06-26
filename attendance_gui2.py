import streamlit as st
import pandas as pd
import os
import datetime
import random
import string
from geopy.distance import geodesic
import cv2
import numpy as np
from utils import load_facenet_model, get_embedding
from PIL import Image
import joblib
import importlib
import shutil
from streamlit_geolocation import streamlit_geolocation


if "page" not in st.session_state:
    st.session_state.page = "login"
if "preview_data" not in st.session_state:
    st.session_state.preview_data = []
if "attendance_submitted" not in st.session_state:
    st.session_state.attendance_submitted = False

def load_credentials():
    try:
        import credentials
        importlib.reload(credentials)
        return credentials.CREDENTIALS
    except (ImportError, SyntaxError) as e:
        st.error(f"Error loading credentials file: {e}. Please ensure 'credentials.py exists and is correct.")
        return{}
    
def save_credentials(updated_credentials):
    try:
        with open("credentials.py", "w", encoding="utf-8") as f:
            f.write("CREDENTIALS = {\n")
            for user, details in updated_credentials.items():
                f.write(f"    \"{user}\": {details},\n")
            f.write("}\n")
        return True
    except Exception as e:
        st.error(f"Failed to save credentials: {e}")
        return False
    

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

# sidebar info
if st.session_state.page != "login":
    with st.sidebar:
        st.markdown("## Account Info")
        st.markdown(f"**Username:** `{st.session_state.get('username', 'N/A')}`")
        st.markdown(f"**Role:** `{st.session_state.get('role', 'N/A')}`")
        st.divider()
        if st.button("Log Out"):

            # clear session
            for key in list(st.session_state.keys()):
                if key not in ['page']:
                    del st.session_state[key]

            st.session_state.page = "login"
            st.rerun()

# Login Page 
def login_page():
    st.set_page_config(page_title="Attendance Recorder")
    st.title("Attendance Recorder")
    st.divider()
    st.subheader("Welcome!")

    CREDENTIALS = load_credentials()
    if not CREDENTIALS:
        return

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

    option = st.selectbox("Please select an operation:", ["Select an option", "View Class Attendance", "Add New Student", "Manage Students"])

    if option == "View Class Attendance":
        st.subheader("View Attendance by Class")
        lecturer_classes = ["Select a class", "BSC124 - Algebra", "BSC121 - Calculus", "AIT102 - Python"]
        selected_class = st.selectbox("View Attendance for:", lecturer_classes)

        if selected_class != "Select a class":
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

    elif option == "Add New Student":
        st.session_state.page = "add_student"
        st.rerun()

    elif option == "Manage Students":
        st.session_state.page = "manage_students"
        st.rerun()

def add_student_page():
    temp_data = []
    st.title("Add New Student")
    st.info("Upload one or more photos of the new student(s). Then enter their names.")

    uploaded_photos = st.file_uploader("Upload Student Photo(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_photos:
        with st.form("student_form"):
            for i, file in enumerate(uploaded_photos):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(file, width=100)
                with col2:
                    name = st.text_input(f"Student Name", key=f"name_{i}", placeholder="Enter unique username")
                temp_data.append({"file": file, "name": name})

            submitted = st.form_submit_button("Save Student(s)")
            if submitted:
                all_ok = True
                st.session_state.preview_data.clear()
                CREDENTIALS = load_credentials()

                for entry in temp_data:
                    name = entry["name"]
                    file = entry["file"]
                    if not name:
                        st.warning(f"A name is required for the file: {file.name}")
                        all_ok = False
                    if name in CREDENTIALS:
                        st.warning(f"Username '{name}' already exists. Please choose a different name.")
                        all_ok = False
                        break

                if all_ok:
                    for entry in temp_data:
                        password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                        st.session_state.preview_data.append({
                            "PhotoName": entry["file"].name,
                            "Name": entry["name"],
                            "Password": f"{entry['name']}1",
                            "File": entry["file"]
                        })
                    st.session_state.page = "confirm_student"
                    st.rerun()

    if st.button("Back to Lecturer Menu"):
        st.session_state.page = "lecturer"
        st.rerun()

def confirm_student_page():
    st.title("Confirm New Student Data")

    if not st.session_state.preview_data:
        st.warning("No student data to confirm. Please go back and add students.")
        if st.button("Back to Add Student"):
            st.session_state.page = "add_student"
            st.rerun()
        return
    
    st.subheader("Please review the data before saving:")
    for s in st.session_state.preview_data:
        st.image(s["File"], width=100)
        st.markdown(f"**Name/Username:** `{s['Name']}`")
        st.markdown(f"**Generated Password:** `{s['Password']}`")
        st.divider()

    if st.button("Confirm and Save All"):
        CREDENTIALS = load_credentials()
        photo_dir = "student_photos"
        os.makedirs(photo_dir, exist_ok=True)

        for s in st.session_state.preview_data:
            photo_path = os.path.join(photo_dir, s["Name"] + os.path.splitext(s["PhotoName"])[1])
            with open(photo_path, "wb") as f:
                f.write(s["File"].getbuffer())

            CREDENTIALS[s["Name"]] = {"password": s["Password"], "role": "Student"}

        if save_credentials(CREDENTIALS):
            st.success("All new students have been saved successfully!")
            st.session_state.preview_data.clear()
            st.session_state.page = "lecturer"
            st.balloons()
            st.rerun()
        else:
            st.error("An error occurred while saving. Please try again.")

    if st.button("Back to Edit"):
        st.session_state.page = "add_student"
        st.rerun()

def manage_students_page():
    st.title("Manage Students")
    st.divider()

    CREDENTIALS = load_credentials()
    students = {user: details for user, details in CREDENTIALS.items() if details["role"] == "Student"}

    if not students:
        st.warning("No students found.")
    else:
        st.info(f"Found {len(students)} student record(s).")
        for username in sorted(students.keys()):
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Username:** `{username}`")
                    st.markdown(f"**Password:** `{students[username]['password']}`")
                with col2:
                    if st.button("🗑️ Delete", key=f"del_{username}"):
                        del CREDENTIALS[username]
                        photo_dir = "student_photos"
                        for ext in ['.jpg', '.jpeg', '.png']:
                            photo_path = os.path.join(photo_dir, username + ext)
                            if os.path.exists(photo_path):
                                os.remove(photo_path)
                                
                        if save_credentials(CREDENTIALS):
                            st.success(f"Successfully deleted {username}.")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {username}.")
    
    if st.button("Back to Lecturer Menu"):
        st.session_state.page = "lecturer"
        st.rerun()

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
# Verification Page (students) - CORRECTED FOR KEY ERROR
def verify_page():
    st.title("Attendance Process")
    st.divider()

    # --- Initialize session state variables ---
    if 'face_verified' not in st.session_state:
        st.session_state['face_verified'] = False
    if 'location_verified' not in st.session_state:
        st.session_state['location_verified'] = False
    # This new state will control when we show the location widget
    if 'checking_location' not in st.session_state:
        st.session_state['checking_location'] = False

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
    
    # This button will now just set the state, not call the widget
    if st.button("Get My Location"):  
        st.session_state['checking_location'] = True

    # --- NEW LOGIC: Only render the widget if we are in the 'checking_location' state ---
    if st.session_state.get('checking_location', False):
        st.info("Getting location data... Please approve the request in your browser.")
        location_data = streamlit_geolocation() # The widget is now safely isolated

        if location_data and 'latitude' in location_data:
            user_location = (location_data['latitude'], location_data['longitude'])
            TARGET_LOCATION = (2.830973, 101.703846) # XMUM Campus A3
            ALLOWED_DISTANCE_METERS = 1000

            distance = calculate_distance(user_location, TARGET_LOCATION)
            
            if distance <= ALLOWED_DISTANCE_METERS:
                st.success(f"Location Verified! You are on campus ({distance:.2f} meters away).")
                st.session_state['location_verified'] = True
            else:
                st.error(f"Location Check Failed. You are {distance:.2f} meters away from the target location.")
                st.session_state['location_verified'] = False
            
            # IMPORTANT: Reset the state so the widget disappears after we are done with it
            st.session_state['checking_location'] = False
            st.rerun() # Rerun to update the checklist display immediately
        else:
            st.warning("Could not retrieve location. Please make sure you have granted permission.")


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
elif st.session_state.page == "add_student":
    add_student_page()
elif st.session_state.page == "confirm_student":
    confirm_student_page()
elif st.session_state.page == "manage_students":
    manage_students_page()