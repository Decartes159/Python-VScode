import streamlit as st
import pandas as pd
import os
import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import streamlit.components.v1 as components
import av
import geocoder 
from geopy.distance import geodesic

CREDENTIALS = {
    "student1": {"password": "studpass1", "role": "Student"},
    "student2": {"password": "studpass2", "role": "Student"},
    "lecturer1": {"password": "lecpass1", "role": "Lecturer"},
    "lecturer2": {"password": "lecpass2", "role": "Lecturer"},
}

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
        "**Disclaimer:** This method uses IP Geolocation. It checks the city of your "
        "Internet provider, which may be many kilometers away from your actual physical location. "
        "It is not accurate enough to verify if you are on campus."
    )
    
    if st.button("Verify My Location (IP Based)"):
        with st.spinner("Getting location based on IP..."):
            g = geocoder.ip('me')

            if g.ok:
                user_location = g.latlng  
                TARGET_LOCATION = (2.8327, 101.7032)
                ALLOWED_DISTANCE_METERS = 5000  # Increased to 5km due to inaccuracy

                distance = calculate_distance(user_location, TARGET_LOCATION)

                if distance <= ALLOWED_DISTANCE_METERS:
                    st.session_state['location_verified'] = True
                    st.success(f"Location Verified! Your IP resolves to within {ALLOWED_DISTANCE_METERS / 1000}km of the target.")
                    st.write(f"Approximate Location: {g.city}, {g.country}")
                else:
                    st.session_state['location_verified'] = False
                    st.error(f"Location Check Failed. Your IP resolves to a location too far from the target.")
            else:
                st.error("Could not determine location from IP address. Please try again.")

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
    
    #######
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            # Convert the frame to a numpy array
            img = frame.to_ndarray(format="bgr24")

        # Perform a simple operation (vertical flip)
            img = img[::-1, :, :]

        # Convert back to a VideoFrame and return
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        rtc_configuration=RTC_CONFIGURATION, # Add this config
        media_stream_constraints={"video": True, "audio": False} # Specify you only want video
    )
    
    st.divider()

    if st.button("Return to Verification"):
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