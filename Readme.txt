# Face & Location-Based Attendance System

This is a comprehensive attendance recording system built with Streamlit. It features a dual-role interface for Students and Lecturers, incorporating modern verification methods like facial recognition and geolocation to ensure attendance integrity.

## ✨ Features

### General
- 👤 **Role-Based Access:** Separate, secure login portals for Students and Lecturers.
- ⚙️ **Dynamic Interface:** The user interface changes based on the user's role and actions.
- 💨 **Efficient Model Loading:** Machine learning models are cached on first load for a fast and responsive user experience.

### 👨‍🎓 For Students
- 📚 **Class Selection:** Students can select their class from a dropdown menu.
- ✅ **Two-Factor Attendance Verification:**
    1.  **Face Verification:** Uses the device's camera to capture the student's face and verifies their identity against a registered profile using a pre-trained facial recognition model.
    2.  **Location Verification:** Uses the browser's geolocation API to ensure the student is within a predefined distance from the campus.
- 📝 **Checklist & Submission:** A clear checklist shows the status of both verification steps before the final attendance submission is allowed.

### 👨‍🏫 For Lecturers
- 📊 **View Attendance:** Lecturers can view and monitor attendance records for each class.
- 📥 **Download Records:** Attendance data for any class can be downloaded as a `.csv` file for offline analysis or record-keeping.
- ➕ **Add New Students:** A streamlined workflow to add new students to the system:
    - Upload one or more student photos.
    - Enter student names (usernames).
    - The system auto-generates a secure password.
    - A confirmation screen ensures data accuracy before saving.
- 🗑️ **Manage Students:** Lecturers can view a list of all registered students and delete student records and their associated photos if necessary.

## 🛠️ Technology Stack

- **Framework:** Streamlit
- **Data Handling:** Pandas, NumPy
- **Machine Learning & Computer Vision:**
    - TensorFlow (for FaceNet model)
    - Scikit-learn (for classification)
    - OpenCV (for image processing & face detection)
    - Pillow (PIL)
- **Geolocation:** Geopy, `streamlit-geolocation`
- **File I/O:** `os`, `joblib`, `shutil`

## 📂 Project Structure

Your project directory should be set up as follows for the application to work correctly:
your-project-root/
│
├── 📜 main_app.py               # Your main Streamlit application file
├── 🔑 credentials.py            # Stores user login credentials
├── 🛠️ utils.py                  # Helper functions (load_facenet_model, get_embedding)
├── 📋 requirements.txt          # List of Python dependencies
│
├── 📁 data/
│   └── 📄 attendance_BSC124.csv  # Example attendance file (created automatically)
│
├── 📁 facenet_model/
│   └── 📄 20180402-114759.pb    # The pre-trained FaceNet model
│
├── 📁 trained_model/
│   ├── 📄 classifier.joblib       # Your trained classifier model
│   └── 📄 label_encoder.joblib    # Your trained label encoder
│
└── 📁 student_photos/
└── 📄 student_name.jpg      # Student photos are saved here automatically