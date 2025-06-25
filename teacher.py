import streamlit as st
import pandas as pd
import random
import string
from PIL import Image

# Initialize page states
if "page" not in st.session_state:
    st.session_state.page = "home"
if "preview_data" not in st.session_state:
    st.session_state.preview_data = []

# Function to reset to home page
def go_home():
    st.session_state.page = "home"
    st.session_state.preview_data.clear()
    st.rerun() #refresh and back to the page you set(line 15)

# Home page
if st.session_state.page == "home":
    st.title("👩‍🏫 Lecturer Online Managing System (LOMS)")
    option = st.selectbox("Please select an operation.", ["Please select", "📥 Add New Student", "📊 View Attendance List"])

    if option == "📥 Add New Student":
        st.session_state.page = "add"
        st.rerun() #refresh and back to the page you set(line 25)
    elif option == "📊 View Attendance List":
        st.session_state.page = "view"
        st.rerun() #refresh and back to the page you set(line 28)

# Add new student page
elif st.session_state.page == "add":
    st.title("📥 Add New Student")

    uploaded_photos = st.file_uploader("Upload Student Photo", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    temp_data = []

    if uploaded_photos:
        st.subheader("Please enter student's name：")

        with st.form("student_form"):
            temp_data.clear()  # clear cached data
            for i, file in enumerate(uploaded_photos):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(file, width=100)
                with col2:
                    name = st.text_input(f"Student {i+1} name", key=f"name_{i}")
                temp_data.append({"file": file, "name": name})

            submitted = st.form_submit_button("➡️ Next: Preview and Generate Password")

        if submitted:
            def generate_password(length=6):
                return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

            st.session_state.preview_data.clear()
            all_ok = True

            for entry in temp_data:
                name = entry["name"]
                file = entry["file"]
                if not name:
                    st.warning(f"{file.name} Does not enter a name. Please fill in student's name")
                    all_ok = False
                    break
                password = generate_password()
                st.session_state.preview_data.append({
                    "PhotoName": file.name,
                    "Name": name,
                    "Password": password,
                    "File": file
                })

            if all_ok:
                st.success("Please confirm the student data below:")
                for s in st.session_state.preview_data:
                    st.image(s["File"], width=100)
                    st.markdown(f"**Name：** {s['Name']}")
                    st.markdown(f"🔒 **Password：** `{s['Password']}`")
                st.session_state.page = "confirm"
                st.rerun()

    if st.button("↩️ Back to Menu"):
        go_home()

# confirm studant data page
elif st.session_state.page == "confirm":
    st.title("✅ Student Data Confirmation")

    for s in st.session_state.preview_data:
        st.image(s["File"], width=100)
        st.markdown(f"**Name：** {s['Name']}")
        st.markdown(f"🔒 **Password：** `{s['Password']}`")

    if st.button("✅ Confirm and Save Data"):
        import os

        # 1️⃣ Create a directory for student photos if it doesn't exist
        os.makedirs("student_photos", exist_ok=True)

        # 2️⃣ Create new student data list and save photos
        new_students = []
        for s in st.session_state.preview_data:
            photo_path = f"student_photos/{s['PhotoName']}"
            with open(photo_path, "wb") as out_file:
                out_file.write(s["File"].getbuffer())
            new_students.append({
                'name': s['Name'],
                'photo': photo_path,  # Save the path to the photo
                'password': s['Password']
            })

        # 3️⃣ load existing student data if available
        existing_students = []
        if os.path.exists("student_data.py"):
            try:
                from student_data import student_list
                existing_students = student_list
            except Exception as e:
                st.warning("⚠️ Unable to read old student_data.py. Only new students will be saved.")

        # 4️⃣ Combine existing and new students
        full_list = existing_students + new_students

        # 5️⃣ Write the combined student data to student_data.py
        with open("student_data.py", "w", encoding="utf-8") as f:
            f.write("student_list = [\n")
            for s in full_list:
                f.write(f"    {{'name': '{s['name']}', 'photo': '{s['photo']}', 'password': '{s['password']}'}},\n")
            f.write("]\n")

        st.success("✅ Student data saved and photos written to 'student_photos/' folder.")
        go_home()

    if st.button("↩️ Back to Add Student"):
        st.session_state.page = "add"
        st.rerun()

# view attendance list page
elif st.session_state.page == "view":
    st.title("📊 View Attendance List")

    try:
        from student_data import student_list
    except:
        st.error("❌ Data file not found. Please upload student data first.")
        if st.button("↩️ Back to Home"):
            go_home()
        st.stop()

    try:
        from attendance import attendance_data
    except:
        st.error("❌ Attendance data not found. Please ensure attendance data is available.")
        if st.button("↩️ Back to Home"):
            go_home()
        st.stop()

    # generate attendance table
    student_names = [s["name"] for s in student_list]
    date_columns = list(attendance_data.keys())

    st.subheader("🧾 Attendance Table")

    # Header row
    header_cols = st.columns([2] + [1] * len(date_columns))
    header_cols[0].markdown("**Student Name**")
    for i, date in enumerate(date_columns):
        header_cols[i + 1].markdown(f"**{date}**")

    # Data rows
    for student in student_list:
        name = student["name"]
        row_cols = st.columns([2] + [1] * len(date_columns))

        # 👇 make name into button
        if row_cols[0].button(name, key=f"student_link_{name}"):
            st.session_state.selected_student = name
            st.session_state.page = "student_detail"
            st.rerun()

        for i, date in enumerate(date_columns):
            status = attendance_data.get(date, {}).get(name, "")
            row_cols[i + 1].write(status)

    # 🧾 Regenerate DataFrame for downloading CSV
    table = []

    for student in student_list:
        name = student["name"]
        row = {"Student Name": name}
        for date in date_columns:
            row[date] = attendance_data.get(date, {}).get(name, "")
        table.append(row)

    df = pd.DataFrame(table)

    # download button for attendance table
    st.download_button("📥 Download Attendance List (CSV)",
                       data=df.to_csv(index=False).encode('utf-8'),
                       file_name="attendance_table.csv",
                       mime="text/csv")

    if st.button("↩️ Back to Home"):
        go_home()

elif st.session_state.page == "student_detail":
    st.title("🧑‍🎓 Student Detail")

    selected_name = st.session_state.get("selected_student")

    if not selected_name:
        st.warning("No student selected.")
        if st.button("↩️ Back to Attendance"):
            st.session_state.page = "view"
            st.rerun()
        st.stop()

    # search student data
    from student_data import student_list
    student = next((s for s in student_list if s["name"] == selected_name), None)

    if student:
        st.image(student["photo"], width=150)
        st.markdown(f"**Name:** {student['name']}")
        st.markdown(f"🔒 **Password:** `{student['password']}`")

        if st.button("🗑️ Delete this Student"):
            import os

            # delete student
            remaining_students = [s for s in student_list if s["name"] != selected_name]

            # rewrite student.py
            with open("student_data.py", "w", encoding="utf-8") as f:
                f.write("student_list = [\n")
                for s in remaining_students:
                    f.write(f"    {{'name': '{s['name']}', 'photo': '{s['photo']}', 'password': '{s['password']}'}},\n")
                f.write("]\n")

            st.success(f"✅ {selected_name} has been deleted.")
            st.session_state.page = "view"
            st.rerun()
        
    else:
        st.error("Student not found.")

    if st.button("↩️ Back to Attendance"):
        st.session_state.page = "view"
        st.rerun()

