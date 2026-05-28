import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

# Load all known face images and encodings
known_face_encodings = []
known_face_names = []

faces_dir = "faces"

for filename in os.listdir(faces_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Make sure face is detected
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Attendance tracking
attendance_log = "attendance.csv"
logged_today = set()


def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    entry = f"{name},{date},{time}\n"

    # Avoid logging duplicates for the same person and day
    unique_id = f"{name}-{date}"
    if unique_id not in logged_today:
        with open(attendance_log, "a") as f:
            f.write(entry)
        logged_today.add(unique_id)
        print(f"Logged: {entry.strip()}")


# Setup webcam
video_capture = cv2.VideoCapture(0)
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name != "Unknown":
                mark_attendance(name)

    process_this_frame = not process_this_frame

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
