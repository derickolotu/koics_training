import face_recognition
import cv2
import numpy as np
import os


known_faces_encodings = []
known_faces_names = []

faces_dir = "faces"

# We go through each image in the folder
for filename in os.listdir(faces_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_faces_encodings.append(encodings[0])
            known_faces_names.append(os.path.splitext(filename)[0])


cam = cv2.VideoCapture(0)
process_frame = True

while True:
    connected, frame = cam.read()

    if process_frame:
        small_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_faces_encodings, face_encoding
            )

            face_distance = face_recognition.face_distance(
                known_faces_encodings, face_encoding
            )

            name = "unkown"

            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

    process_frame = not process_frame

    for (left, top, bottom, right), name in zip(face_locations, face_names):
        top *= 4
        left *= 4
        bottom *= 4
        right *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.putText(
            frame, name, (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255)
        )

        cv2.imshow("camera", frame)

    if cv2.waitKey(0) == ord("q"):
        break


cam.release()
cv2.destroyAllWindows()
