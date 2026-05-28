import face_recognition
import cv2
import numpy as np
import os


known_faces_encodings = []
known_faces_names = []

faces_dir = "faces"


# Go through each image
for filename in os.listdir(faces_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        # add only the faces found in the image
        if encodings:
            known_faces_encodings.append(encodings[0])
            known_faces_names.append(os.path.splitext(filename)[0])


cam = cv2.VideoCapture(0)
process_frame = True


while True:
    connected, frame = cam.read()

    if process_frame:
        # reduce the frame size for high perfomance
        resized_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        # Get all face encodings in the frame
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
                known_faces_encodings, face_encoding
            )  # [False, True, False]

            faces_distances = face_recognition.face_distance(
                known_faces_encodings, face_encoding
            )

            name = "Unknown"
            # get the lowest value from the list. remembe the lower the number the similar the faces look
            best_index = np.argmin(faces_distances)
            if matches[best_index]:
                name = known_faces_names[best_index]
            face_names.append(name)
    process_frame = False

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back the points to the original frame size
        top *= 4
        left *= 4
        bottom *= 4
        right *= 4
        print(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 100, 255), 2)
        cv2.putText(
            frame,
            name,
            (left, top - 16),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
        )

    cv2.imshow("Recognition", frame)

    # wait for the q key to be pressed then break the loop
    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
