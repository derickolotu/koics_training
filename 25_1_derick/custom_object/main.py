import cv2 as cv
import numpy as np
import streamlit as st

st.title("YOLOV5 Custom Object Detections")

uploaded_file = st.file_uploader(
    label="Please upload an image", type=["jpg", "png", "jpeg"]
)


labels = [
    "person",
    "car",
    "chair",
    "bottle",
    "pottedplant",
    "bird",
    "dog",
    "sofa",
    "bicycle",
    "horse",
    "boat",
    "motorbike",
    "cat",
    "tvmonitor",
    "cow",
    "sheep",
    "aeroplane",
    "train",
    "diningtable",
    "bus",
]
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

net = cv.dnn.readNetFromONNX("Model/weights/best.onnx")


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, 1)

    st.image(cv.cvtColor(image, cv.COLOR_BGR2RGB), caption="This is the uploaded image")

    def get_predictions(image, net):
        image_copy = image.copy()
        r, c, d = image_copy.shape

        max_rc = max(r, c)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:r, 0:c] = image

        # convert the image to blob format BGR RGB
        blob = cv.dnn.blobFromImage(
            input_image, 1 / 255, (INPUT_HEIGHT, INPUT_WIDTH), swapRB=True, crop=False
        )
        net.setInput(blob)
        preds = net.forward()
        detections = preds[0]

        return input_image, detections

    def non_max_suppression(input_image, detections):
        boxes = []
        confidences = []
        class_ids = []

        image_h, image_w = input_image.shape[:2]
        proportion_y = image_h / INPUT_HEIGHT
        proportion_x = image_w / INPUT_WIDTH
        # [cx, cy, w, h, confidence, 0.3, 0.8, 0]
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.28:
                class_confidence = row[5:].max()
                class_id = row[5:].argmax()
                if class_confidence > 0.02:
                    cx, cy, w, h = row[:4]
                    x = int((cx - 0.5 * w) * proportion_x)
                    y = int((cy - 0.5 * h) * proportion_y)
                    width = int(w * proportion_x)
                    height = int(h * proportion_y)
                    box = np.array([x, y, width, height])

                    boxes.append(box)
                    confidences.append(confidence)
                    class_ids.append(class_id)
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()
        class_ids_np = np.array(class_ids).tolist()

        index = cv.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)  # [3]

        return boxes_np, confidences_np, class_ids_np, index

    def results(image, boxes_np, confidences_np, class_ids_np, index):
        for ind in index:
            x, y, w, h = boxes_np[ind]
            bb_conf = int(confidences_np[ind] * 100)
            class_id = class_ids_np[ind]
            class_name = labels[class_id]

            cv.rectangle(image, (x, y), (x + w, y + h), (255, 56, 0), 3)
            cv.rectangle(image, (x, y - 50), (x + w, y), (0, 0, 0), -1)
            cv.putText(
                image,
                f"{class_name} {bb_conf}%",
                (x, y - 10),
                cv.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                3,
            )

        return image

    def yolo_custom(img, net):
        input_image, detections = get_predictions(img, net)
        boxes_np, confidences_np, class_ids_np, index = non_max_suppression(
            input_image, detections
        )
        result = results(img, boxes_np, confidences_np, class_ids_np, index)
        return result

    result = yolo_custom(image, net)
    st.image(result)
