import json
import importlib.util
from pathlib import Path
import urllib.parse
import urllib.request

import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parents[1]
YOLO_MODEL_PATH = APP_DIR / "Model" / "weights" / "best.onnx"
LICENSE_PLATE_MODEL_PATH = (
    ROOT_DIR / "24_2_python" / "license_plate" / "Model" / "weights" / "best.onnx"
)
FACES_DIR = ROOT_DIR / "face-recognition" / "faces"
OCR_FONT_PATH = ROOT_DIR / "24_2_python" / "calibri.ttf"
HAS_FACE_RECOGNITION_PACKAGE = importlib.util.find_spec("face_recognition") is not None
HAS_OPENCV_FACE_RECOGNITION = (
    hasattr(cv, "face") and hasattr(cv.face, "LBPHFaceRecognizer_create")
)
HAS_FACE_RECOGNITION = (
    HAS_FACE_RECOGNITION_PACKAGE or HAS_OPENCV_FACE_RECOGNITION
)
HAS_EASYOCR = importlib.util.find_spec("easyocr") is not None
HAS_STREAMLIT_WEBRTC = importlib.util.find_spec("streamlit_webrtc") is not None
WEBRTC_DEVICE_PATCHED = False

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CUSTOM_OBJECT_LABELS = [
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
LICENSE_PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CAMERA_VIDEO_LAYOUT = [65, 35]
FACE_CASCADE_PATH = Path(cv.data.haarcascades) / "haarcascade_frontalface_default.xml"
OPENCV_FACE_SIZE = (160, 160)
OPENCV_LBPH_MIN_THRESHOLD = 40
OPENCV_LBPH_THRESHOLD_SCALE = 100
DEFAULT_WEBRTC_ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
]
WEBRTC_TURN_FETCH_TIMEOUT_SECONDS = 5
WEBRTC_FORCE_RELAY_WITH_TURN = True


st.set_page_config(page_title="Vision Projects", layout="wide")
st.title("Computer Vision Projects")


def decode_image(uploaded_file):
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv.imdecode(file_bytes, cv.IMREAD_COLOR)


def uploaded_image_input(key_prefix):
    uploaded_file = st.file_uploader(
        "Image",
        type=["jpg", "jpeg", "png"],
        key=f"{key_prefix}_upload",
    )
    return decode_image(uploaded_file)


def image_video_input(key_prefix, video_renderer):
    upload_tab, camera_tab = st.tabs(["Upload Image", "Camera Video"])
    with upload_tab:
        image = uploaded_image_input(key_prefix)
    with camera_tab:
        video_renderer()

    return image


def get_webrtc_secret(key):
    try:
        webrtc_secrets = st.secrets.get("webrtc", {})
    except Exception:
        return ""

    value = webrtc_secrets.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def normalize_metered_app_name(value):
    app_name = value.strip()
    app_name = app_name.removeprefix("https://").removeprefix("http://")
    app_name = app_name.split("/", maxsplit=1)[0]
    return app_name.removesuffix(".metered.live")


def parse_ice_servers_json(value):
    ice_servers = json.loads(value)
    if not isinstance(ice_servers, list) or not ice_servers:
        raise ValueError("ice_servers_json must be a non-empty JSON array.")
    return ice_servers


def get_ice_server_urls(ice_server):
    urls = ice_server.get("urls", []) if isinstance(ice_server, dict) else []
    if isinstance(urls, str):
        return [urls]
    if isinstance(urls, list):
        return [url for url in urls if isinstance(url, str)]
    return []


def has_turn_server(ice_servers):
    return any(
        url.startswith(("turn:", "turns:"))
        for ice_server in ice_servers
        for url in get_ice_server_urls(ice_server)
    )


def make_rtc_configuration(ice_servers):
    configuration = {"iceServers": ice_servers}
    if WEBRTC_FORCE_RELAY_WITH_TURN and has_turn_server(ice_servers):
        configuration["iceTransportPolicy"] = "relay"
    return configuration


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_metered_ice_servers(app_name, api_key):
    encoded_key = urllib.parse.quote(api_key)
    url = (
        f"https://{app_name}.metered.live/api/v1/turn/credentials"
        f"?apiKey={encoded_key}"
    )
    with urllib.request.urlopen(  # noqa: S310 - configured TURN credential endpoint
        url, timeout=WEBRTC_TURN_FETCH_TIMEOUT_SECONDS
    ) as response:
        payload = response.read().decode("utf-8")

    ice_servers = json.loads(payload)
    if not isinstance(ice_servers, list) or not ice_servers:
        raise ValueError("Metered did not return a non-empty iceServers array.")
    return ice_servers


def build_webrtc_rtc_configuration():
    ice_servers_json = get_webrtc_secret("ice_servers_json")
    if ice_servers_json:
        try:
            return make_rtc_configuration(parse_ice_servers_json(ice_servers_json))
        except Exception:
            st.warning("Invalid TURN server JSON in Streamlit secrets; using STUN only.")

    metered_app_name = normalize_metered_app_name(
        get_webrtc_secret("metered_app_name")
    )
    metered_api_key = get_webrtc_secret("metered_api_key")
    if metered_app_name and metered_api_key:
        try:
            return make_rtc_configuration(
                fetch_metered_ice_servers(metered_app_name, metered_api_key)
            )
        except Exception:
            st.warning("TURN server credentials could not be loaded; using STUN only.")

    return make_rtc_configuration(DEFAULT_WEBRTC_ICE_SERVERS)


def show_webrtc_network_status(rtc_configuration):
    ice_servers = rtc_configuration.get("iceServers", [])
    if has_turn_server(ice_servers):
        st.caption("Camera network: TURN relay")
    else:
        st.warning(
            "Camera network: STUN only. On Streamlit Cloud, add TURN secrets "
            "if the camera does not connect."
        )


def render_video_camera(key, video_processor_factory):
    if not HAS_STREAMLIT_WEBRTC:
        st.error("Install streamlit-webrtc to use camera video.")
        return

    patch_streamlit_webrtc_device_selection()

    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except Exception as exc:
        st.error(f"Camera video could not start: {exc}")
        return

    try:
        video_col, _ = st.columns(CAMERA_VIDEO_LAYOUT)
        with video_col:
            st.caption(
                "To switch cameras: STOP, SELECT DEVICE, choose the camera, "
                "Done, then START."
            )
            rtc_configuration = build_webrtc_rtc_configuration()
            show_webrtc_network_status(rtc_configuration)
            webrtc_streamer(
                key=key,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=video_processor_factory,
                async_processing=True,
            )
    except Exception as exc:
        st.error(f"Camera video could not initialize: {exc}")


def patch_streamlit_webrtc_device_selection():
    global WEBRTC_DEVICE_PATCHED
    if WEBRTC_DEVICE_PATCHED:
        return

    spec = importlib.util.find_spec("streamlit_webrtc")
    if spec is None or spec.origin is None:
        WEBRTC_DEVICE_PATCHED = True
        return

    asset_dir = Path(spec.origin).parent / "frontend" / "dist" / "assets"
    if not asset_dir.is_dir():
        WEBRTC_DEVICE_PATCHED = True
        return

    replacements = {
        "index-*.js": {
            "r.video={deviceId:t}": "r.video={deviceId:{exact:t}}",
            "r.video={...r.video,deviceId:t}": (
                "r.video={...r.video,deviceId:{exact:t}}"
            ),
            "r.audio={deviceId:n}": "r.audio={deviceId:{exact:n}}",
            "r.audio={...r.audio,deviceId:n}": (
                "r.audio={...r.audio,deviceId:{exact:n}}"
            ),
        },
        "DeviceSelectForm-*.js": {
            "video:{deviceId:e.deviceId},audio:!1": (
                "video:{deviceId:{exact:e.deviceId}},audio:!1"
            ),
            "video:t&&e?{deviceId:e}:t": (
                "video:t&&e?{deviceId:{exact:e}}:t"
            ),
            "audio:n&&r?{deviceId:r}:n": (
                "audio:n&&r?{deviceId:{exact:r}}:n"
            ),
        },
    }

    for pattern, patch_set in replacements.items():
        for asset_path in asset_dir.glob(pattern):
            text = asset_path.read_text()
            patched = text
            for old, new in patch_set.items():
                patched = patched.replace(old, new)
            if patched != text:
                try:
                    asset_path.write_text(patched)
                except OSError:
                    WEBRTC_DEVICE_PATCHED = True
                    return

    WEBRTC_DEVICE_PATCHED = True


def show_bgr_image(image, caption=None):
    st.image(cv.cvtColor(image, cv.COLOR_BGR2RGB), caption=caption)


def show_rgb_image(image, caption=None):
    st.image(image, caption=caption)


@st.cache_resource
def load_yolo_model(model_path):
    return create_yolo_model(model_path)


def create_yolo_model(model_path):
    net = cv.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net


def get_yolo_predictions(image, net):
    image_copy = image.copy()
    rows, cols, _ = image_copy.shape
    max_size = max(rows, cols)
    input_image = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    input_image[0:rows, 0:cols] = image_copy

    blob = cv.dnn.blobFromImage(
        input_image,
        1 / 255,
        (INPUT_HEIGHT, INPUT_WIDTH),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    preds = net.forward()
    return input_image, preds[0]


def non_max_suppression(input_image, detections):
    boxes = []
    confidences = []
    class_ids = []
    image_h, image_w = input_image.shape[:2]
    proportion_y = image_h / INPUT_HEIGHT
    proportion_x = image_w / INPUT_WIDTH

    for row in detections:
        confidence = row[4]
        if confidence <= 0.28:
            continue

        class_confidence = row[5:].max()
        class_id = row[5:].argmax()
        if class_confidence <= 0.02:
            continue

        cx, cy, w, h = row[:4]
        x = int((cx - 0.5 * w) * proportion_x)
        y = int((cy - 0.5 * h) * proportion_y)
        width = int(w * proportion_x)
        height = int(h * proportion_y)
        boxes.append([x, y, width, height])
        confidences.append(float(confidence))
        class_ids.append(int(class_id))

    if not boxes:
        return [], [], [], []

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    indexes = np.array(indexes).reshape(-1).tolist() if len(indexes) else []
    return boxes, confidences, class_ids, indexes


def draw_yolo_results(image, boxes, confidences, class_ids, indexes):
    result = image.copy()
    rows = []

    for index in indexes:
        x, y, w, h = boxes[index]
        confidence = confidences[index]
        class_id = class_ids[index]
        class_name = (
            CUSTOM_OBJECT_LABELS[class_id]
            if class_id < len(CUSTOM_OBJECT_LABELS)
            else f"class_{class_id}"
        )

        cv.rectangle(result, (x, y), (x + w, y + h), (255, 56, 0), 3)
        cv.rectangle(result, (x, y - 50), (x + w, y), (0, 0, 0), -1)
        cv.putText(
            result,
            f"{class_name} {int(confidence * 100)}%",
            (x, y - 10),
            cv.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 255),
            3,
        )
        rows.append(
            {
                "label": class_name,
                "confidence": round(confidence, 3),
                "box": f"{x}, {y}, {w}, {h}",
            }
        )

    return result, rows


def run_yolo(image, net):
    input_image, detections = get_yolo_predictions(image, net)
    boxes, confidences, class_ids, indexes = non_max_suppression(
        input_image, detections
    )
    return draw_yolo_results(image, boxes, confidences, class_ids, indexes)


def non_max_suppression_license_plate(input_image, detections):
    boxes = []
    confidences = []
    image_h, image_w = input_image.shape[:2]
    proportion_y = image_h / INPUT_HEIGHT
    proportion_x = image_w / INPUT_WIDTH

    for row in detections:
        confidence = row[4]
        if confidence <= 0.28:
            continue

        class_confidence = row[5] if len(row) > 5 else confidence
        if class_confidence <= 0.02:
            continue

        cx, cy, w, h = row[:4]
        x = int((cx - 0.5 * w) * proportion_x)
        y = int((cy - 0.5 * h) * proportion_y)
        width = int(w * proportion_x)
        height = int(h * proportion_y)
        boxes.append([x, y, width, height])
        confidences.append(float(confidence))

    if not boxes:
        return [], [], []

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    indexes = np.array(indexes).reshape(-1).tolist() if len(indexes) else []
    return boxes, confidences, indexes


@st.cache_resource
def load_known_faces(faces_dir, faces_signature):
    return create_known_faces(faces_dir)


def faces_dir_signature(faces_dir):
    paths = sorted(
        path
        for path in Path(faces_dir).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in paths)


def create_known_faces(faces_dir):
    if HAS_FACE_RECOGNITION_PACKAGE:
        return create_known_faces_with_face_recognition(faces_dir)
    if HAS_OPENCV_FACE_RECOGNITION:
        return create_known_faces_with_opencv(faces_dir)
    raise RuntimeError(
        "Install face_recognition or opencv-contrib-python-headless."
    )


def create_known_faces_with_face_recognition(faces_dir):
    import face_recognition

    encodings = []
    names = []
    for image_path in sorted(Path(faces_dir).iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = load_training_face_image(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            encodings.append(face_encodings[0])
            names.append(training_face_name(image_path))

    return encodings, names


@st.cache_resource
def load_opencv_face_detector():
    detector = cv.CascadeClassifier(str(FACE_CASCADE_PATH))
    if detector.empty():
        raise RuntimeError(f"OpenCV face cascade not found: {FACE_CASCADE_PATH}")
    return detector


def detect_opencv_face_boxes(gray):
    detector = load_opencv_face_detector()
    boxes = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    return sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)


def prepare_opencv_face_crop(gray, box=None):
    if box is not None:
        x, y, width, height = box
        gray = gray[y : y + height, x : x + width]
    face = cv.resize(gray, OPENCV_FACE_SIZE)
    return cv.equalizeHist(face)


def create_known_faces_with_opencv(faces_dir):
    face_images = []
    labels = []
    names = []
    name_to_label = {}

    for image_path in sorted(Path(faces_dir).iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = load_training_face_image(image_path)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        gray = cv.equalizeHist(gray)
        boxes = detect_opencv_face_boxes(gray)
        face = prepare_opencv_face_crop(gray, boxes[0] if boxes else None)
        name = training_face_name(image_path)
        if name not in name_to_label:
            name_to_label[name] = len(names)
            names.append(name)
        face_images.append(face)
        labels.append(name_to_label[name])

    if not face_images:
        return None, []

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(face_images, np.asarray(labels, dtype=np.int32))
    return recognizer, names


def load_training_face_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.array(image)


def training_face_name(image_path):
    stem = Path(image_path).stem
    for separator in ("_", "-"):
        name, found, suffix = stem.rpartition(separator)
        if found and suffix.isdigit() and name:
            return name
    return stem


def run_face_recognition(image, known_encodings, known_names, tolerance):
    if HAS_FACE_RECOGNITION_PACKAGE:
        return run_face_recognition_with_face_recognition(
            image, known_encodings, known_names, tolerance
        )
    if HAS_OPENCV_FACE_RECOGNITION:
        return run_face_recognition_with_opencv(
            image, known_encodings, known_names, tolerance
        )
    raise RuntimeError(
        "Install face_recognition or opencv-contrib-python-headless."
    )


def run_face_recognition_with_face_recognition(
    image, known_encodings, known_names, tolerance
):
    import face_recognition

    result = image.copy()
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    rows = []

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        name = "Unknown"
        distance = None

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = int(np.argmin(distances))
            distance = float(distances[best_match_index])
            if distance <= tolerance:
                name = known_names[best_match_index]

        cv.rectangle(result, (left, top), (right, bottom), (0, 160, 255), 2)
        cv.rectangle(result, (left, max(top - 34, 0)), (right, top), (0, 160, 255), -1)
        cv.putText(
            result,
            name,
            (left + 6, max(top - 10, 20)),
            cv.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 0, 0),
            1,
        )
        rows.append(
            {
                "name": name,
                "distance": round(distance, 3) if distance is not None else None,
                "box": f"{left}, {top}, {right - left}, {bottom - top}",
            }
        )

    return result, rows


def opencv_lbph_threshold(tolerance):
    return OPENCV_LBPH_MIN_THRESHOLD + (tolerance * OPENCV_LBPH_THRESHOLD_SCALE)


def run_face_recognition_with_opencv(image, recognizer, known_names, tolerance):
    result = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    boxes = detect_opencv_face_boxes(gray)
    rows = []
    threshold = opencv_lbph_threshold(tolerance)

    for x, y, width, height in boxes:
        face = prepare_opencv_face_crop(gray, (x, y, width, height))
        name = "Unknown"
        distance = None

        if recognizer is not None and known_names:
            label, confidence = recognizer.predict(face)
            distance = float(confidence)
            if 0 <= label < len(known_names) and confidence <= threshold:
                name = known_names[label]

        left, top, right, bottom = x, y, x + width, y + height
        cv.rectangle(result, (left, top), (right, bottom), (0, 160, 255), 2)
        cv.rectangle(result, (left, max(top - 34, 0)), (right, top), (0, 160, 255), -1)
        cv.putText(
            result,
            name,
            (left + 6, max(top - 10, 20)),
            cv.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 0, 0),
            1,
        )
        rows.append(
            {
                "name": name,
                "distance": round(distance, 1) if distance is not None else None,
                "box": f"{left}, {top}, {width}, {height}",
            }
        )

    return result, rows


@st.cache_resource
def load_ocr_reader(languages, use_gpu):
    return create_ocr_reader(languages, use_gpu)


def create_ocr_reader(languages, use_gpu):
    if not HAS_EASYOCR:
        raise RuntimeError("The easyocr package is not installed.")
    from easyocr import Reader

    return Reader(list(languages), gpu=use_gpu)


def get_font(size):
    try:
        return ImageFont.truetype(str(OCR_FONT_PATH), size)
    except OSError:
        return ImageFont.load_default()


def box_points(box):
    left_top, right_top, right_bottom, left_bottom = box
    return (
        (int(left_top[0]), int(left_top[1])),
        (int(right_top[0]), int(right_top[1])),
        (int(right_bottom[0]), int(right_bottom[1])),
        (int(left_bottom[0]), int(left_bottom[1])),
    )


def draw_ocr_text(image_rgb, text, x, y, color=(50, 50, 255), font_size=22):
    image_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image_pil)
    draw.text((x, max(y - font_size, 0)), text, font=get_font(font_size), fill=color)
    return np.array(image_pil)


def run_easyocr(image, reader):
    result = cv.cvtColor(image.copy(), cv.COLOR_BGR2RGB)
    detections = reader.readtext(result)
    rows = []

    for box, text, confidence in detections:
        left_top, _, right_bottom, _ = box_points(box)
        cv.rectangle(result, left_top, right_bottom, color=(255, 0, 50), thickness=2)
        result = draw_ocr_text(text=text, x=left_top[0], y=left_top[1], image_rgb=result)
        rows.append({"text": text, "confidence": round(float(confidence), 3)})

    return result, rows


def crop_box(image, x, y, width, height, padding=6):
    image_h, image_w = image.shape[:2]
    left = max(x - padding, 0)
    top = max(y - padding, 0)
    right = min(x + width + padding, image_w)
    bottom = min(y + height + padding, image_h)
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def read_plate_text(crop_bgr, reader):
    if reader is None or crop_bgr is None or crop_bgr.size == 0:
        return "", None

    crop_rgb = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)
    detections = reader.readtext(crop_rgb, allowlist=LICENSE_PLATE_ALLOWLIST)
    cleaned_text = []
    confidences = []

    for _, text, confidence in detections:
        normalized = "".join(char for char in text.upper() if char.isalnum())
        if normalized:
            cleaned_text.append(normalized)
            confidences.append(float(confidence))

    if not cleaned_text:
        return "", None

    average_confidence = sum(confidences) / len(confidences)
    return " ".join(cleaned_text), average_confidence


def run_license_plate_recognition(image, net, reader=None):
    input_image, detections = get_yolo_predictions(image, net)
    boxes, confidences, indexes = non_max_suppression_license_plate(
        input_image, detections
    )
    result = image.copy()
    rows = []

    for index in indexes:
        x, y, width, height = boxes[index]
        detector_confidence = confidences[index]
        crop = crop_box(image, x, y, width, height)
        plate_text, ocr_confidence = read_plate_text(crop, reader)
        label = plate_text if plate_text else "license plate"

        top = max(y, 0)
        left = max(x, 0)
        right = min(x + width, image.shape[1])
        bottom = min(y + height, image.shape[0])
        cv.rectangle(result, (left, top), (right, bottom), (30, 180, 255), 3)
        label_top = max(top - 38, 0)
        cv.rectangle(result, (left, label_top), (right, top), (0, 0, 0), -1)
        cv.putText(
            result,
            f"{label} {int(detector_confidence * 100)}%",
            (left + 5, max(top - 10, 24)),
            cv.FONT_HERSHEY_PLAIN,
            1.4,
            (255, 255, 255),
            2,
        )
        rows.append(
            {
                "plate_text": plate_text or None,
                "detector_confidence": round(detector_confidence, 3),
                "ocr_confidence": round(ocr_confidence, 3)
                if ocr_confidence is not None
                else None,
                "box": f"{left}, {top}, {right - left}, {bottom - top}",
            }
        )

    return result, rows


def render_rows(rows):
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No results found.")


class BaseVideoProcessor:
    def __init__(self):
        self.status_message = "Loading camera model..."

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        try:
            result = self.process(image)
        except Exception as exc:
            result = image.copy()
            cv.putText(
                result,
                str(exc)[:100],
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        result = np.ascontiguousarray(result)
        output_frame = type(frame).from_ndarray(result, format="bgr24")
        output_frame.pts = frame.pts
        output_frame.time_base = frame.time_base
        return output_frame

    def process(self, image):
        return image

    def draw_status(self, image):
        result = image.copy()
        cv.rectangle(result, (0, 0), (result.shape[1], 70), (0, 0, 0), -1)
        cv.putText(
            result,
            self.status_message,
            (20, 44),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        return result


class ObjectDetectionVideoProcessor(BaseVideoProcessor):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.net = None

    def process(self, image):
        if self.net is None:
            self.net = create_yolo_model(self.model_path)
        result, _ = run_yolo(image, self.net)
        return result


class LicensePlateVideoProcessor(BaseVideoProcessor):
    def __init__(self, model_path, read_text, languages, use_gpu):
        super().__init__()
        self.model_path = model_path
        self.read_text = read_text
        self.languages = languages
        self.use_gpu = use_gpu
        self.net = None
        self.reader = None

    def process(self, image):
        if self.net is None:
            self.net = create_yolo_model(self.model_path)
        if self.read_text and self.reader is None:
            self.reader = create_ocr_reader(self.languages, self.use_gpu)
        result, _ = run_license_plate_recognition(image, self.net, self.reader)
        return result


class FaceRecognitionVideoProcessor(BaseVideoProcessor):
    def __init__(self, faces_dir, tolerance, faces_signature):
        super().__init__()
        self.faces_dir = faces_dir
        self.faces_signature = faces_signature
        self.known_encodings = None
        self.known_names = None
        self.tolerance = tolerance

    def process(self, image):
        current_signature = faces_dir_signature(self.faces_dir)
        if (
            self.known_encodings is None
            or self.known_names is None
            or current_signature != self.faces_signature
        ):
            self.faces_signature = current_signature
            self.known_encodings, self.known_names = create_known_faces(self.faces_dir)
        result, _ = run_face_recognition(
            image, self.known_encodings, self.known_names, self.tolerance
        )
        return result


class EasyOCRVideoProcessor(BaseVideoProcessor):
    def __init__(self, languages, use_gpu):
        super().__init__()
        self.languages = languages
        self.use_gpu = use_gpu
        self.reader = None

    def process(self, image):
        if self.reader is None:
            self.reader = create_ocr_reader(self.languages, self.use_gpu)
        result, _ = run_easyocr(image, self.reader)
        return cv.cvtColor(result, cv.COLOR_RGB2BGR)


class CombinedVideoProcessor(BaseVideoProcessor):
    def __init__(self, faces_dir, tolerance, languages, use_gpu, faces_signature):
        super().__init__()
        self.faces_dir = faces_dir
        self.faces_signature = faces_signature
        self.known_encodings = None
        self.known_names = None
        self.tolerance = tolerance
        self.languages = languages
        self.use_gpu = use_gpu
        self.reader = None

    def process(self, image):
        current_signature = faces_dir_signature(self.faces_dir)
        if (
            self.known_encodings is None
            or self.known_names is None
            or current_signature != self.faces_signature
        ):
            self.faces_signature = current_signature
            self.known_encodings, self.known_names = create_known_faces(self.faces_dir)
        if self.reader is None:
            self.reader = create_ocr_reader(self.languages, self.use_gpu)
        face_result, _ = run_face_recognition(
            image, self.known_encodings, self.known_names, self.tolerance
        )
        ocr_result, _ = run_easyocr(face_result, self.reader)
        return cv.cvtColor(ocr_result, cv.COLOR_RGB2BGR)


def render_object_detection():
    st.header("YOLOV5 Custom Object Detection")
    if not YOLO_MODEL_PATH.exists():
        st.error(f"Model file not found: {YOLO_MODEL_PATH}")
        return

    image = image_video_input(
        "object_detection",
        lambda: render_video_camera(
            "object_detection_video",
            lambda: ObjectDetectionVideoProcessor(str(YOLO_MODEL_PATH)),
        ),
    )
    if image is None:
        return

    show_bgr_image(image, "Input")
    with st.spinner("Running object detection..."):
        net = load_yolo_model(str(YOLO_MODEL_PATH))
        result, rows = run_yolo(image, net)

    show_bgr_image(result, "Detections")
    render_rows(rows)


def render_face_recognition():
    st.header("Face Recognition")
    if not HAS_FACE_RECOGNITION:
        st.error(
            "Install face_recognition or opencv-contrib-python-headless "
            "to use this project."
        )
        return

    tolerance = st.slider("Tolerance", 0.35, 0.75, 0.6, 0.01)
    faces_signature = faces_dir_signature(FACES_DIR)
    image = image_video_input(
        "face_recognition",
        lambda: render_video_camera(
            "face_recognition_video",
            lambda: FaceRecognitionVideoProcessor(
                str(FACES_DIR), tolerance, faces_signature
            ),
        ),
    )
    if image is None:
        return

    with st.spinner("Loading known faces..."):
        known_encodings, known_names = load_known_faces(
            str(FACES_DIR), faces_signature
        )

    if not known_names:
        st.warning(f"No known face encodings found in {FACES_DIR}")

    with st.spinner("Recognizing faces..."):
        result, rows = run_face_recognition(
            image, known_encodings, known_names, tolerance
        )

    show_bgr_image(result, "Recognized faces")
    render_rows(rows)


def render_easyocr():
    st.header("EasyOCR")
    if not HAS_EASYOCR:
        st.error("Install easyocr to use this project.")
        return

    languages = st.multiselect("Languages", ["en", "es"], default=["en", "es"])
    use_gpu = st.checkbox("GPU", value=False)
    if not languages:
        st.warning("Select at least one language.")
        return

    image = image_video_input(
        "easyocr",
        lambda: render_video_camera(
            "easyocr_video",
            lambda: EasyOCRVideoProcessor(tuple(languages), use_gpu),
        ),
    )
    if image is None:
        return

    with st.spinner("Reading text..."):
        reader = load_ocr_reader(tuple(languages), use_gpu)
        result, rows = run_easyocr(image, reader)

    show_rgb_image(result, "OCR results")
    render_rows(rows)


def render_license_plate_recognition():
    st.header("License Plate Recognition")
    if not LICENSE_PLATE_MODEL_PATH.exists():
        st.error(f"Model file not found: {LICENSE_PLATE_MODEL_PATH}")
        return

    if not HAS_EASYOCR:
        st.info(
            "EasyOCR is not installed in this deployment, so license plate text "
            "reading is disabled. Plate detection still works."
        )

    read_text = st.checkbox("Read plate text with EasyOCR", value=HAS_EASYOCR)

    if read_text and not HAS_EASYOCR:
        st.error("Install easyocr to read license plate text.")
        return

    languages = []
    use_gpu = False
    if read_text:
        languages = st.multiselect(
            "OCR languages",
            ["en", "es"],
            default=["en"],
            key="license_plate_languages",
        )
        use_gpu = st.checkbox("GPU", value=False, key="license_plate_gpu")

    if read_text and not languages:
        st.warning("Select at least one OCR language.")
        return

    image = image_video_input(
        "license_plate",
        lambda: render_video_camera(
            "license_plate_video",
            lambda: LicensePlateVideoProcessor(
                str(LICENSE_PLATE_MODEL_PATH), read_text, tuple(languages), use_gpu
            ),
        ),
    )
    if image is None:
        return

    with st.spinner("Detecting license plates..."):
        net = load_yolo_model(str(LICENSE_PLATE_MODEL_PATH))
        reader = load_ocr_reader(tuple(languages), use_gpu) if read_text else None
        result, rows = run_license_plate_recognition(image, net, reader)

    show_bgr_image(result, "License plate results")
    render_rows(rows)


def render_combined():
    st.header("Face Recognition + EasyOCR")
    missing = []
    if not HAS_FACE_RECOGNITION:
        missing.append("face_recognition or opencv-contrib-python-headless")
    if not HAS_EASYOCR:
        missing.append("easyocr")
    if missing:
        st.error(f"Install {', '.join(missing)} to use this project.")
        return

    tolerance = st.slider("Tolerance", 0.35, 0.75, 0.6, 0.01, key="combined_tol")
    languages = st.multiselect(
        "Languages", ["en", "es"], default=["en", "es"], key="combined_languages"
    )
    use_gpu = st.checkbox("GPU", value=False, key="combined_gpu")
    faces_signature = faces_dir_signature(FACES_DIR)

    if not languages:
        st.warning("Select at least one language.")
        return

    image = image_video_input(
        "combined",
        lambda: render_video_camera(
            "combined_video",
            lambda: CombinedVideoProcessor(
                str(FACES_DIR), tolerance, tuple(languages), use_gpu, faces_signature
            ),
        ),
    )
    if image is None:
        return

    with st.spinner("Running face recognition and OCR..."):
        known_encodings, known_names = load_known_faces(
            str(FACES_DIR), faces_signature
        )
        face_result, face_rows = run_face_recognition(
            image, known_encodings, known_names, tolerance
        )
        reader = load_ocr_reader(tuple(languages), use_gpu)
        ocr_result, ocr_rows = run_easyocr(face_result, reader)

    show_rgb_image(ocr_result, "Combined results")
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Faces")
        render_rows(face_rows)
    with right_col:
        st.subheader("Text")
        render_rows(ocr_rows)


project = st.sidebar.radio(
    "Project",
    [
        "Object Detection",
        "License Plate Recognition",
        "Face Recognition",
        "EasyOCR",
        "Face + OCR",
    ],
)

if project == "Object Detection":
    render_object_detection()
elif project == "License Plate Recognition":
    render_license_plate_recognition()
elif project == "Face Recognition":
    render_face_recognition()
elif project == "EasyOCR":
    render_easyocr()
else:
    render_combined()
