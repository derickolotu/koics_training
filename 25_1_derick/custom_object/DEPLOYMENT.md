# Free Deployment

Recommended host: Streamlit Community Cloud.

Use these settings:

- Repository: `derickolotu/koics_training`
- Branch: `main`
- Main file path: `25_1_derick/custom_object/main.py`
- Python version: `3.12`

Notes:

- The app uses `streamlit-webrtc`, so camera access must run over HTTPS. Streamlit Community Cloud provides HTTPS automatically.
- Python dependencies are in the repository root so Community Cloud detects them reliably.
- The free Community Cloud deployment excludes `face-recognition` and `easyocr` because they pull large native or ML dependencies that commonly fail or time out on free builders. The deployed app will still run object detection, license plate detection, and camera video. Face Recognition and EasyOCR remain available locally where those packages are installed.
