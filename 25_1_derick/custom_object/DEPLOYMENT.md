# Free Deployment

Recommended host: Streamlit Community Cloud.

Use these settings:

- Repository: `derickolotu/koics_training`
- Branch: `main`
- Main file path: `25_1_derick/custom_object/main.py`
- Python version: `3.12`

Notes:

- The app uses `streamlit-webrtc`, so camera access must run over HTTPS. Streamlit Community Cloud provides HTTPS automatically.
- The dependency files are in the repository root so Community Cloud detects both Python packages and Debian packages reliably.
- The free Community Cloud deployment excludes `face-recognition` because it requires `dlib`, which commonly fails or times out while compiling on free builders. The deployed app will still run object detection, license plate detection, camera video, and EasyOCR. Face Recognition remains available locally where `face-recognition` is installed.
