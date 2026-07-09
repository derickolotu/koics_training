# Free Deployment

Recommended host: Streamlit Community Cloud.

Use these settings:

- Repository: `derickolotu/koics_training`
- Branch: `main`
- Main file path: `25_1_derick/custom_object/main.py`
- Python version: `3.12`

Notes:

- The app uses `streamlit-webrtc`, so camera access must run over HTTPS. Streamlit Community Cloud provides HTTPS automatically.
- The dependency files live beside `main.py` because Community Cloud looks in the entrypoint directory for `requirements.txt` and `packages.txt`.
- If the app build fails, open the Streamlit Cloud logs first. The likely failure point is compiling `dlib`, which is required by `face-recognition`.
