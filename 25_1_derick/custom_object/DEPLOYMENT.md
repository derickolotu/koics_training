# Free Deployment

Recommended host: Streamlit Community Cloud.

Use these settings:

- Repository: `derickolotu/koics_training`
- Branch: `main`
- Main file path: `25_1_derick/custom_object/main.py`
- Python version: `3.12`

Notes:

- The app uses `streamlit-webrtc`, so camera access must run over HTTPS. Streamlit Community Cloud provides HTTPS automatically.
- WebRTC camera streaming can still fail on some networks. If the camera does not connect and the logs show `Transaction.__retry()` or ICE connection errors, use a different network or configure a TURN server. Public STUN servers are configured in the app, but they are not enough for every firewall/NAT.
- Python dependencies are in the repository root so Community Cloud detects them reliably.
- The free Community Cloud deployment excludes `face-recognition` and `easyocr` because they pull large native or ML dependencies that commonly fail or time out on free builders. The deployed app will still run object detection, license plate detection, and camera video. Face Recognition and EasyOCR remain available locally where those packages are installed.

## Free TURN setup for camera video

Use this when Streamlit Cloud logs show `Transaction.__retry()` or ICE errors.

1. Create a free Metered/Open Relay account.
2. Create or copy your Open Relay app name and TURN REST API key.
3. In Streamlit Cloud, open `Manage app` -> `Settings` -> `Secrets`.
4. Add:

```toml
[webrtc]
metered_app_name = "your-metered-app-name"
metered_api_key = "your-metered-turn-api-key"
```

5. Save secrets, then reboot the app.

After reboot, the camera tab should show `Camera network: TURN relay`. If it still says `Camera network: STUN only`, the secrets were not saved, the key names are different, or the TURN credential request failed.

You can also use another TURN provider by setting a raw `iceServers` array:

```toml
[webrtc]
ice_servers_json = '''
[
  {"urls": ["stun:stun.l.google.com:19302"]},
  {
    "urls": ["turn:your-turn-host.example:443?transport=tcp"],
    "username": "your-username",
    "credential": "your-password"
  }
]
'''
```

Do not commit real TURN credentials to GitHub.
