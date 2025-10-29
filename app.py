# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import yt_dlp
import tempfile
import numpy as np

st.set_page_config(page_title="YOLO Live Stream", layout="wide")

st.title("🎥 YOLO Live Stream Object Detection")
st.caption("Detect cars, pedestrians, and cyclists in a YouTube live stream ")

# Input field for YouTube live URL
yt_url ="https://www.youtube.com/watch?v=tujkoXI8rWM"

if yt_url:
    st.write("🔄 Connecting to stream...")

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Get direct stream URL
    ydl_opts = {'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(yt_url, download=False)
        stream_url = info['url']

    # Open video stream
    cap = cv2.VideoCapture(stream_url)
    frame_placeholder = st.empty()

    st.write("✅ Streaming... press **Stop** (square icon) to end")

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Stream ended or unavailable.")
            break

        # Resize for speed
        frame = cv2.resize(frame, (640, 360))

        # Run YOLO inference
        results = model(frame, stream=True, classes=[0, 1, 2])
        for r in results:
            annotated_frame = r.plot()

            # Convert BGR to RGB for Streamlit display
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame, channels="RGB")

    cap.release()
