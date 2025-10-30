import streamlit as st
from ultralytics import YOLO
import cv2
import yt_dlp
import threading
import queue
import numpy as np
import time

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="YOLO Live Stream", layout="wide")
st.title("🎥 Real-Time YOLO Stream Detection")
st.caption("Detect people, cars, etc. on a YouTube live stream in near real-time")

# Input (you can also use st.text_input if you want dynamic entry)
yt_url = "https://www.youtube.com/watch?v=6dp-bvQ7RWo"
#https://www.youtube.com/watch?v=cH7VBI4QQzA
# -------------------- YouTube Stream Setup --------------------
st.write("🔄 Connecting to stream...")

ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(yt_url, download=False)
    stream_url = info['url']

cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    st.error("❌ Unable to open YouTube stream.")
    st.stop()

st.success("✅ Connected to stream!")
st.info("Press **Stop** (square icon in toolbar) to end.")

# -------------------- YOLO Model --------------------
model = YOLO("yolov8n.pt")

# -------------------- Frame Reader Thread --------------------
frame_queue = queue.Queue(maxsize=1)
stop_flag = False

def read_frames():
    """Continuously read frames from stream and keep only the latest one."""
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            continue
        # Keep the queue fresh — discard old frame
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()
                frame_queue.put(frame)
            except:
                pass

reader_thread = threading.Thread(target=read_frames, daemon=True)
reader_thread.start()

# -------------------- Display Loop --------------------
frame_placeholder = st.empty()
target_fps = 25
frame_interval = 1.0 / target_fps

while True:
    start_time = time.time()

    if not frame_queue.empty():
        frame = frame_queue.get()

        # Resize for faster processing
        frame = cv2.resize(frame, (960, 540))


        # Run YOLO inference
        results = model(frame, stream=True)
        for r in results:
            annotated = r.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated, channels="RGB", use_container_width=True)

    # Keep display in sync with real-time playback
    elapsed = time.time() - start_time
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)

# -------------------- Cleanup --------------------
cap.release()
stop_flag = True
reader_thread.join()
