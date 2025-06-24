import os
import cv2
import time
from datetime import datetime

import numpy as np

# Configuration
RTSP_URL = "rtsp://forthtech-food:tyttet-cejvam-hiSmu0@192.168.1.74:554/stream1?tcp&buffer_size=8192&framerate=15&resolution=720p"  # Replace with your RTSP URL
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"
CONFIDENCE_THRESHOLD = 0.5
HUMAN_CLASS_ID = 15  # Person class ID in MobileNet SSD
NO_HUMAN_WAIT_TIME = 10  # seconds to wait when no human is detected

# Load pre-trained model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# Initialize variables
last_human_detected_time = None
process_started = False

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def detect_humans(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    human_detected = False
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = int(detections[0, 0, i, 1])
            
            if class_id == HUMAN_CLASS_ID:
                human_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    return frame, human_detected

# Connect to RTSP stream
# Replace your VideoCapture initialization with:
cap = cv2.VideoCapture()
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce internal buffer
cap.set(cv2.CAP_PROP_FPS, 15)        # Request lower FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))  # Force H264
cap.open(RTSP_URL)

# Add network optimization parameters
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;8192"

if not cap.isOpened():
    log_message("Error: Could not open RTSP stream")
    exit()

log_message("System started. Monitoring for humans...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            log_message("Error reading frame. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            time.sleep(1)
            continue
        
        # Detect humans
        processed_frame, human_detected = detect_humans(frame)
        
        # Display the frame (optional)
        cv2.imshow('Human Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Handle human detection logic
        if human_detected:
            last_human_detected_time = time.time()
            if process_started:
                log_message("Human detected, process stop")
                process_started = False
        else:
            current_time = time.time()
            if last_human_detected_time is None:
                last_human_detected_time = current_time
            
            # If no human for 30 seconds
            if (current_time - last_human_detected_time) >= NO_HUMAN_WAIT_TIME and not process_started:
                log_message("No human detected for 30 seconds, process start")
                process_started = True
        
        time.sleep(0.1)  # Reduce CPU usage

except KeyboardInterrupt:
    log_message("System stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    log_message("System shutdown")