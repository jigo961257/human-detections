import cv2
import time
import numpy as np
from datetime import datetime

# Configuration
WEBCAM_INDEX = 1  # Typically 0 for built-in webcam, 1 for external
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"
CONFIDENCE_THRESHOLD = 0.5
HUMAN_CLASS_ID = 15  # Person class ID in MobileNet SSD
NO_HUMAN_WAIT_TIME = 30  # seconds to wait when no human is detected

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
                label = f"Person: {confidence * 100:.2f}%"
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, human_detected

# Connect to webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)

# Set webcam properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)          # Limit frame rate

if not cap.isOpened():
    log_message("Error: Could not open webcam")
    exit()

log_message("System started. Monitoring for humans using webcam...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            log_message("Error reading frame from webcam")
            time.sleep(1)
            continue
        
        # Flip frame horizontally for mirror effect (more intuitive)
        frame = cv2.flip(frame, 1)
        
        # Detect humans
        processed_frame, human_detected = detect_humans(frame)
        
        # Display the frame
        cv2.imshow('Webcam Human Detection', processed_frame)
        
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
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    log_message("System stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    log_message("System shutdown")