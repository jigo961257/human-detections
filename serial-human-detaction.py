import cv2
import time
import numpy as np
from datetime import datetime
import serial

# Configuration
WEBCAM_INDEX = 1
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"
CONFIDENCE_THRESHOLD = 0.5
HUMAN_CLASS_ID = 15

# Serial port configuration
SERIAL_PORT = '/dev/cu.usbserial-120'  # Change to your Arduino port (COM3, COM4, /dev/ttyUSB0, etc.)
BAUD_RATE = 9600

# Detection area configuration (percentage of frame width/height)
DETECTION_AREA = {
    'x_start': 0.25,
    'y_start': 0.25,
    'x_end': 0.75,
    'y_end': 0.75
}

# Load pre-trained model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# Initialize serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for connection to establish
    print(f"Connected to serial port {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    ser = None

# Initialize variables
last_human_state = None

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def is_in_detection_area(box, frame_width, frame_height):
    """Check if detection is within our defined area"""
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    
    x_min = frame_width * DETECTION_AREA['x_start']
    x_max = frame_width * DETECTION_AREA['x_end']
    y_min = frame_height * DETECTION_AREA['y_start']
    y_max = frame_height * DETECTION_AREA['y_end']
    
    return (x_min <= x_center <= x_max) and (y_min <= y_center <= y_max)

def send_serial_data(state):
    """Send data over serial if port is open"""
    global last_human_state
    
    # Only send if state changed
    if state != last_human_state and ser is not None:
        try:
            data = b'1' if state else b'0'
            ser.write(data)
            log_message(f"Serial data sent: {data.decode()}")
            last_human_state = state
        except serial.SerialException as e:
            log_message(f"Serial write error: {e}")

def detect_humans(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    human_detected = False
    
    # Draw detection area rectangle
    x1 = int(w * DETECTION_AREA['x_start'])
    y1 = int(h * DETECTION_AREA['y_start'])
    x2 = int(w * DETECTION_AREA['x_end'])
    y2 = int(h * DETECTION_AREA['y_end'])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "Detection Zone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = int(detections[0, 0, i, 1])
            
            if class_id == HUMAN_CLASS_ID:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                if is_in_detection_area((startX, startY, endX, endY), w, h):
                    human_detected = True
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"Person: {confidence * 100:.2f}%"
                    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (200, 200, 200), 1)
    
    # Send serial data based on detection
    send_serial_data(human_detected)
    
    return frame, human_detected

# Connect to webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    log_message("Error: Could not open webcam")
    exit()

log_message("System started. Monitoring for humans in detection zone...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            log_message("Error reading frame from webcam")
            time.sleep(1)
            continue
        
        frame = cv2.flip(frame, 1)
        processed_frame, human_detected = detect_humans(frame)
        
        # Display status text
        status_text = "DETECTED (1)" if human_detected else "NO HUMAN (0)"
        status_color = (0, 255, 0) if human_detected else (0, 0, 255)
        cv2.putText(processed_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow('Human Detection with Serial Output', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    log_message("System stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        send_serial_data(0)
        ser.close()
    log_message("System shutdown")