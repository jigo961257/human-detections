import cv2
import time
import numpy as np
from datetime import datetime
import serial

# Configuration
WEBCAM_INDEX = 0
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"
CONFIDENCE_THRESHOLD = 0.5
HUMAN_CLASS_ID = 15

# Serial port configuration
SERIAL_PORT = '/dev/cu.usbserial-120'  # Change to your Arduino port
BAUD_RATE = 9600

# Initial Detection area configuration (percentage of frame width/height)
DETECTION_AREA = {
    'x_start': 0.25,
    'y_start': 0.25,
    'x_end': 0.75,
    'y_end': 0.75
}

# UI Configuration
BUTTONS = {
    'reset_zone': {'text': "Reset Zone", 'pos': (10, 70), 'size': (120, 30), 'color': (100, 100, 255)},
    'disable_zone': {'text': "Disable Zone", 'pos': (10, 110), 'size': (120, 30), 'color': (100, 255, 100)},
    'enable_zone': {'text': "Enable Zone", 'pos': (10, 150), 'size': (120, 30), 'color': (255, 100, 100)},
    'zoom_in': {'text': "+", 'pos': (500, 10), 'size': (50, 30), 'color': (200, 200, 200)},
    'zoom_out': {'text': "-", 'pos': (560, 10), 'size': (50, 30), 'color': (200, 200, 200)}
}

# Global variables
drawing = False
zone_enabled = True
last_human_state = None
human_count = 0
start_point = (-1, -1)
end_point = (-1, -1)
zoom_factor = 1.0
zoom_center = (320, 240)  # Default center for 640x480 frame

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

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def check_button_click(x, y):
    """Check if a button was clicked and perform the corresponding action"""
    global zone_enabled, DETECTION_AREA, zoom_factor, zoom_center
    
    for btn_name, btn in BUTTONS.items():
        pos = btn['pos']
        size = btn['size']
        
        if (pos[0] <= x <= pos[0] + size[0] and 
            pos[1] <= y <= pos[1] + size[1]):
            
            if btn_name == 'reset_zone':
                DETECTION_AREA = {
                    'x_start': 0.25,
                    'y_start': 0.25,
                    'x_end': 0.75,
                    'y_end': 0.75
                }
                log_message("Detection zone reset to default")
                
            elif btn_name == 'disable_zone':
                zone_enabled = False
                log_message("Detection zone disabled - monitoring entire frame")
                
            elif btn_name == 'enable_zone':
                zone_enabled = True
                log_message("Detection zone enabled")
                
            elif btn_name == 'zoom_in':
                zoom_factor = min(zoom_factor * 1.2, 3.0)  # Max zoom 3x
                log_message(f"Zoom in - current zoom: {zoom_factor:.1f}x")
                
            elif btn_name == 'zoom_out':
                zoom_factor = max(zoom_factor / 1.2, 1.0)  # Min zoom 1x
                log_message(f"Zoom out - current zoom: {zoom_factor:.1f}x")
                
            return True
    return False

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point, DETECTION_AREA, zoom_center, zoom_factor
    
    # Adjust coordinates for zoom
    adj_x = int(x / zoom_factor + zoom_center[0] * (1 - 1/zoom_factor))
    adj_y = int(y / zoom_factor + zoom_center[1] * (1 - 1/zoom_factor))
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if a button was clicked first
        if check_button_click(x, y):
            # If a button was clicked, we don't start drawing a new zone
            return

        # If not a button, and zone is enabled, then start drawing
        if zone_enabled:
            drawing = True
            start_point = (adj_x, adj_y)
            end_point = (adj_x, adj_y)
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (adj_x, adj_y)
        
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            end_point = (adj_x, adj_y)
            
            # Convert to relative coordinates if a zone was being drawn
            frame_height, frame_width = param.shape[:2]
            DETECTION_AREA = {
                'x_start': min(start_point[0], end_point[0]) / frame_width,
                'y_start': min(start_point[1], end_point[1]) / frame_height,
                'x_end': max(start_point[0], end_point[0]) / frame_width,
                'y_end': max(start_point[1], end_point[1]) / frame_height
            }
        # After drawing or if not drawing, also check for button click on release
        else:
            check_button_click(x, y)
    
    elif event == cv2.EVENT_MOUSEWHEEL:
        # Handle mouse wheel for zoom (alternative to buttons)
        if flags > 0:  # Scroll up
            zoom_factor = min(zoom_factor * 1.1, 3.0)
        else:  # Scroll down
            zoom_factor = max(zoom_factor / 1.1, 1.0)
        log_message(f"Mouse wheel zoom - current zoom: {zoom_factor:.1f}x")

def apply_zoom(frame):
    """Apply zoom transformation to the frame"""
    if zoom_factor == 1.0:
        return frame.copy()
    
    h, w = frame.shape[:2]
    
    # Calculate the zoomed region
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    
    x1 = max(0, zoom_center[0] - new_w // 2)
    y1 = max(0, zoom_center[1] - new_h // 2)
    x2 = min(w, x1 + new_w)
    y2 = min(h, y1 + new_h)
    
    # Adjust if we're at the edges
    if x2 - x1 < new_w:
        x1 = max(0, x2 - new_w)
    if y2 - y1 < new_h:
        y1 = max(0, y2 - new_h)
    
    # Crop and resize
    zoomed = frame[y1:y2, x1:x2]
    return cv2.resize(zoomed, (w, h))

def is_in_detection_area(box, frame_width, frame_height):
    """Check if detection is within our defined area"""
    if not zone_enabled:
        return True
        
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
            log_message(f"Serial data sent: {data.decode()} (Humans detected: {human_count})")
            last_human_state = state
        except serial.SerialException as e:
            log_message(f"Serial write error: {e}")

def draw_buttons(frame):
    """Draw interactive buttons on the frame"""
    for btn_name, btn in BUTTONS.items():
        pos = btn['pos']
        size = btn['size']
        color = btn['color']
        
        # Draw button rectangle
        cv2.rectangle(frame, pos, (pos[0] + size[0], pos[1] + size[1]), color, -1)
        cv2.rectangle(frame, pos, (pos[0] + size[0], pos[1] + size[1]), (0, 0, 0), 1)
        
        # Calculate text position to center it
        text_size = cv2.getTextSize(btn['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = pos[0] + (size[0] - text_size[0]) // 2
        text_y = pos[1] + (size[1] + text_size[1]) // 2
        
        # Draw button text
        cv2.putText(frame, btn['text'], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def detect_humans(frame):
    global human_count, zoom_factor, zoom_center
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    human_count = 0
    
    # Draw detection area rectangle if enabled
    if zone_enabled:
        x1 = int(w * DETECTION_AREA['x_start'])
        y1 = int(h * DETECTION_AREA['y_start'])
        x2 = int(w * DETECTION_AREA['x_end'])
        y2 = int(h * DETECTION_AREA['y_end'])
        
        # Draw the rectangle being defined
        if drawing:
            # Adjust drawing coordinates for zoom
            adj_x1 = int((start_point[0] - zoom_center[0] * (1 - 1/zoom_factor)) * zoom_factor)
            adj_y1 = int((start_point[1] - zoom_center[1] * (1 - 1/zoom_factor)) * zoom_factor)
            adj_x2 = int((end_point[0] - zoom_center[0] * (1 - 1/zoom_factor)) * zoom_factor)
            adj_y2 = int((end_point[1] - zoom_center[1] * (1 - 1/zoom_factor)) * zoom_factor)
            cv2.rectangle(frame, (adj_x1, adj_y1), (adj_x2, adj_y2), (0, 255, 255), 2)
        else:
            # Adjust zone coordinates for zoom
            adj_x1 = int((x1 - zoom_center[0] * (1 - 1/zoom_factor)) * zoom_factor)
            adj_y1 = int((y1 - zoom_center[1] * (1 - 1/zoom_factor)) * zoom_factor)
            adj_x2 = int((x2 - zoom_center[0] * (1 - 1/zoom_factor)) * zoom_factor)
            adj_y2 = int((y2 - zoom_center[1] * (1 - 1/zoom_factor)) * zoom_factor)
            cv2.rectangle(frame, (adj_x1, adj_y1), (adj_x2, adj_y2), (255, 0, 0), 2)
            cv2.putText(frame, "Detection Zone", (adj_x1, adj_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = int(detections[0, 0, i, 1])
            
            if class_id == HUMAN_CLASS_ID:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                if is_in_detection_area((startX, startY, endX, endY), w, h):
                    human_count += 1
                    # Adjust detection box coordinates for zoom
                    adj_startX = int((startX - zoom_center[0] * (1 - 1/zoom_factor)) * zoom_factor)
                    adj_startY = int((startY - zoom_center[1] * (1 - 1/zoom_factor)) * zoom_factor)
                    adj_endX = int((endX - zoom_center[0] * (1 - 1/zoom_factor)) * zoom_factor)
                    adj_endY = int((endY - zoom_center[1] * (1 - 1/zoom_factor)) * zoom_factor)
                    cv2.rectangle(frame, (adj_startX, adj_startY), (adj_endX, adj_endY), (0, 255, 0), 2)
                    label = f"Person: {confidence * 100:.2f}%"
                    cv2.putText(frame, label, (adj_startX, adj_startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Adjust detection box coordinates for zoom
                    adj_startX = int((startX - zoom_center[0] * (1 - 1/zoom_factor)) * zoom_factor)
                    adj_startY = int((startY - zoom_center[1] * (1 - 1/zoom_factor)) * zoom_factor)
                    adj_endX = int((endX - zoom_center[0] * (1 - 1/zoom_factor)) * zoom_factor)
                    adj_endY = int((endY - zoom_center[1] * (1 - 1/zoom_factor)) * zoom_factor)
                    cv2.rectangle(frame, (adj_startX, adj_startY), (adj_endX, adj_endY), (200, 200, 200), 1)
    
    # Send serial data based on detection (1 if more than 1 human detected)
    send_serial_data(human_count > 1)
    
    return frame

# Connect to webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    log_message("Error: Could not open webcam")
    exit()

# Create window and set mouse callback
cv2.namedWindow('Human Detection with Serial Output')
# Pass the initial frame to mouse_callback so it can get frame dimensions immediately
ret, initial_frame = cap.read()
if ret:
    cv2.setMouseCallback('Human Detection with Serial Output', mouse_callback, initial_frame)
else:
    log_message("Warning: Could not read initial frame for mouse callback setup. Mouse interaction might be affected.")

log_message("System started. Monitoring for humans...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            log_message("Error reading frame from webcam")
            time.sleep(1)
            continue
        
        frame = cv2.flip(frame, 1)
        
        # Apply zoom before processing
        zoomed_frame = apply_zoom(frame)
        
        # Detect humans and draw UI on zoomed frame
        processed_frame = detect_humans(zoomed_frame)
        
        # Display status text
        status_text = f"DETECTED: {human_count} (Sending {'1' if human_count > 1 else '0'})"
        status_color = (0, 255, 0) if human_count > 1 else (0, 0, 255)
        cv2.putText(processed_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display zone status
        zone_status = "ZONE: ENABLED" if zone_enabled else "ZONE: DISABLED"
        zone_color = (0, 255, 0) if zone_enabled else (0, 0, 255)
        cv2.putText(processed_frame, zone_status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)
        
        # Display zoom level
        zoom_text = f"Zoom: {zoom_factor:.1f}x"
        cv2.putText(processed_frame, zoom_text, (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw buttons
        draw_buttons(processed_frame)
        
        cv2.imshow('Human Detection with Serial Output', processed_frame)
        
        # Handle keyboard events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            zone_enabled = not zone_enabled
            log_message(f"Detection zone {'enabled' if zone_enabled else 'disabled'}")
        elif key == ord('+') or key == ord('='):
            zoom_factor = min(zoom_factor * 1.2, 3.0)
            log_message(f"Zoom in - current zoom: {zoom_factor:.1f}x")
        elif key == ord('-') or key == ord('_'):
            zoom_factor = max(zoom_factor / 1.2, 1.0)
            log_message(f"Zoom out - current zoom: {zoom_factor:.1f}x")
        elif key == ord('0'):
            zoom_factor = 1.0
            log_message("Zoom reset to 1.0x")

except KeyboardInterrupt:
    log_message("System stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        send_serial_data(0) # Ensure serial state is reset on exit
        ser.close()
    log_message("System shutdown")