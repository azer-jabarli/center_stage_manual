import cv2
import numpy as np
import time

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start camera
cap = cv2.VideoCapture(0)

# Tracking parameters
current_zoom = 1.0
min_zoom = 0.4  # Minimum zoom level
max_zoom = 3.0  # Maximum zoom level
zoom_speed = 0.01  # Very slow zoom adjustment
update_interval = 2  # Face position updates every 3 seconds
face_target_ratio = 0.2  # Face should be 20% of frame width
last_update = 0
saved_position = None  # (center_x, center_y, target_zoom)


def get_face_position(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        center_x = x + w // 2
        center_y = y + h // 2
        # Calculate zoom needed to make face 20% of frame width
        target_zoom = min(max_zoom, max(min_zoom, (face_target_ratio * frame.shape[1]) / w))
        return (center_x, center_y, target_zoom)
    return None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    current_time = time.time()

    # Update face position every 3 seconds
    if current_time - last_update > update_interval:
        new_position = get_face_position(frame)
        if new_position:
            saved_position = new_position
            # Visual confirmation of detection
            cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 2)
        last_update = current_time

    # Use saved position or default view
    if saved_position:
        center_x, center_y, target_zoom = saved_position
    else:
        center_x, center_y, target_zoom = w // 2, h // 2, min_zoom

    # Extremely gradual zoom adjustment
    if abs(current_zoom - target_zoom) > 0.01:
        if current_zoom < target_zoom:
            current_zoom = min(current_zoom + zoom_speed, target_zoom)
        else:
            current_zoom = max(current_zoom - zoom_speed, target_zoom)

    # Calculate crop area (centered on face)
    crop_w = int(w / current_zoom)
    crop_h = int(h / current_zoom)
    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    # Ensure crop dimensions are valid
    if x2 <= x1: x2 = x1 + 1
    if y2 <= y1: y2 = y1 + 1

    # Apply zoom if valid
    if x2 > x1 and y2 > y1:
        cropped = frame[y1:y2, x1:x2]
        if cropped.size > 0:
            frame = cv2.resize(cropped, (w, h))

    # Visual feedback
    cv2.circle(frame, (w // 2, h // 2), 5, (0, 0, 255), -1)  # Red center dot

    # Display status
    timer_text = f"Update in: {max(0, update_interval - (current_time - last_update)):.1f}s"
    zoom_text = f"Zoom: {current_zoom:.2f}x"
    ratio_text = f"Face: {face_target_ratio * 100:.0f}% width"
    cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, zoom_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, ratio_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Face Zoom (20% Width)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()