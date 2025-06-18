import cv2
import numpy as np
import time

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start camera
cap = cv2.VideoCapture(0)

# Tracking parameters
current_zoom = 1.0
min_zoom = 0.5
max_zoom = 2.0
zoom_speed = 0.01
update_interval = 2  # seconds
last_update = 0
saved_position = None  # (center_x, center_y, target_zoom)


def get_face_position(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        center_x = x + w // 2
        center_y = y + h // 2
        target_zoom = min(max_zoom, max(min_zoom, 0.3 * frame.shape[1] / w))
        return (center_x, center_y, target_zoom)
    return None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Update face position every 3 seconds
    current_time = time.time()
    if current_time - last_update > update_interval:
        new_position = get_face_position(frame)
        if new_position:
            saved_position = new_position
        last_update = current_time

    # Use saved position or default
    if saved_position:
        center_x, center_y, target_zoom = saved_position
    else:
        center_x, center_y, target_zoom = w // 2, h // 2, min_zoom

    # Smooth zoom adjustment
    if current_zoom < target_zoom:
        current_zoom = min(current_zoom + zoom_speed, target_zoom)
    elif current_zoom > target_zoom:
        current_zoom = max(current_zoom - zoom_speed, target_zoom)

    # Calculate and apply crop
    crop_w = int(w / current_zoom)
    crop_h = int(h / current_zoom)
    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    # Adjust crop if near edges
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w) if x1 > 0 else 0
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h) if y1 > 0 else 0

    # Apply zoom
    cropped = frame[y1:y2, x1:x2]
    if cropped.size > 0:
        frame = cv2.resize(cropped, (w, h))

    # Visual feedback
    cv2.circle(frame, (w // 2, h // 2), 5, (0, 0, 255), -1)
    if saved_position and (time.time() - last_update < 1.0):  # Show rectangle for 1s after detection
        cv2.rectangle(frame, (w // 2 - 50, h // 2 - 50), (w // 2 + 50, h // 2 + 50), (0, 255, 0), 2)

    # Display update timer
    timer_text = f"Next update in: {max(0, update_interval - (current_time - last_update)):.1f}s"
    cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('3-Second Face Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()