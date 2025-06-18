import cv2
import numpy as np

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start camera
cap = cv2.VideoCapture(0)

# Zoom parameters
current_zoom = 1.0
min_zoom = 0.5
max_zoom = 2.0
zoom_speed = 0.05  # Lower = slower zoom (0.01-0.1 recommended)
saved_position = None  # Stores last known good position

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # Get largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        # Calculate target zoom (face should be ~30% of frame width)
        target_zoom = min(max_zoom, max(min_zoom, 0.3 * w / fw))

        # Calculate center point
        center_x = x + fw // 2
        center_y = y + fh // 2

        # Save current position
        saved_position = (center_x, center_y, target_zoom)
    else:
        # Use saved position if no face detected
        if saved_position:
            center_x, center_y, target_zoom = saved_position
        else:
            # Default to center if no saved position
            center_x, center_y, target_zoom = w // 2, h // 2, min_zoom

    # Very slow zoom adjustment
    if current_zoom < target_zoom:
        current_zoom = min(current_zoom + zoom_speed, target_zoom)
    elif current_zoom > target_zoom:
        current_zoom = max(current_zoom - zoom_speed, target_zoom)

    # Calculate crop area (centered on face/last position)
    crop_w = int(w / current_zoom)
    crop_h = int(h / current_zoom)

    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    # Adjust if near edges
    if x2 - x1 < crop_w:
        if x1 == 0:
            x2 = min(w, x1 + crop_w)
        else:
            x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        if y1 == 0:
            y2 = min(h, y1 + crop_h)
        else:
            y1 = max(0, y2 - crop_h)

    # Crop and resize
    cropped = frame[y1:y2, x1:x2]
    if cropped.size > 0:  # Only process if we have valid crop
        frame = cv2.resize(cropped, (w, h))

        # Draw visual feedback if face detected
        if len(faces) > 0:
            new_x = int((x - x1) * (w / crop_w))
            new_y = int((y - y1) * (h / crop_h))
            new_w = int(fw * (w / crop_w))
            new_h = int(fh * (h / crop_h))
            cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
        cv2.circle(frame, (w // 2, h // 2), 5, (0, 0, 255), -1)

    cv2.imshow('Face Tracking Zoom', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()