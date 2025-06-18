import cv2
import numpy as np

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start camera
cap = cv2.VideoCapture(0)

# Target face width (as fraction of frame width)
TARGET_FACE_WIDTH = 0.4
SMOOTHING = 0.8  # Smoothing factor for zoom/pan (0-1, higher = smoother)

# Initialize variables
current_zoom = 1.0
current_pan_x = 0
current_pan_y = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2])

        # Calculate target zoom to make face width consistent
        target_zoom = (width * TARGET_FACE_WIDTH) / w
        target_zoom = np.clip(target_zoom, 0.5, 2.0)  # Limit zoom range

        # Smooth zoom transition
        current_zoom = current_zoom * SMOOTHING + target_zoom * (1 - SMOOTHING)

        # Calculate center point
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Calculate target pan to center face
        target_pan_x = face_center_x - (width // 2) / current_zoom
        target_pan_y = face_center_y - (height // 2) / current_zoom

        # Smooth pan transition
        current_pan_x = current_pan_x * SMOOTHING + target_pan_x * (1 - SMOOTHING)
        current_pan_y = current_pan_y * SMOOTHING + target_pan_y * (1 - SMOOTHING)

        # Apply zoom and pan
        M = np.float32([[current_zoom, 0, -current_pan_x],
                        [0, current_zoom, -current_pan_y]])
        zoomed = cv2.warpAffine(frame, M, (width, height))

        # Draw rectangle and center dot on zoomed frame
        new_x = int((x - current_pan_x) * current_zoom)
        new_y = int((y - current_pan_y) * current_zoom)
        new_w = int(w * current_zoom)
        new_h = int(h * current_zoom)

        cv2.rectangle(zoomed, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
        cv2.circle(zoomed, (new_x + new_w // 2, new_y + new_h // 2), 5, (0, 0, 255), -1)

        output = zoomed
    else:
        # No face detected - show normal view
        output = frame

    cv2.imshow('Smart Zoom Face Tracking', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()