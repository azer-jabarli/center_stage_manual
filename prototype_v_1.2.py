import cv2
import numpy as np

# Load face detection model
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Camera setup (OpenCV may need additional macOS permissions)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera access denied. Check permissions and NSCameraUsageDescription in Info.plist")
    exit()

# Tracking parameters
SMOOTHING_FACTOR = 0.85
ZOOM_SMOOTHING = 0.88
MIN_ZOOM = 0.4
MAX_ZOOM = 1.0
TARGET_FACE_RATIO = 0.35
PAN_THRESHOLD = 0.1  # Percentage from center to trigger pan

# State variables
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
current_zoom = MIN_ZOOM
current_pan = (0, 0)


def calculate_adjusted_region(frame_size, face_center, current_zoom):
    frame_w, frame_h = frame_size
    crop_w = int(frame_w * current_zoom)
    crop_h = int(frame_h * current_zoom)

    target_x = face_center[0] - crop_w // 2
    target_y = face_center[1] - crop_h // 2

    # Apply boundaries
    target_x = max(0, min(target_x, frame_w - crop_w))
    target_y = max(0, min(target_y, frame_h - crop_h))

    return target_x, target_y, crop_w, crop_h


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror and convert
    frame = cv2.flip(frame, 1)
    debug_frame = frame.copy()

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    best_face = None
    max_area = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            x1, y1, x2, y2 = box.astype("int")
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_face = (x1, y1, x2, y2)

    if best_face:
        x1, y1, x2, y2 = best_face
        face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        face_size = max(x2 - x1, y2 - y1)

        # Dynamic zoom calculation
        target_zoom = (frame_w * TARGET_FACE_RATIO) / face_size
        target_zoom = np.clip(target_zoom, MIN_ZOOM, MAX_ZOOM)
        current_zoom = current_zoom * ZOOM_SMOOTHING + target_zoom * (1 - ZOOM_SMOOTHING)

        # Calculate crop region
        t_x, t_y, c_w, c_h = calculate_adjusted_region(
            (frame_w, frame_h),
            face_center,
            current_zoom
        )

        # Smooth panning
        current_pan = (
            current_pan[0] * SMOOTHING_FACTOR + t_x * (1 - SMOOTHING_FACTOR),
            current_pan[1] * SMOOTHING_FACTOR + t_y * (1 - SMOOTHING_FACTOR)
        )
    else:
        # Reset to center when no face detected
        current_zoom = MIN_ZOOM
        current_pan = (
            current_pan[0] * SMOOTHING_FACTOR,
            current_pan[1] * SMOOTHING_FACTOR
        )

    # Apply final crop
    c_w = int(frame_w * current_zoom)
    c_h = int(frame_h * current_zoom)
    pan_x = int(np.clip(current_pan[0], 0, frame_w - c_w))
    pan_y = int(np.clip(current_pan[1], 0, frame_h - c_h))

    cropped = frame[pan_y:pan_y + c_h, pan_x:pan_x + c_w]
    final = cv2.resize(cropped, (frame_w, frame_h))

    # Add center indicator
    cv2.circle(final, (frame_w // 2, frame_h // 2), 10, (0, 0, 255), -1)

    cv2.imshow("Center Stage", final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()