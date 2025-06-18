import cv2
import numpy as np

# Load face detection model
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Tracking parameters
SMOOTHING = 0.85
ZOOM_SMOOTH = 0.88
MIN_ZOOM = 0.4
MAX_ZOOM = 1.0
TARGET_RATIO = 0.3
EDGE_MARGIN = 0.15  # 15% of frame border for zoom adjustment

# Frame dimensions
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tracking state
current_zoom = MIN_ZOOM
current_pan = (0, 0)


def calculate_crop(face_center, frame_size, zoom):
    fw, fh = frame_size
    cw = int(fw * zoom)
    ch = int(fh * zoom)

    tx = face_center[0] - cw // 2
    ty = face_center[1] - ch // 2

    # Constrain to frame boundaries
    tx = max(0, min(tx, fw - cw))
    ty = max(0, min(ty, fh - ch))

    return (tx, ty, cw, ch)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
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

        # Calculate base zoom from face size
        base_zoom = (frame_w * TARGET_RATIO) / face_size

        # Calculate edge proximity zoom adjustment
        h_dist = min(face_center[0], frame_w - face_center[0]) / (frame_w * EDGE_MARGIN)
        v_dist = min(face_center[1], frame_h - face_center[1]) / (frame_h * EDGE_MARGIN)
        edge_zoom = min(h_dist, v_dist)
        edge_zoom = np.clip(edge_zoom, 0.5, 1.0)

        # Combine zoom factors
        target_zoom = base_zoom * edge_zoom
        target_zoom = np.clip(target_zoom, MIN_ZOOM, MAX_ZOOM)

        # Smooth zoom transition
        current_zoom = current_zoom * ZOOM_SMOOTH + target_zoom * (1 - ZOOM_SMOOTH)

        # Calculate crop parameters
        tx, ty, cw, ch = calculate_crop(face_center, (frame_w, frame_h), current_zoom)

        # Smooth pan movement
        current_pan = (
            current_pan[0] * SMOOTHING + tx * (1 - SMOOTHING),
            current_pan[1] * SMOOTHING + ty * (1 - SMOOTHING)
        )
    else:
        # No face detected - reset to center
        current_zoom = MIN_ZOOM
        current_pan = (
            current_pan[0] * SMOOTHING,
            current_pan[1] * SMOOTHING
        )

    # Apply final crop with boundary checks
    cw = int(frame_w * current_zoom)
    ch = int(frame_h * current_zoom)
    pan_x = int(np.clip(current_pan[0], 0, frame_w - cw))
    pan_y = int(np.clip(current_pan[1], 0, frame_h - ch))

    cropped = frame[pan_y:pan_y + ch, pan_x:pan_x + cw]
    final = cv2.resize(cropped, (frame_w, frame_h))

    # Draw tracking elements if face detected
    if best_face:
        # Calculate final coordinates for visualization
        fx1 = int((x1 - pan_x) * (frame_w / cw))
        fy1 = int((y1 - pan_y) * (frame_h / ch))
        fx2 = int((x2 - pan_x) * (frame_w / cw))
        fy2 = int((y2 - pan_y) * (frame_h / ch))

        # Draw green rectangle
        cv2.rectangle(final, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)

        # Draw red center dot
        center_x = (fx1 + fx2) // 2
        center_y = (fy1 + fy2) // 2
        cv2.circle(final, (center_x, center_y), 8, (0, 0, 255), -1)

    cv2.imshow("Smart Camera Tracking", final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()