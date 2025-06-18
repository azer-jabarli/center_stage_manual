import cv2
import numpy as np

# Load face detection model (DNN)
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Center Stage parameters
smoothing_factor = 0.8  # Higher = smoother movement
crop_scale = 0.7 # Scale for cropping
show_debug = True  # Display detection information

# Initial values
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
current_x, current_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate crop size based on current frame size
    crop_w = int(frame_w * crop_scale)
    crop_h = int(frame_h * crop_scale)

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    best_face = None
    max_area = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            (x1, y1, x2, y2) = box.astype("int")
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                best_face = (x1, y1, x2, y2)

    # Update target position if face detected
    if best_face is not None:
        x1, y1, x2, y2 = best_face
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2

        # Calculate target crop position
        target_x = face_center_x - crop_w // 2
        target_y = face_center_y - crop_h // 2

        # Apply smoothing
        current_x = int(current_x * smoothing_factor + target_x * (1 - smoothing_factor))
        current_y = int(current_y * smoothing_factor + target_y * (1 - smoothing_factor))

        # Clamp values to valid range
        current_x = max(0, min(current_x, frame_w - crop_w))
        current_y = max(0, min(current_y, frame_h - crop_h))

        if show_debug:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)

    # Apply crop
    cropped = frame[current_y:current_y + crop_h, current_x:current_x + crop_w]

    # Resize to original dimensions if needed
    final_output = cv2.resize(cropped, (frame_w, frame_h))

    # Display result
    cv2.imshow("Center Stage", final_output)

    if show_debug:
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame,
                      (current_x, current_y),
                      (current_x + crop_w, current_y + crop_h),
                      (255, 0, 0), 2)
        cv2.imshow("Debug View", debug_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()