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
smoothing_factor = 0.8  # Position smoothing
smoothing_factor_zoom = 0.7  # Zoom smoothing
initial_crop_scale = 0.7
desired_face_ratio = 0.3  # Target face width/height ratio
min_crop_scale = 0.3
max_crop_scale = 1.0
show_debug = True

# Initialize variables
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
current_x, current_y = 0, 0
crop_scale = initial_crop_scale

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)

    # Calculate crop size based on current scale
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

    # Update position and zoom if face detected
    if best_face is not None:
        x1, y1, x2, y2 = best_face
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2

        # Calculate target position
        target_x = face_center_x - crop_w // 2
        target_y = face_center_y - crop_h // 2

        # Smooth position
        current_x = int(current_x * smoothing_factor + target_x * (1 - smoothing_factor))
        current_y = int(current_y * smoothing_factor + target_y * (1 - smoothing_factor))

        # Calculate dynamic zoom
        face_width = x2 - x1
        face_height = y2 - y1
        desired_face_width = desired_face_ratio * frame_w
        desired_face_height = desired_face_ratio * frame_h

        # Calculate new crop scale based on face size
        scale_width = face_width / desired_face_width
        scale_height = face_height / desired_face_height
        new_crop_scale = min(scale_width, scale_height)

        # Clamp and smooth zoom
        new_crop_scale = np.clip(new_crop_scale, min_crop_scale, max_crop_scale)
        crop_scale = (crop_scale * smoothing_factor_zoom +
                      new_crop_scale * (1 - smoothing_factor_zoom))
        crop_scale = np.clip(crop_scale, min_crop_scale, max_crop_scale)

        # Update crop size
        crop_w = int(frame_w * crop_scale)
        crop_h = int(frame_h * crop_scale)

        # Clamp position to valid bounds
        current_x = max(0, min(current_x, frame_w - crop_w))
        current_y = max(0, min(current_y, frame_h - crop_h))

        if show_debug:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)

    # Apply crop and resize
    cropped = frame[current_y:current_y + crop_h, current_x:current_x + crop_w]
    final_output = cv2.resize(cropped, (frame_w, frame_h))

    # Add red center dot
    center_x = final_output.shape[1] // 2
    center_y = final_output.shape[0] // 2
    cv2.circle(final_output, (center_x, center_y), 8, (0, 0, 255), -1)

    # Display output
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