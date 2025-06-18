import cv2
import time


def detect_faces():
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the default camera
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced height
    cap.set(cv2.CAP_PROP_FPS, 60)  # Try to set high FPS mode

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Variables for FPS calculation
    prev_time = 0
    fps = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Mirror the frame (horizontal flip)
        frame = cv2.flip(frame, 1)

        # Apply bilateral filter for smoothing while preserving edges
        smoothed_frame = cv2.bilateralFilter(frame, 9, 75, 75)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(smoothed_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Slightly more sensitive scaling
            minNeighbors=6,  # Fewer false positives
            minSize=(50, 50),  # Larger minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw rectangle and point for each face
        for (x, y, w, h) in faces:
            # Draw semi-transparent green rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate center point for the red dot
            center_x = x + w // 2
            center_y = y + h // 2

            # Draw red point at the center
            cv2.circle(overlay, (center_x, center_y), 4, (0, 0, 255), -1)

            # Add the overlay with transparency
            alpha = 0.7  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection (Press Q to quit)', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_faces()