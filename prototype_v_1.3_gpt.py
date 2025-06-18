import cv2

def main():
    # 1. Open the default webcam (0 = first camera)
    cap = cv2.VideoCapture(0)   # on Linux/mac just use cv2.VideoCapture(0)

    # 2. Load the built-in Haar Cascade model for frontal faces
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise IOError("Could not load Haar cascade at " + cascade_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to grab frame")
            break

        # 3. Convert to grayscale (face detector works on gray images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4. Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,     # how much the image size is reduced at each scale
            minNeighbors=5,      # how many neighbors each rectangle should have to be retained
            minSize=(60, 60)     # ignore tiny detections
        )

        # 5. Draw rectangle and dot for every detected face
        for (x, y, w, h) in faces:
            # green rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # center of the rectangle
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(frame, (cx, cy), radius=4, color=(0, 0, 255), thickness=-1)  # filled red dot

        # 6. Display the annotated frame
        cv2.imshow("Face Detector – press 'q' to quit", frame)

        # 7. Quit when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
