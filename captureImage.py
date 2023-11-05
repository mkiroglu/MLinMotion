import cv2
import time

def capture_image():
    # Open the webcam (0 is the default camera, change the number if you have multiple cameras)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    count = 0
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # If the frame was captured successfully
        if ret:
            # Save the frame to a file
            filename = f"captured_image_{count}.jpg"
            cv2.imwrite(filename, frame)

            # Display the frame
            cv2.imshow('Captured Image', frame)

            # Wait for 1 second or until 'q' is pressed
            key = cv2.waitKey(1000)  # waits for 1 second
            if key == ord('q'):
                break

            count += 1

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
