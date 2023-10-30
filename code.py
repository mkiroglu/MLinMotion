import cv2
import numpy as np
import mediapipe as mp
import threading

# Load the SSD pre-trained model and configuration
prototxt_path = "deploy.prototxt"
caffemodel_path = "mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def capture_images(camera_index, interval):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    try:
        count = 0
        while True:
            # Capture a frame
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame from camera {camera_index}.")
                break

            # Prepare the frame for detection
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            # Detect objects in the frame
            net.setInput(blob)
            detections = net.forward()

            person_count = 0
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Assuming class 15 is for "person" in the MobileNet SSD model
                if confidence > 0.2 and int(detections[0, 0, i, 1]) == 15:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    cropped = frame[startY:endY, startX:endX]

                    # Pose estimation using mediapipe on the cropped person
                    pose_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(pose_image)

                    if pose_results.pose_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(cropped, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    person_count += 1
                    filename = f"cam_{camera_index}_capture_{count}_person_{person_count}.jpg"
                    cv2.imwrite(filename, cropped)
                    print(f"Captured {filename}")

                    # Show the captured frame with pose overlay in a window 
                    cv2.imshow(f"Camera {camera_index} - Captured Person {person_count}", cropped)

            count += 1

            # Wait for the specified interval or for 'q' key press
            key = cv2.waitKey(interval * 1000)
            if key == ord('q'):
                break
    finally:
        # Release the camera and close OpenCV windows related to this camera
        cap.release()
        cv2.destroyWindow(f"Camera {camera_index} - Captured Image")

# Specify the indices of the cameras you want to use
camera_indices = [0, 1, 2]

threads = []
for index in camera_indices:
    thread = threading.Thread(target=capture_images, args=(index, 1))
    thread.start()
    threads.append(thread)

# Join all threads
for thread in threads:
    thread.join()