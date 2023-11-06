import cv2
import numpy as np
from SimpleHRNet import SimpleHRNet
import os
import matplotlib.pyplot as plt

# Load model (assuming you have already set the correct path to the weights)
model = SimpleHRNet(48, 17, "/path/to/pose_hrnet_w48_384x288.pth")

# Initialize an empty list to keep track of people (tracks)
tracks = []

# Define the tracking functions here (update_tracks and euclidean_distance from the previous code)

# Function to capture and process images
def capture_and_process_images():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    count = 0
    SAVE_DIRECTORY = "/Users/mertryankiroglu/Desktop/simple-HRNet/session/photos"
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            filename = os.path.join(SAVE_DIRECTORY, f"captured_image_{count}.jpg")
            cv2.imwrite(filename, frame)

            # Here we use the HRNet model to predict joints on the fly and update tracks
            detected_joints = model.predict(frame)
            update_tracks(detected_joints, tracks)

            cv2.imshow('Captured Image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Plot the tracked movements
def plot_tracks(tracks):
    plt.figure(figsize=(10, 10))

    for track in tracks:
        track_history = np.array([joints[:, :2] for joints in track['track_history']])
        for joint_track in track_history.transpose(1, 0, 2):
            plt.plot(joint_track[:, 0], joint_track[:, 1], marker='o')

    plt.gca().invert_yaxis()  # Invert the y-axis to match image coordinates
    plt.show()

if __name__ == "__main__":
    capture_and_process_images()
    plot_tracks(tracks)