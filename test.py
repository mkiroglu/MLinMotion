import numpy as np
import cv2
from SimpleHRNet import SimpleHRNet
import os

# Load model
model_path = "/path/to/pose_hrnet_w48_384x288.pth"
model = SimpleHRNet(48, 17, model_path)

# Initialize an empty list to keep track of people
tracks = []

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def update_tracks(detected_joints, tracks, threshold=50):
    # If no tracks are present, add all detections as new tracks
    if not tracks:
        for joints in detected_joints:
            tracks.append({'joints': joints, 'track_history': [joints]})
        return

    for joints in detected_joints:
        # Find the closest track to current detected joints
        distances = [euclidean_distance(joints[:, :2], track['joints'][:, :2]) for track in tracks]
        closest_track_idx = np.argmin(distances)

        # If the closest track is within the threshold, update the track with new joints
        if distances[closest_track_idx] < threshold:
            tracks[closest_track_idx]['joints'] = joints
            tracks[closest_track_idx]['track_history'].append(joints)
        else:
            # If no track is close enough, start a new track
            tracks.append({'joints': joints, 'track_history': [joints]})

# Function to process and update tracks with new image frames
def process_frame(frame):
    # Predict joints using HRNet model
    detected_joints = model.predict(frame)
    
    # Update the tracks with the detected joints
    update_tracks(detected_joints, tracks)

# Code for capturing frames and processing them
# ... (your image capturing code)

# After capturing is done, you can plot the track histories
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

for track in tracks:
    track_history = np.array([joints[:, :2] for joints in track['track_history']])
    for joint_track in track_history.transpose(1, 0, 2):
        plt.plot(joint_track[:, 0], joint_track[:, 1], marker='o')

plt.gca().invert_yaxis()  # Invert the y-axis to match image coordinates
plt.show()