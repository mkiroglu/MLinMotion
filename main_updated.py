import cv2
from SimpleHRNet import SimpleHRNet
from misc.visualization import joints_dict
import os
import numpy as np

# Directory path where you want to save the images
SAVE_DIRECTORY = "/path/to/save/directory"

# Load model
model = SimpleHRNet(48, 17, "/path/to/pose_hrnet_w48_384x288.pth")

def initialize_tracker(frame, bbox):
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, tuple(bbox))
    return tracker

def extract_joints_from_image(joints):
    joint_names = joints_dict()["coco"]["keypoints"]
    extracted_joints = {}
    num_persons, num_joints, _ = joints.shape
    for p in range(num_persons):
        person_joints = {}
        for j in range(num_joints):
            x, y = joints[p, j, 0], joints[p, j, 1]
            joint_name = joint_names[j]
            person_joints[joint_name] = (x, y)
        extracted_joints[p] = person_joints
    return extracted_joints

def detect_and_initialize_trackers(frame, trackers, person_ids, unique_id_counter):
    joints = model.predict(frame)
    extracted_joints = extract_joints_from_image(joints)
    # Process extracted joints as needed

    for p in range(joints.shape[0]):
        x_min, y_min = np.min(joints[p, :, :2], axis=0)
        x_max, y_max = np.max(joints[p, :, :2], axis=0)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        already_tracked = any([tracker.get_position().contains(bbox) for tracker in trackers])
        if not already_tracked:
            tracker = initialize_tracker(frame, bbox)
            trackers.append(tracker)
            person_ids[tracker] = unique_id_counter
            unique_id_counter += 1

    return unique_id_counter

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    trackers = []
    person_ids = {}
    unique_id_counter = 0
    frame_index = 0
    detection_interval = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % detection_interval == 0:
            unique_id_counter = detect_and_initialize_trackers(frame, trackers, person_ids, unique_id_counter)

        for tracker in trackers:
            success, bbox = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                person_id = person_ids[tracker]
                cv2.putText(frame, f"ID: {person_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Captured Image', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
