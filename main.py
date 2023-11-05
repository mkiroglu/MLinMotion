import cv2
from SimpleHRNet import SimpleHRNet
from misc.visualization import joints_dict
import os

# Directory path where you want to save the images
SAVE_DIRECTORY = "/Users/mertryankiroglu/Desktop/simple-HRNet/session/photos"

# Load model
model = SimpleHRNet(48, 17, "/Users/mertryankiroglu/Desktop/simple-HRNet/models_/detectors/yolo/weights/pose_hrnet_w48_384x288.pth")

def extract_joints_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Predict joints
    joints = model.predict(image)

    # Extract and print the coordinates of every joint for every person
    num_persons, num_joints, _ = joints.shape
    joint_names = joints_dict()["coco"]["keypoints"]
    for p in range(num_persons):
        print(f"Image: {image_path} - Person {p + 1}:")
        for j in range(num_joints):
            x, y = joints[p, j, 0], joints[p, j, 1]
            joint_name = joint_names[j]
            print(f"{joint_name}: (x={y:.2f}, y={x:.2f})")
        print("-----")

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
            # Save the frame to the specified directory
            filename = os.path.join(SAVE_DIRECTORY, f"captured_image_{count}.jpg")
            cv2.imwrite(filename, frame)

            # Extract joints from the saved image
            extract_joints_from_image(filename)

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
