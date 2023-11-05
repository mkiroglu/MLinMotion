import cv2
import matplotlib.pyplot as plt
import os
from SimpleHRNet import SimpleHRNet
from misc.visualization import joints_dict

# Load model
model = SimpleHRNet(48, 17, "/Users/mertryankiroglu/Desktop/simple-HRNet/models_/detectors/yolo/weights/pose_hrnet_w48_384x288.pth")

# Directory containing the images
image_directory = "/Users/mertryankiroglu/Desktop/simple-HRNet/session/photos"

# List all files in the directory
all_files = os.listdir(image_directory)
# Filter out the image files (assuming they have extensions .jpg, .jpeg, .png)
image_files = [f for f in all_files if f.lower().endswith(('jpg', 'jpeg', 'png'))]

for image_file in image_files:
    # Read image
    image_path = os.path.join(image_directory, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Width and height of the image
    height, width, channels = image.shape
    print(f"Image: {image_file}, Width: {width}, Height: {height}")

    # Predict joints
    joints = model.predict(image)

    def plot_joints(ax, output):
        bones = joints_dict()["coco"]["skeleton"]
        for bone in bones:
            xS = [output[:, bone[0], 1], output[:, bone[1], 1]]
            yS = [output[:, bone[0], 0], output[:, bone[1], 0]]
            ax.plot(xS, yS, linewidth=3, c=(0, 0.3, 0.7))
        ax.scatter(joints[:, :, 1], joints[:, :, 0], s=20, c='r')

    # Plotting
    fig = plt.figure(figsize=(30/2.54, 15/2.54))
    ax = fig.add_subplot(121)
    ax.imshow(image_rgb)
    ax = fig.add_subplot(122)
    ax.imshow(image_rgb)
    plot_joints(ax, joints)
    plt.show()