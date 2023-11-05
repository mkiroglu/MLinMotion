import cv2
import matplotlib.pyplot as plt
from SimpleHRNet import SimpleHRNet
from misc.visualization import joints_dict

# Load model
model = SimpleHRNet(48, 17, "/Users/mertryankiroglu/Desktop/simple-HRNet/models_/detectors/yolo/weights/pose_hrnet_w48_384x288.pth")

# Read image
image_path = "capture.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

#width and height of the image
height, width, channels = image.shape
print(f"Image Width: {width}, Image Height: {height}")

# Predict joints
joints = model.predict(image)

# Extract and print the coordinates of every joint for every person
num_persons, num_joints, _ = joints.shape
joint_names = joints_dict()["coco"]["keypoints"]
for p in range(num_persons):
    print(f"Person {p + 1}:")
    for j in range(num_joints):
        x, y = joints[p, j, 0], joints[p, j, 1]  # Correctly accessing the coordinates
        joint_name = joint_names[j]
        print(f"{joint_name}: (x={y:.2f}, y={x:.2f})")
    print("-----")

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