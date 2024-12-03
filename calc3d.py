import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Intrinsic parameters from camera calibration
K = np.array([[2.60452056e+03, 0.00000000e+00, 1.62104331e+03],
              [0.00000000e+00, 2.60335521e+03, 1.22438418e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([0.20669488, -0.62582447, -0.00114616, -0.00350343, 0.5119328])

# Example corresponding points from two views
points1 = np.array([
    [100, 150],
    [200, 250],
    [300, 350],
    [400, 450],
    [500, 550],
    [600, 650]
], dtype=np.float32)

points2 = np.array([
    [110, 160],
    [210, 260],
    [310, 360],
    [410, 460],
    [510, 560],
    [610, 660]
], dtype=np.float32)

# Find the essential matrix
E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

if E is None:
    print("Error: Essential matrix could not be computed.")
    exit()

# Decompose the essential matrix to get the rotation and translation
_, R, T, _ = cv2.recoverPose(E, points1, points2, K)

# Define the projection matrices for the two camera positions
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, T))

# Calculate 3D coordinates using cv2.triangulatePoints
def calculate_3d_coordinates(points1, points2, P1, P2):
    points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous to 3D coordinates
    return points_3d.T

coordinates_3d = calculate_3d_coordinates(points1, points2, P1, P2)

# Ensure the directory exists
home_directory = os.path.expanduser("~")
working_directory = os.path.join(home_directory, "projects", "led_mapper")
os.makedirs(working_directory, exist_ok=True)

# Save the 3D coordinates to a CSV file
csv_filename = os.path.join(working_directory, "led_3d_coordinates.csv")
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["LED Index", "X Position", "Y Position", "Z Position"])
    for i, coord in enumerate(coordinates_3d):
        csv_writer.writerow([i, coord[0], coord[1], coord[2]])

print(f"Saved 3D LED positions to CSV file: {csv_filename}")

# Visualize the 3D points and save the plot as an image file
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coordinates_3d[:, 0], coordinates_3d[:, 1], coordinates_3d[:, 2], c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Save the plot as an image file
plot_filename = os.path.join(working_directory, "led_3d_plot.png")
plt.savefig(plot_filename)
print(f"Saved 3D plot to file: {plot_filename}")