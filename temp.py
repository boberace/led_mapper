import time
import board
import numpy as np
import neopixel_spi as neopixel
import cv2
from picamera2 import Picamera2, Preview
import csv
import os

home_directory = os.path.expanduser("~")
working_directory = os.path.join(home_directory, "projects", "led_mapper")
#sudo chmod 666 /dev/dma_heap/linux,cma

# LED configuration
LED_COUNT = 3  # Number of LEDs in your strip
PIXEL_ORDER = neopixel.GRB

# Initialize the LED strip
spi = board.SPI()
strip = neopixel.NeoPixel_SPI(spi, LED_COUNT, pixel_order=PIXEL_ORDER, auto_write=False)

# Camera setup

# intrinsic parameters from camera calibration
K = np.array([[2.60452056e+03, 0.00000000e+00, 1.62104331e+03],
              [0.00000000e+00, 2.60335521e+03, 1.22438418e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([0.20669488, -0.62582447, -0.00114616, -0.00350343, 0.5119328])

picam2 = Picamera2()
camera_modes = picam2.sensor_modes
native_resolution = max(camera_modes, key=lambda mode: mode["size"])["size"]  # Get the highest resolution mode
camera_config = picam2.create_still_configuration(main={"size": native_resolution})
picam2.configure(camera_config)
picam2.start()

def detect_led(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold to find bright spots
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assume the largest contour is the LED
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, (0, 0, 255), 10)
        return center

    return None


results = {}

def capture_images(angle):
    results = {}
    for i in range(LED_COUNT):
        # Turn off all LEDs
        strip.fill((0, 0, 0))
        strip.show() 

        # Light up the current LED
        strip[i] = (255, 255, 255)
        strip.show()
        time.sleep(0.1)  # Allow time for the camera to capture

        # Capture image
        frame = picam2.capture_array()

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ensure the directory exists
        ledimages_directory = os.path.join(working_directory, "ledimages")
        os.makedirs(ledimages_directory, exist_ok=True)


        coord = detect_led(frame_rgb)
        if coord:
            results[i] = coord
            print(f"LED {i}: {coord}")
        else:
            results[i] = None
            print(f"LED {i}: No LED detected")

        # Save the captured image
        filename = os.path.join(ledimages_directory, f"led_{i}_angle_{angle}.jpg")
        cv2.imwrite(filename, frame_rgb)
        print(f"Saved image: {filename}")

    return results

# Capture images from the first angle
results_angle_1 = capture_images(angle=1)

# Prompt the user to move the camera to a different angle
input("Move the camera to a different angle and press Enter to continue...")

# Capture images from the second angle
results_angle_2 = capture_images(angle=2)

# Turn off all LEDs
strip.fill((0, 0, 0))
strip.show()

picam2.stop()

# Convert results to arrays of points
points1 = np.array([results_angle_1[i] for i in range(LED_COUNT) if results_angle_1[i] is not None and results_angle_2[i] is not None])
points2 = np.array([results_angle_2[i] for i in range(LED_COUNT) if results_angle_1[i] is not None and results_angle_2[i] is not None])

print("Points 1:", points1)
print("Points 2:", points2)

# Find the essential matrix
E, mask = cv2.findEssentialMat(points1, points2, K,  method=cv2.RANSAC, prob=0.999, threshold=1.0)

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
os.makedirs(working_directory, exist_ok=True)

# Save the 3D coordinates to a CSV file
csv_filename = os.path.join(working_directory, "led_3d_coordinates.csv")
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["LED Index", "X Position", "Y Position", "Z Position"])
    for i, coord in enumerate(coordinates_3d):
        csv_writer.writerow([i, coord[0], coord[1], coord[2]])

print(f"Saved 3D LED positions to CSV file: {csv_filename}")