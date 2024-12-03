import time
import os
from picamera2 import Picamera2
import cv2

#sudo chmod 666 /dev/dma_heap/linux,cma

num_shots = 20

# Camera setup
picam2 = Picamera2()
camera_modes = picam2.sensor_modes
native_resolution = max(camera_modes, key=lambda mode: mode["size"])["size"]  # Get the highest resolution mode
camera_config = picam2.create_still_configuration(main={"size": native_resolution})
picam2.configure(camera_config)
picam2.start()

# Create the subfolder if it doesn't exist
subfolder = "calibphotos"
os.makedirs(subfolder, exist_ok=True)

picam2.start()
time.sleep(3)  # Allow the camera to warm up

# Capture 20 images with a 500ms delay
for i in range(1, num_shots + 1):

    # input("Press the space bar to capture the next image...")

    frame = picam2.capture_array()


    # Save the captured image
    filename = os.path.join(subfolder, f"checkerboard{i}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved image: {filename}")

    # Wait for 500 milliseconds
    time.sleep(1)

picam2.stop()

print("Finished capturing images.")