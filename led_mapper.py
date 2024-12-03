import time
import board
import numpy as np
import neopixel_spi as neopixel
import cv2
from picamera2 import Picamera2, Preview
import csv
import os

current_dir = os.getcwd()

#sudo chmod 666 /dev/dma_heap/linux,cma

# LED configuration
LED_COUNT = 3  # Number of LEDs in your strip
PIXEL_ORDER = neopixel.GRB

# Initialize the LED strip
spi = board.SPI()
strip = neopixel.NeoPixel_SPI(spi, LED_COUNT, pixel_order=PIXEL_ORDER, auto_write=False)

# Camera setup
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
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assume the largest contour is the LED
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None


results = {}

def capture_images(angle):
    results = {}
    for i in range(LED_COUNT):
        # Turn off all LEDs
        strip.fill((0, 0, 0))
        strip.show()
        time.sleep(0.5)  # Ensure LEDs are off before turning on the next one

        # Light up the current LED
        strip[i] = (255, 255, 255)
        strip.show()
        time.sleep(0.5)  # Allow time for the camera to capture

        # Capture image
        frame = picam2.capture_array()

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save the captured image
        filename = os.path.join(current_dir, "led_{i}_angle_{angle}.jpg") 
        cv2.imwrite(filename, frame_rgb)
        print(f"Saved image: {filename}")

        coord = detect_led(frame_rgb)
        if coord:
            results[i] = coord
            print(f"LED {i}: {coord}")
        else:
            results[i] = None
            print(f"LED {i}: No LED detected")

        # Turn off the current LED
        strip[i] = (0, 0, 0)
        strip.show()
        # time.sleep(0.5)  # Ensure the LED is off before moving to the next one

    return results

results = capture_images(0)


# Turn off all LEDs
strip.fill((0, 0, 0))
strip.show()

picam2.stop()

# Save the results to a CSV file
csv_filename = os.path.join(current_dir, "led_coordinates.csv")
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["LED Index", "X Position", "Y Position"])
    for i in range(LED_COUNT):
        if results[i] is not None:
            csv_writer.writerow([i, results[i][0], results[i][1]])
        else:
            csv_writer.writerow([i, "Not Detected", "Not Detected"])

print(f"Saved LED positions to CSV file: {csv_filename}")