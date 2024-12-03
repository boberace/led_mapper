import numpy as np
import cv2 as cv
import glob
import os

current_dir = os.getcwd()

# Chessboard dimensions
chessboard_rows = 10  # Number of inner corners per row
chessboard_cols = 7  # Number of inner corners per column

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Create the subfolder if it doesn't exist
subfolder = "calibresults"
os.makedirs(subfolder, exist_ok=True)

images = glob.glob('*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and save the corners
        cv.drawChessboardCorners(img, (chessboard_cols, chessboard_rows), corners2, ret)
        result_filename = os.path.join(subfolder, os.path.basename(fname))
        cv.imwrite(result_filename, img)
        print(f"Saved image with corners: {result_filename}")
    else:
        print(f"No corners found in image: {fname}")

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the intrinsic parameters
print("Camera matrix (intrinsic parameters):\n", mtx)
print("Distortion coefficients:\n", dist)

# Save the intrinsic parameters to a file
np.savez('camera_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


# Camera matrix (intrinsic parameters):
#  [[2.60452056e+03 0.00000000e+00 1.62104331e+03]
#  [0.00000000e+00 2.60335521e+03 1.22438418e+03]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# Distortion coefficients:
#  [[ 0.20669488 -0.62582447 -0.00114616 -0.00350343  0.5119328 ]]