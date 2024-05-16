# Taken entirely from [0], none of this is mine.
#
# [0]: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import glob
import numpy as np
import cv2 as cv
import os

# Dimensions of the "inside" of the checkerboard here,
# i.e. for an 8x8 board, set to `(7, 7)`.
BOARD_DIM = (6, 7)

# Glob pattern for the images to use for calibration.
IMAGE_PATHS = glob.glob("./assets/warp/*.png")

# Whether to show each image where checkerboard was detected,
# for debugging purposes.
DEBUG_IMGS = False

def load_unwarper():
    """Returns a function used for undistorting an image.

    Unfortunately, results in a reduction in the resolution, so
    whether this is worth doing is to be decided on the resulting
    improvement from undistorting.
    """

    here = os.path.basename(__file__)
    print(f"{here}: loading unwarp params...")

    if len(IMAGE_PATHS) == 0:
        raise Exception(f"{here}: no images found for unwarping")

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((BOARD_DIM[0] * BOARD_DIM[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : BOARD_DIM[0], 0 : BOARD_DIM[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in IMAGE_PATHS:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, BOARD_DIM, None)

        if not ret:
            print(f"{here}: failed to find {BOARD_DIM} checkerboard in {fname}")
            continue

        # If found, add object points, image points (after refining them)
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        if DEBUG_IMGS:
            print(f"{here}: found {BOARD_DIM} checkerboard in {fname}")
            cv.drawChessboardCorners(img, BOARD_DIM, corners2, ret)
            cv.imshow("img", img)
            cv.waitKey()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    def f(img):
        """Unwarps the camera image."""

        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]

        return dst

    print(f"{here}: loaded unwarp params")
    return f


# This is the function to be used.
unwarper = load_unwarper()

if __name__ == "__main__":
    img = unwarper(cv.imread("./assets/warp/capture-2.png"))
    cv.imshow("img", img)
    cv.waitKey()
