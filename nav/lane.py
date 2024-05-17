import cv2 as cv
import numpy as np
from skimage import io

'''
from cv2.typing import MatLike
from numpy.typing import NDArray
'''
# TODO: more clever threshold finding? like otsu's method.

def find_lanes(img_orig, debug=False):
    """Find lanes within an OpenCV image.

    Returns a list of [(rho, theta)] and the original image with
    lines drawn on it. For what `rho` and `theta` mean, see [0].

    [0]: https://docs.opencv.org/4.9.0/d6/d10/tutorial_py_houghlines.html
    """

    img_orig = img_orig.copy()

    # threshold on white/gray sidewalk stripes
    lower = (80,80,80)
    upper = (255,255,255)
    thresh = cv.inRange(img_orig, lower, upper)

    img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)

    # ensure image is grayscale, 8-bit
    assert len(img.shape) == 2
    assert img.dtype == "uint8"

    # canny edge detection
    canny = cv.Canny(image=thresh, threshold1=20, threshold2=70)

    # trapezoidal roi
    h, w = img.shape
    # verts = np.array(
    #     [  # ruff: noqa: F401
    #         [0, 3 * h // 4],  # left midpoint
    #         [w, 3 * h // 4],  # right midpoint
    #         [3 * w // 4, h // 4],  # top right
    #         [1 * w // 4, h // 4],  # top left
    #     ]
    # )
    # TODO: maybe tilt roi according to the yaw
    if debug: print(f'w:{w}; h:{h}')
    verts = np.array([(w//2,h),(w,h),(w,h//3),(w//2,h//3)])
    '''
    verts = np.array(
        [  # ruff: noqa: F401
            [int(0.0*w), h],  # left midpoint
            [int(1.0*w), h],  # right midpoint
            [int(0.7*w), 3 * h // 5],  # top right
            [int(0.3*w), 3 * h // 5],  # top left
        ]
    )
    '''
    mask = np.zeros_like(canny, dtype="uint8")
    cv.fillPoly(mask, [verts], 255)
    roi = cv.bitwise_and(canny, canny, mask=mask)

    # > ρ is measured in pixels and θ is measured in radians. First parameter,
    # > Input image should be a binary image, so apply threshold or use canny
    # > edge detection before finding applying hough transform. Second and third
    # > parameters are ρ and θ accuracies respectively. Fourth argument is the
    # > threshold, which means minimum vote it should get for it to be considered
    # > as a line
    #
    # https://docs.opencv.org/4.9.0/d6/d10/tutorial_py_houghlines.html
    lines = cv.HoughLines(roi, 1, np.pi / 180, 55)
    # filter out horizontal lines
    filtered_lines = list(filter(lambda x: abs(x[0][1] - np.pi / 2) > 0.3, lines))

    # discard all the lines for later imgs
    orig = img_orig.copy()
    if debug: print(filtered_lines)
    theta=np.mean(np.asarray(filtered_lines),axis=0)[0][1]
    return theta
    
if __name__ == "__main__":
    img = io.imread("https://raw.githubusercontent.com/TheFloatingString/bfmc-gradient-copy/main/data/frame_85.png")
    # img = cv.imread("https://raw.githubusercontent.com/TheFloatingString/bfmc-finals-debug-remote/main/data/exp_debug_3/frame_1.png")
    # show, val, avg, inter = find_lanes(img, debug=True)
    #print(val, avg)
    theta = find_lanes(img, debug=True)
    print(theta)
