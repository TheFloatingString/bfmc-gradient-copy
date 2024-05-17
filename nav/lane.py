import cv2 as cv
import numpy as np
from skimage import io

'''
from cv2.typing import MatLike
from numpy.typing import NDArray
'''
# TODO: more clever threshold finding? like otsu's method.

def find_lanes(img_orig, debug=True):
    """Find lanes within an OpenCV image.

    Returns a list of [(rho, theta)] and the original image with
    lines drawn on it. For what `rho` and `theta` mean, see [0].

    [0]: https://docs.opencv.org/4.9.0/d6/d10/tutorial_py_houghlines.html
    """

    img_orig = img_orig.copy()
    if debug:
        cv.imshow("Display window", img_orig)
        cv.waitKey()
    # threshold on white/gray sidewalk stripes
    lower = (80,80,80)
    upper = (255,255,255)
    thresh = cv.inRange(img_orig, lower, upper)
    if debug:
        cv.imshow("Display window", thresh)
        cv.waitKey()
    img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)

    # ensure image is grayscale, 8-bit
    assert len(img.shape) == 2
    assert img.dtype == "uint8"

    # canny edge detection
    canny = cv.Canny(image=thresh, threshold1=20, threshold2=70)
    if debug:
        cv.imshow("Display window", canny)
        cv.waitKey()

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
    if debug:
        print('Display ROI')
        cv.imshow("Display window", roi)
        cv.waitKey()

    # > ρ is measured in pixels and θ is measured in radians. First parameter,
    # > Input image should be a binary image, so apply threshold or use canny
    # > edge detection before finding applying hough transform. Second and third
    # > parameters are ρ and θ accuracies respectively. Fourth argument is the
    # > threshold, which means minimum vote it should get for it to be considered
    # > as a line
    #
    # https://docs.opencv.org/4.9.0/d6/d10/tutorial_py_houghlines.html
    lines = cv.HoughLines(roi, 1, np.pi / 180, 55)
    if debug: print(lines)
    # filter out horizontal lines
    filtered_lines = list(filter(lambda x: abs(x[0][1] - np.pi / 2) > 0.3, lines))

    # discard all the lines for later imgs
    orig = img_orig.copy()
    if debug:
        for [[rho, theta]] in filtered_lines:
            # thank u chatgpt
            # https://en.wikipedia.org/wiki/Hesse_normal_form
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img_orig, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv.imshow("Display window", img_orig)
        cv.waitKey()
    if debug: print(filtered_lines)
    theta=np.mean(np.asarray(filtered_lines),axis=0)[0][1]
    return theta

    sorted_lines = sorted(filtered_lines, key=lambda x: x[0][1])
    # alternate the lines by smallest -> largest -> smallest
    def alternate_sort(arr):
        result = []
        for i in range(len(arr) // 2):
            result.append(arr[i])
            result.append(arr[-(i + 1)])
        if len(arr) % 2:  # If the original list has odd number of elements, add the middle element at the end.
            result.append(arr[len(arr) // 2])
        return result
    alternate_lines = alternate_sort(sorted_lines)
    # print(alternate_lines)

    # cluster + grab average 2 lines
    # np.random.seed(42)
    # c1 = np.random.uniform(np.pi / 4, np.pi / 2)
    # c2 = np.random.uniform(np.pi / 2, np.pi * 3 / 4)
    clusters = ([], [])
    c1 = np.pi / 4
    c2 = np.pi * 3 / 4
    cs = [c1, c2]
    for l in alternate_lines:
        theta = l[0][1]
        _, c_idx = min((abs(theta - c), i) for i, c in enumerate(cs))
        n = len(clusters[c_idx])
        clusters[c_idx].append(l)
        cs[c_idx] = (cs[c_idx] * n + theta) / (n + 1)

    avg = (np.average(clusters[0], axis=0), np.average(clusters[1], axis=0))
    print(clusters)
    # print(clusters)
    #'''
    if clusters[0] == [] or clusters[1] == []:
        raise Exception("There was not 2 lines found")
    # print(clusters)

    # taken from: https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    def intersection(line1, line2):
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        return [[x0, y0]]


    [[[vx], [vy]]] = intersection(avg[0], avg[1])
    cv.circle(orig, (int(vx), int(vy)), 10, (255, 0, 0), -1)
    val = vx - w / 2

    print('avg')
    print(avg)
    for [[rho, theta]] in avg:
        # thank u chatgpt
        # https://en.wikipedia.org/wiki/Hesse_normal_form
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(orig, (x1, y1), (x2, y2), (0, 0, 255), 1)
        print(x1,y1,x2,y2)

    mask_overlay = np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], axis=-1)
    show = cv.addWeighted(orig, 1, mask_overlay, 0.2, 0)

    if debug:
        cv.imshow("Display window", show)
        cv.waitKey()

    return show, val, avg, (vx, vy)
    #'''
    
if __name__ == "__main__":
    img = io.imread("https://raw.githubusercontent.com/TheFloatingString/bfmc-gradient-copy/main/data/frame_85.png")
    # img = cv.imread("https://raw.githubusercontent.com/TheFloatingString/bfmc-finals-debug-remote/main/data/exp_debug_3/frame_1.png")
    # show, val, avg, inter = find_lanes(img, debug=True)
    #print(val, avg)
    theta = find_lanes(img, debug=True)
    print(theta)
