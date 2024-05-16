import cv2 as cv
import numpy as np
from nav.lane import find_lanes


def find_crosswalk_dist(img, debug=False) -> float:
    img_orig = img.copy()
    img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)

    # ensure image is grayscale, 8-bit
    assert len(img.shape) == 2
    assert img.dtype == "uint8"

    h, w = img.shape

    _, _, avg, inter = find_lanes(img_orig, debug)
    [[[rho1, theta1]], [[rho2, theta2]]] = avg
    # find x intercepts
    lx1 = (rho1 - w * np.sin(theta1)) / np.cos(theta1)
    rx1 = (rho2 - w * np.sin(theta2)) / np.cos(theta2)

    lxi = [int(lx1), w]
    rxi = [int(rx1), w]

    # grab bottom triangle under the lane
    tri = np.array([lxi, rxi, (inter)]).astype(int)
    print(lxi)
    print(rxi)
    print(np.array(inter, np.int32))
    mask = np.zeros((h, w), np.uint8)
    cv.drawContours(mask, [tri], 0, (255, 255, 255), -1)
    mask_bool = mask.astype(bool)
    masked_image = np.zeros_like(img)
    masked_image[mask_bool] = img[mask_bool]

    # cv.imshow("Display window", masked_image)
    # cv.waitKey()

    lower = 200
    upper = 255
    thresh = cv.inRange(masked_image, lower, upper)

    # apply morphology close to fill interior regions in mask
    kernel = np.ones((3, 3), np.uint8)
    morph = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)
    if debug:
        cv.imshow("Display window", morph)
        cv.waitKey()

    # praise gpt
    edges = cv.Canny(morph, 50, 150, apertureSize=3)
    # why does probabilistic hough lines give back (x0, y0), (x1, y1) instead of (rho, theta)???
    lines = cv.HoughLines(edges, 1, np.pi / 180, 60)
    distance_to_crosswalk = float("inf")
    if lines is not None:
        if True:
            for [[rho, theta]] in lines:
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
                if abs(theta - np.pi / 2) < 0.3:
                    mid = (rho - w / 2 * np.cos(theta)) / np.sin(theta)
                    distance_to_crosswalk = min(distance_to_crosswalk, h - mid)
                    cv.line(img_orig, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # cv.imshow("Display window", img_orig)
    # cv.waitKey()
    return distance_to_crosswalk


if __name__ == "__main__":
    img = cv.imread("./data/crosswalk-and-stop-lines-with-parking-sign/frame_8.png")
    dist = find_crosswalk_dist(img, debug=True)
    print(dist)
