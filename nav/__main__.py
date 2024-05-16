import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv

from .lane import find_lanes

rclpy.init()
br = CvBridge()

def image_cb(data: Image):
    img = br.imgmsg_to_cv2(data, "bgr8")
    img = find_lanes(img)
    cv.imshow("preview", img)
    cv.waitKey(1)

node = rclpy.create_node('nav')
# TODO: Fix here and in `camera.py` in `example`.
img_sub = node.create_subscription(Image, "/camera1/image_raw", image_cb, 1)

rclpy.spin(node)
