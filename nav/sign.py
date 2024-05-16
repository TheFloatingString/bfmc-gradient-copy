# Code taken from [0], with model from [1]. YOLOv8 seems to be
# SOTA for object detection based on my cursory glance. I also
# saw a YOLOv9 but v8 is popular and good enough.
#
# The "model_id"s, from smallest to larges are `yolov8{n,s,m,l,x}-640`.
# The `yolov8n-640` works well enough.
#
# [0]: https://inference.roboflow.com/quickstart/explore_models/#run-a-model-on-universe
# [1]: https://github.com/ultralytics/ultralytics

from inference import get_model
import supervision as sv
# from paddleocr import PaddleOCR
import cv2 as cv
import os
import numpy as np

STOP_SIGN_CLASS_ID = 11
MODEL_ID = "yolov8s-640"

_here = os.path.basename(__file__)
print(f"{_here}: loading models...")
_yolo = get_model(model_id=MODEL_ID)
# _ocr = PaddleOCR(use_gpu=False, lang="en", debug=False, cls=False)
print(f"{_here}: loaded models")


def stop_signs(img):
    """Given an OpenCV image, return the bounding box for the stop sign.

    The return value is either `None` if nothing was found or
    `np.array([ax, ay, bx, by])`, where `(ax, ay)` is the top
    left point and `(bx, by)` the bottom right point.
    """

    results = _yolo.infer(img)
    detections = sv.Detections.from_inference(
        results[0].dict(by_alias=True, exclude_none=True)
    )
    (indices,) = np.where(detections.class_id == STOP_SIGN_CLASS_ID)

    if len(indices) == 0:
        return None

    return detections.xyxy[indices[0]]


def parking_signs(img):
    """Return the bounding box for the "P" in the image.

    Returns `None` if no `P` is found. Otherwise, returns
    [[ax, ay], [bx, by]], where a and b are respectively
    the top left and bottom right points.
    """
    results = _ocr.ocr(img)

    for xs in results:
        for x in xs:
            letter, conf = x[1]
            a, b = x[0][0], x[0][2]

            if letter == "P":
                return [a, b]

    return None


if __name__ == "__main__":
    image = "./assets/track_test_stop.png"
    image = cv.imread(image)

    results = _model.infer(image)
    detections = sv.Detections.from_inference(
        results[0].dict(by_alias=True, exclude_none=True)
    )

    print(f"found stop signs at: {stop_signs(image)}")
    print(f"detections: {detections}")

    # create supervision annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections
    )

    # display the image
    sv.plot_image(annotated_image)
