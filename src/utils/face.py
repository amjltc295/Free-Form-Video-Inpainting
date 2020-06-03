# Based on https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py
import sys
import os

from collections import OrderedDict

import dlib
import numpy as np
import cv2

from utils.logging_config import logger


detector = dlib.get_frontal_face_detector()
predictor = None
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "shape_predictor_68_face_landmarks.dat"
)


# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


# For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", (4))
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS


def landmarks_to_np(landmarks, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((landmarks.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, landmarks.num_parts):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def connect_landmarks(image_shape, landmarks, color=None, alpha=0.75):
    landmarks_contour = np.zeros(image_shape)

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if color is None:
        color = 1

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = landmarks[j:k]

        for x in range(1, len(pts)):
            ptA = tuple(pts[x - 1])
            ptB = tuple(pts[x])
            cv2.line(landmarks_contour, ptA, ptB, color, 2)

    # apply the transparent landmarks_contour
    # cv2.addWeighted(landmarks_contour, alpha, output, 1 - alpha, 0, output)

    # return the output image
    return landmarks_contour


def get_landmarks_contour(image, landmarks_predictor_path=DEFAULT_PATH):
    global predictor
    if predictor is None:
        assert os.path.exists(landmarks_predictor_path)
        predictor = dlib.shape_predictor(landmarks_predictor_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        rects = detector(gray, 1)
        assert len(rects) == 1

        landmarks = predictor(gray, rects[0])
        landmarks = landmarks_to_np(landmarks)
        img = connect_landmarks(gray.shape, landmarks)
    except Exception as err:
        logger.error(err, exc_info=True)
        logger.error("Set countour to black")
        img = np.zeros_like(gray)
    return img


if __name__ == "__main__":
    image = cv2.imread(sys.argv[1])
    landmarks_contour = get_landmarks_contour(image)
    cv2.imshow('landmarks contour', landmarks_contour)
    # create two copies of the input image -- one for the
    cv2.waitKey()
