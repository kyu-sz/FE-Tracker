from typing import Tuple

import cv2
import numpy as np


def draw_bbox(img: np.ndarray, bbox: Tuple[float, float, float, float], color: tuple) -> None:
    cv2.rectangle(img,
                  (int(bbox[0]), int(bbox[1])),
                  (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                  color)
