from typing import Tuple, List

import cv2
import numpy as np

from fent import config


def generate_random_samples(frame: np.ndarray, bbox: Tuple[float, float, float, float], n_samples: int) \
        -> List[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    bbox_centre = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
    bbox_longer_side = max(bbox[2], bbox[3])
    frame_shorter_side = min(frame.shape[0], frame.shape[1])

    search_area_size = min(frame_shorter_side, r2i(bbox_longer_side * config.SEARCH_AREA_SIZE_RATIO))

    # compute the vectors of the 4 corners of the original bounding box relative to the centre
    right_upper = (-bbox[2] / 2, bbox[3] / 2)
    left_upper = (-right_upper[0], right_upper[1])
    right_lower = (right_upper[0], -right_upper[1])
    left_lower = (-right_upper[0], -right_upper[1])

    def rotate_vector_by_angle(vec, angle):
        return vec[0] * np.cos(angle) - vec[1] * np.sin(angle), vec[0] * np.sin(angle) + vec[1] * np.cos(angle)

    # a patch larger than the search area is needed for rotation
    rot_patch_size = min(frame_shorter_side, search_area_size * 2)
    rot_patch_x = min(max(0, r2i(bbox_centre[0] - rot_patch_size / 2)), frame.shape[1] - rot_patch_size / 2)
    rot_patch_y = min(max(0, r2i(bbox_centre[1] - rot_patch_size / 2)), frame.shape[0] - rot_patch_size / 2)
    rot_patch = frame[rot_patch_y:rot_patch_y + rot_patch_size, rot_patch_x:rot_patch_x + rot_patch_size]

    # compute the search area's location inside the rotation patch
    search_area_x = min(max(0, r2i(bbox_centre[0] - search_area_size / 2)),
                        frame.shape[1] - search_area_size) - rot_patch_x
    search_area_y = min(max(0, r2i(bbox_centre[1] - search_area_size / 2)),
                        frame.shape[0] - search_area_size) - rot_patch_y

    # compute the new bounding box's centre inside the patches
    centre_in_rot_patch = (bbox_centre[0] - rot_patch_x, bbox_centre[1] - rot_patch_y)
    centre_in_search_patch = (centre_in_rot_patch[0] - search_area_x, centre_in_rot_patch[1] - search_area_y)

    samples = []
    for i in range(n_samples):
        # perform random rotation and scaling
        rot_angle = i and np.random.uniform(-180, 180)
        scale = pow(2, np.random.uniform(-1, 1)) if i else 1
        rot_mat = cv2.getRotationMatrix2D(centre_in_rot_patch, rot_angle, scale)

        rotated_patch = cv2.warpAffine(rot_patch, rot_mat, rot_patch.shape[:2])

        # crop the search area
        search_area_patch = rotated_patch[
                            search_area_y:search_area_y + search_area_size,
                            search_area_x:search_area_x + search_area_size]

        # randomly mirror the sample
        mirror = np.random.randint(0, 2) == 1
        if mirror:
            cv2.flip(search_area_patch, 1)

        # compute the new bounding box
        rotated_right_upper = rotate_vector_by_angle(right_upper, rot_angle)
        rotated_left_upper = rotate_vector_by_angle(left_upper, rot_angle)
        rotated_right_lower = rotate_vector_by_angle(right_lower, rot_angle)
        rotated_left_lower = rotate_vector_by_angle(left_lower, rot_angle)
        new_width = (max(rotated_left_lower[0],
                         rotated_left_upper[0],
                         rotated_right_lower[0],
                         rotated_right_upper[0]) -
                     min(rotated_left_lower[0],
                         rotated_left_upper[0],
                         rotated_right_lower[0],
                         rotated_right_upper[0])
                     ) * scale
        new_height = (max(rotated_left_lower[1],
                          rotated_left_upper[1],
                          rotated_right_lower[1],
                          rotated_right_upper[1]) -
                      min(rotated_left_lower[1],
                          rotated_left_upper[1],
                          rotated_right_lower[1],
                          rotated_right_upper[1])
                      ) * scale
        new_bbox = (
            (centre_in_search_patch[0] if not mirror else search_area_size - centre_in_search_patch[0])
            - new_width / 2,
            (centre_in_search_patch[1] if not mirror else search_area_size - centre_in_search_patch[1])
            - new_height / 2,
            new_width,
            new_height
        )

        # resize to a fixed sample size
        fixed_size_sample = cv2.resize(search_area_patch, (config.INPUT_SAMPLE_SIZE, config.INPUT_SAMPLE_SIZE))
        resize_ratio = config.INPUT_SAMPLE_SIZE / search_area_size
        new_bbox = (new_bbox[0] * resize_ratio,
                    new_bbox[1] * resize_ratio,
                    new_bbox[2] * resize_ratio,
                    new_bbox[3] * resize_ratio)
        # canvas = fixed_size_sample.copy()
        # draw_bbox(canvas, new_bbox, (255, 0, 0))
        # cv2.imshow("display", canvas)
        # cv2.waitKey()
        samples.append((fixed_size_sample, new_bbox))

    return samples


def gaussian_kernel(kernel_shape: tuple, centre: tuple, sigma: float, amplitude: int = 1) -> np.ndarray:
    """Returns a 2D Gaussian kernel array."""
    y, x = np.ogrid[0:kernel_shape[0], 0:kernel_shape[1]]
    h = np.exp(-(pow(x - centre[0], 2) + pow(y - centre[1], 2)) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h * amplitude / np.max(h)


def choose_closest_anchor(bbox: Tuple[float, float, float, float]):
    return np.argmin([np.linalg.norm(anchor - bbox[2:]) for anchor in config.ANCHORS])


def r2i(x) -> int:
    return int(round(x))
