from typing import Union, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn

from fent import config
from fent.filter_evolving_net import FilterEvolvingNet
from fent.sample_management import SampleManager
from fent.static_features import StaticFeaturesExtractor


def r2i(x) -> int:
    return int(round(x))


class Tracker:
    @staticmethod
    def _choose_closest_anchor(bbox: Tuple[float, float, float, float]):
        return np.argmin([np.linalg.norm(anchor - bbox) for anchor in config.ANCHORS])

    @staticmethod
    def gaussian_kernel(kernel_shape: tuple, centre: tuple, sigma: float, amplitude: int = 1) -> np.ndarray:
        """Returns a 2D Gaussian kernel array."""
        y, x = np.ogrid[0:kernel_shape[0], 0:kernel_shape[1]]
        h = np.exp(-(pow(x - centre[0], 2) + pow(y - centre[1], 2)) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h * amplitude / np.max(h)

    @staticmethod
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
            scale = pow(2, np.random.uniform(-1, 1))
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
            new_bbox = [
                (centre_in_search_patch[0] if not mirror else search_area_size - centre_in_search_patch[0])
                - new_width / 2,
                (centre_in_search_patch[1] if not mirror else search_area_size - centre_in_search_patch[1])
                - new_height / 2,
                new_width,
                new_height
            ]
            rel_bbox = ((new_bbox[0] + new_bbox[2] / 2) / search_area_size,
                        (new_bbox[1] + new_bbox[3] / 2) / search_area_size,
                        new_bbox[2] / search_area_size,
                        new_bbox[3] / search_area_size)

            # cv2.imshow("display", draw_bbox(search_area_patch.copy(), new_bbox, (255, 0, 0)))
            # cv2.waitKey()

            samples.append((cv2.resize(search_area_patch, (config.IMPUT_SAMPLE_SIZE, config.IMPUT_SAMPLE_SIZE)),
                            rel_bbox))

        return samples

    def _add_samples_from_frame(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], init=False):
        # Create samples by rotating the searching area for initial training.
        # Directly store the static features of them instead of the original image patches into the sample manager.
        random_samples = self.generate_random_samples(frame, bbox, config.BATCH_SIZE)
        static_features = self._static_features_extractor.extract_features([sample[0] for sample in random_samples])
        self._sample_manager.add_samples(list(zip(
            [static_features[i, ...] for i in range(static_features.shape[0])],
            [sample[1] for sample in random_samples]
        )), init=init)

    def _train(self):
        random_samples = self._sample_manager.pick_samples()
        mini_batch = torch.cat([sample[0] for sample in random_samples])
        rel_bbox = [sample[1] for sample in random_samples]
        resp_map, bbox_reg_maps = self._net(mini_batch)
        resp_map_selected = torch.tensor([
            resp_map[i, self._choose_closest_anchor(rel_bbox[i]), :] for i in range(len(random_samples))
        ])
        resp_map_target = torch.tensor([
            self.gaussian_kernel(resp_map.shape[2:],
                                 (sample[1][0] + sample[1][2] / 2,
                                  sample[1][1] + sample[1][3] / 2),
                                 sigma=config.GAUSSIAN_LABEL_SIGMA)
            for sample in random_samples
        ])
        bbox_reg_target = torch.tensor([
            [
                [
                    [
                        [
                            x - rel_bbox[i, 0],
                            y - rel_bbox[i, 1],
                            anchor[0] / rel_bbox[i, 2],
                            anchor[1] / rel_bbox[i, 3],
                        ] for x in range(0, 1, 1 / config.OUTPUT_MAPS_SIZE)
                    ] for y in range(0, 1, 1 / config.OUTPUT_MAPS_SIZE)
                ] for anchor in config.ANCHORS
            ] for i in range(config.BATCH_SIZE)
        ])

        cls_loss = self._criterion(resp_map_selected, resp_map_target)
        bbox_reg_loss = self._criterion(bbox_reg_maps, bbox_reg_target)
        loss = cls_loss * config.LOSS_WEIGHTS['cls'] + bbox_reg_loss * config.LOSS_WEIGHTS['bbox_reg']
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def __init__(self,
                 init_frame: Union[str, np.ndarray],
                 init_bbox: Tuple[float, float, float, float]):
        np.random.seed(0)

        self._net = FilterEvolvingNet()
        self._static_features_extractor = StaticFeaturesExtractor(config.STATIC_FEATURES, config.STATIC_FEATURE_SIZE)
        self._sample_manager = SampleManager(config.BATCH_SIZE, config.MAX_NUM_SAMPLES)
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._net.parameters(),
                                          config.LEARNING_RATE,
                                          momentum=config.MOMENTOM,
                                          weight_decay=config.WEIGHT_DECAY)

        if type(init_frame) is str:
            init_frame = cv2.imread(init_frame)

        self._add_samples_from_frame(init_frame, init_bbox, init=True)
        for i in range(config.INIT_TRAIN_ITER):
            self._train()

        self._last_bbox = init_bbox

    def track(self, frame: Union[str, np.ndarray]):
        if type(frame) is str:
            frame = cv2.imread(frame)

        # Calculate search area.
        last_bbox_centre = (self._last_bbox[0] + self._last_bbox[2] / 2, self._last_bbox[1] + self._last_bbox[3] / 2)
        last_bbox_longer_side = max(self._last_bbox[2], self._last_bbox[3])
        frame_shorter_side = min(frame.shape[0], frame.shape[1])
        sa_size = min(frame_shorter_side, r2i(last_bbox_longer_side * config.SEARCH_AREA_SIZE_RATIO))
        sa_x = min(max(0, r2i(last_bbox_centre[0] - sa_size / 2)), frame.shape[1] - sa_size)
        sa_y = min(max(0, r2i(last_bbox_centre[1] - sa_size / 2)), frame.shape[0] - sa_size)

        # Feed into the network.
        static_features = self._static_features_extractor.extract_features(
            cv2.resize(frame[sa_y:sa_y + sa_size, sa_x:sa_x + sa_size],
                       (config.IMPUT_SAMPLE_SIZE, config.IMPUT_SAMPLE_SIZE)))
        resp_map, bbox_reg = self._net(torch.stack[static_features])

        # Find the greatest response.
        y = 0
        x = 0
        anchor = 0
        max_resp = 0
        for c in range(resp_map.shape[1]):
            peak_y, peak_x = np.unravel_index(np.argmax(resp_map[0, c, ...]), resp_map.shape[2:])
            resp = resp_map[0, c, peak_y, peak_x]
            if resp > max_resp:
                max_resp = resp
            y = peak_y
            x = peak_x
            anchor = c

        # update the latest bounding box
        centre = ((y / resp_map.shape[2] + bbox_reg[0, anchor * 4 + 1, y, x]) * sa_size,
                  (x / resp_map.shape[3] + bbox_reg[0, anchor * 4 + 0, y, x]) * sa_size)
        w = sa_size * (config.ANCHORS[anchor][0] + bbox_reg[0, anchor * 4 + 2, y, x]) / config.SEARCH_AREA_SIZE_RATIO
        h = sa_size * (config.ANCHORS[anchor][1] + bbox_reg[0, anchor * 4 + 3, y, x]) / config.SEARCH_AREA_SIZE_RATIO
        self._last_bbox = (centre[0] - w / 2, centre[1] - h / 2, w, h)

        # Add new samples from this frame and fine-tune the network.
        self._add_samples_from_frame(frame, self._last_bbox)
        for i in range(config.TRAIN_ITER_PER_ROUND):
            self._train()

        # Return the last bounding box.
        return self._last_bbox
