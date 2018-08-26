import cv2
import numpy as np
import torch
import torch.nn as nn

from fent import config as CFG
from fent.filter_evolving_net import FilterEvolvingNet
from fent.sample_management import SampleManager
from fent.static_features import StaticFeaturesExtractor


class Tracker:
    @staticmethod
    def gaussian_kernel(kernel_shape: tuple, centre: tuple, sigma: float, amplitude=1):
        """Returns a 2D Gaussian kernel array."""
        y, x = np.ogrid[0:kernel_shape[0], 0:kernel_shape[1]]
        h = np.exp(-(pow(x - centre[0], 2) + pow(y - centre[1], 2)) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h * amplitude / np.max(h)

    @staticmethod
    def generate_random_samples(frame, bbox, n_samples):
        bbox_centre = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        bbox_longer_side = max(bbox[2], bbox[3])
        frame_shorter_side = min(frame.shape[0], frame.shape[1])

        search_area_size = min(frame_shorter_side, bbox_longer_side * CFG.SEARCH_AREA_SIZE_RATIO)

        # compute the vectors of the 4 corners of the original bounding box relative to the centre
        right_upper = (-bbox[2] / 2, bbox[3] / 2)
        left_upper = (-right_upper[0], right_upper[1])
        right_lower = (right_upper[0], -right_upper[1])
        left_lower = (-right_upper[0], -right_upper[1])

        def rotate_vector_by_angle(vec, angle):
            return vec[0] * np.cos(angle) - vec[1] * np.sin(angle), vec[0] * np.sin(angle) + vec[1] * np.cos(angle)

        # a patch larger than the search area is needed for rotation
        rot_patch_size = min(frame_shorter_side, search_area_size * 2)
        rot_patch_x = min(max(0, bbox_centre[0] - rot_patch_size), frame.shape[1] - rot_patch_size)
        rot_patch_y = min(max(0, bbox_centre[1] - rot_patch_size), frame.shape[0] - rot_patch_size)
        rot_patch_centre = (rot_patch_x + rot_patch_size / 2, rot_patch_y + rot_patch_size / 2)

        # compute the search area's location inside the rotation patch
        search_area_x = min(max(0, bbox_centre[0] - search_area_size), frame.shape[1] - search_area_size) - rot_patch_x
        search_area_y = min(max(0, bbox_centre[1] - search_area_size), frame.shape[0] - search_area_size) - rot_patch_y

        # compute the new bounding box's centre inside the search patch
        new_bbox_centre = (bbox_centre[0] - rot_patch_x - search_area_x, bbox_centre[1] - rot_patch_y - search_area_y)

        samples = [None] * n_samples
        for i in range(n_samples):
            # perform random rotation and scaling
            angle = np.random.uniform(-180, 180)
            scale = np.random.uniform(0.8, 1.2)
            rot_mat = cv2.getRotationMatrix2D(rot_patch_centre, angle, scale)
            rotated_patch = cv2.warpAffine(frame[
                                           rot_patch_y:rot_patch_y + rot_patch_size,
                                           rot_patch_x:rot_patch_x + rot_patch_size],
                                           rot_mat,
                                           frame.shape)

            # crop the search area
            search_area_patch = rotated_patch[
                                search_area_y:search_area_y + search_area_size,
                                search_area_x:search_area_x + search_area_size]

            # randomly mirror the sample
            mirror = np.random.randint(0, 2) == 1
            if mirror:
                cv2.flip(search_area_patch, 1)

            # compute the new bounding box
            rotated_right_upper = rotate_vector_by_angle(right_upper, angle)
            rotated_left_upper = rotate_vector_by_angle(left_upper, angle)
            rotated_right_lower = rotate_vector_by_angle(right_lower, angle)
            rotated_left_lower = rotate_vector_by_angle(left_lower, angle)
            new_bbox = [
                new_bbox_centre[0] if not mirror else search_area_size - new_bbox_centre[0],
                new_bbox_centre[1] if not mirror else search_area_size - new_bbox_centre[1],
                max(rotated_left_lower[0], rotated_left_upper[0], rotated_right_lower[0], rotated_right_upper[0]) -
                min(rotated_left_lower[0], rotated_left_upper[0], rotated_right_lower[0], rotated_right_upper[0]),
                max(rotated_left_lower[1], rotated_left_upper[1], rotated_right_lower[1], rotated_right_upper[1]) -
                min(rotated_left_lower[1], rotated_left_upper[1], rotated_right_lower[1], rotated_right_upper[1])
            ]

            samples[i] = (search_area_patch, new_bbox)

        return samples

    def _add_samples_from_frame(self, frame, bbox):
        # Create samples by rotating the searching area for initial training.
        # Directly store the static features of them instead of the original image patches into the sample manager.
        random_samples = self.generate_random_samples(frame, bbox, CFG.BATCH_SIZE)
        for sample_patch, bbox in random_samples:
            self._sample_manager.add_sample(self._static_features.extract_features(sample_patch), bbox)
        self._sample_manager.update_gmm()

    def _train(self):
        for static_features, bbox in self._sample_manager.pick_samples():
            # Train the network using these samples.
            resp_map, bbox_reg_maps = self._net(static_features)
            label = self.gaussian_kernel(resp_map.shape[2:], (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2))
            loss = self._criterion(resp_map[:, 0, :, :], label)
            # TODO: add bounding box regression loss.
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def __init__(self,
                 init_frame: np.ndarray,
                 init_bbox: list):
        np.random.seed(0)

        self._net = FilterEvolvingNet()
        self._static_features = StaticFeaturesExtractor(CFG.STATIC_FEATURES, CFG.STATIC_FEATURE_SIZE)
        self._sample_manager = SampleManager()
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._net.parameters(),
                                          CFG.LEARNING_RATE, momentum=CFG.MOMENTOM, weight_decay=CFG.WEIGHT_DECAY)

        self._add_samples_from_frame(init_frame, init_bbox)
        for i in range(CFG.INIT_TRAIN_ITER):
            self._train()

        self._last_bbox = init_bbox

    def track(self, frame):
        # Calculate search area.
        x_mid = self._last_bbox[0] + self._last_bbox[2] / 2
        y_mid = self._last_bbox[1] + self._last_bbox[3] / 2
        sa_x_min = max(x_mid - self._last_bbox[2] * CFG.SEARCH_AREA_SIZE_RATIO / 2, 0)
        sa_x_max = max(x_mid + self._last_bbox[2] * CFG.SEARCH_AREA_SIZE_RATIO / 2, frame.shape[1])
        sa_y_min = max(y_mid - self._last_bbox[3] * CFG.SEARCH_AREA_SIZE_RATIO / 2, 0)
        sa_y_max = max(y_mid + self._last_bbox[3] * CFG.SEARCH_AREA_SIZE_RATIO / 2, frame.shape[0])

        # Feed into the network.
        resp_map, bbox_reg = self._net(frame[sa_y_min:sa_y_max, sa_x_min:sa_x_max])

        # Find the greatest response.
        y, x = np.unravel_index(np.argmax(resp_map), resp_map.shape)

        # Update the latest bounding box.
        # TODO: Use the network to predict shape of the bounding box.
        # Currently directly use the original shape.
        self._last_bbox[2] = min(self._last_bbox[2], x * 2, (frame.shape[1] - x) / 2)
        self._last_bbox[3] = min(self._last_bbox[3], y * 2, (frame.shape[0] - y) / 2)
        self._last_bbox[0] = x - self._last_bbox[2] / 2
        self._last_bbox[1] = y - self._last_bbox[3] / 2

        # Add new samples from this frame and fine-tune the network.
        self._add_samples_from_frame(frame, self._last_bbox)
        for i in range(CFG.TRAIN_ITER_PER_ROUND):
            self._train()

        # Return the last bounding box.
        return self._last_bbox
