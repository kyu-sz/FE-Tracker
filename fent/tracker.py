import itertools
import random
from typing import Union, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from fent import config
from fent.filter_evolving_net import FilterEvolvingNet
from fent.sample_management import SampleManager, Sample
from fent.static_features import StaticFeaturesExtractor
from fent.utils import generate_random_samples, gaussian_kernel, choose_closest_anchor, r2i


class Tracker:

    def _add_samples_from_frame(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], frame_index: int):
        # Create samples by rotating the searching area for initial training.
        # Directly store the static features of them instead of the original image patches into the sample manager.
        random_samples = generate_random_samples(frame, bbox, config.BATCH_SIZE)
        static_features = self._static_features_extractor.extract_features([sample[0] for sample in random_samples])
        for f, b in zip(
                [static_features[i, ...] for i in range(static_features.shape[0])],
                [sample[1] for sample in random_samples]
        ):
            self._sample_manager.add_sample(Sample(f, b, frame_index))

    def _train(self):
        self._net.train(True)

        random_samples = self._sample_manager.pick_samples()
        mini_batch = torch.stack([sample.features for sample in random_samples])
        bbox_gt = [sample.bbox for sample in random_samples]
        resp_map, bbox_reg_maps = self._net(mini_batch)
        resp_map_selected = torch.stack([
            resp_map[i, choose_closest_anchor(bbox_gt[i]), :] for i in range(len(random_samples))
        ])

        gt_resp_centres = [
            ((sample.bbox[0] - 0.5) / 2,
             (sample.bbox[1] - 0.5) / 2)
            for sample in random_samples
        ]

        resp_map_gt = torch.tensor([
            gaussian_kernel(resp_map.shape[2:],
                            gt_resp_centre,
                            sigma=config.GAUSSIAN_LABEL_SIGMA)
            for gt_resp_centre in gt_resp_centres
        ], dtype=torch.float32)
        half_unit = 0.5 / config.OUTPUT_MAPS_SIZE
        bbox_reg_maps_gt = torch.stack([
            torch.tensor([
                [
                    list(itertools.chain.from_iterable([gt_resp_centres[i][0] - x,
                                                        gt_resp_centres[i][1] - y,
                                                        bbox_gt[i][2] / config.PERCEPTIVE_FIELD_SIZE / anchor[0] - 1,
                                                        bbox_gt[i][3] / config.PERCEPTIVE_FIELD_SIZE / anchor[1] - 1]
                                                       for anchor in config.ANCHORS))
                    for x in np.linspace(half_unit, 1 - half_unit, config.OUTPUT_MAPS_SIZE)
                ] for y in np.linspace(half_unit, 1 - half_unit, config.OUTPUT_MAPS_SIZE)
            ], dtype=torch.float32) for i in range(config.BATCH_SIZE)
        ]).permute([0, 3, 1, 2])

        cls_loss = self._cls_criterion(resp_map_selected, resp_map_gt)
        bbox_reg_loss = self._bbox_reg_criterion(bbox_reg_maps, bbox_reg_maps_gt)
        loss = cls_loss * config.LOSS_WEIGHTS['cls'] + bbox_reg_loss * config.LOSS_WEIGHTS['bbox_reg']

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def __init__(self,
                 init_frame: Union[str, np.ndarray],
                 init_bbox: Tuple[float, float, float, float]):
        np.random.seed(0)
        random.seed(0)

        self._net = FilterEvolvingNet()
        self._static_features_extractor = StaticFeaturesExtractor(config.STATIC_FEATURES, config.STATIC_FEATURE_SIZE)
        self._sample_manager = SampleManager(config.BATCH_SIZE, config.MAX_NUM_SAMPLES)
        self._cls_criterion = nn.MSELoss()
        self._bbox_reg_criterion = nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._net.parameters(),
                                          config.LEARNING_RATE,
                                          momentum=config.MOMENTOM,
                                          weight_decay=config.WEIGHT_DECAY)

        if type(init_frame) is str:
            init_frame = cv2.imread(init_frame)

        self._frame_index = 0

        self._add_samples_from_frame(init_frame, init_bbox, self._frame_index)
        for i in range(config.INIT_TRAIN_ITER):
            self._train()

        self._last_bbox = init_bbox

    def track(self, frame: Union[str, np.ndarray]):
        self._net.train(False)

        if type(frame) is str:
            frame = cv2.imread(frame)

        # calculate search area
        last_bbox_centre = (self._last_bbox[0] + self._last_bbox[2] / 2, self._last_bbox[1] + self._last_bbox[3] / 2)
        last_bbox_longer_side = max(self._last_bbox[2], self._last_bbox[3])
        frame_shorter_side = min(frame.shape[0], frame.shape[1])
        sa_size = min(frame_shorter_side, r2i(last_bbox_longer_side * config.SEARCH_AREA_SIZE_RATIO))
        sa_x = min(max(0, r2i(last_bbox_centre[0] - sa_size / 2)), frame.shape[1] - sa_size)
        sa_y = min(max(0, r2i(last_bbox_centre[1] - sa_size / 2)), frame.shape[0] - sa_size)

        # feed into the network
        static_features = self._static_features_extractor.extract_features(
            cv2.resize(frame[sa_y:sa_y + sa_size, sa_x:sa_x + sa_size],
                       (config.INPUT_SAMPLE_SIZE, config.INPUT_SAMPLE_SIZE)))
        resp_map, bbox_reg = self._net(torch.stack([static_features]))
        resp_map = resp_map.detach().numpy()
        bbox_reg = bbox_reg.detach().numpy()

        # find the greatest response
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
        centre = (((x + bbox_reg[0, anchor * 4 + 0, y, x]) * 2 + 0.5) / config.INPUT_SAMPLE_SIZE * sa_size + sa_x,
                  ((y + bbox_reg[0, anchor * 4 + 1, y, x]) * 2 + 0.5) / config.INPUT_SAMPLE_SIZE * sa_size + sa_y)
        w = sa_size / config.SEARCH_AREA_SIZE_RATIO \
            * config.ANCHORS[anchor][0] * (1 + bbox_reg[0, anchor * 4 + 2, y, x])
        h = sa_size / config.SEARCH_AREA_SIZE_RATIO \
            * config.ANCHORS[anchor][1] * (1 + bbox_reg[0, anchor * 4 + 3, y, x])
        self._last_bbox = (centre[0] - w / 2, centre[1] - h / 2, w, h)

        # add new samples from this frame and fine-tune the network
        self._frame_index += 1
        self._add_samples_from_frame(frame, self._last_bbox, self._frame_index)
        for i in range(config.TRAIN_ITER_PER_ROUND):
            self._train()

        # return the last bounding box
        return self._last_bbox
