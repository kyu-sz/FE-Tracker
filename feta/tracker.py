import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import queue


def randomly_rotate_img_at_target(frame, bbox, angle_range=30):
    centre = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
    rot_mat = cv2.getRotationMatrix2D(centre, np.random.uniform(-angle_range, angle_range), 1)
    return cv2.warpAffine(frame, rot_mat, frame.shape)


class FilterEvolvingClassifier(nn.Module):
    def __init__(self):
        self._features = models.vgg16(pretrained=True).features
        self._classifier = nn.Conv2d(2048, 1, 3)

    def forward(self, x):
        # TODO: store the response of each filter.
        return self._classifier(self._features(x))


class Tracker:
    def __init__(self,
                 init_frame: np.ndarray,
                 init_bbox: list,
                 search_area_size_ratio=2,
                 lr=0.01,
                 momentum=0.9,
                 weight_decay=1e-4,
                 init_iter=10):
        # Create a VGG network with parameters from the PyTorch model zoo.
        self._net = FilterEvolvingClassifier()
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

        # Calculate search area.
        x_mid = init_bbox[0] + init_bbox[2] / 2
        y_mid = init_bbox[1] + init_bbox[3] / 2
        sa_x_min = max(x_mid - init_bbox[2] * search_area_size_ratio / 2, 0)
        sa_x_max = max(x_mid + init_bbox[2] * search_area_size_ratio / 2, init_frame.shape[1])
        sa_y_min = max(y_mid - init_bbox[3] * search_area_size_ratio / 2, 0)
        sa_y_max = max(y_mid + init_bbox[3] * search_area_size_ratio / 2, init_frame.shape[0])

        # Declare 2D supervision label.
        label = None

        # TODO: Train the network asynchronously.

        # Train the network using the original frame.
        resp_map = self._net(init_frame[sa_y_min:sa_y_max, sa_x_min:sa_x_max])
        label = np.zeros(resp_map.shape)
        label[y_mid - sa_y_max, x_mid - sa_x_min] = 1
        loss = self._criterion(resp_map, label)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # In the rest of the iterations, train the network using the rotated frame.
        for _ in range(init_iter - 1):
            rotated_init_frame = randomly_rotate_img_at_target(init_frame, init_bbox)
            resp_map = self._net(rotated_init_frame[sa_y_min:sa_y_max, sa_x_min:sa_x_max])
            loss = self._criterion(resp_map, label)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        self._last_bbox = init_bbox
        self._search_area_size_ratio = search_area_size_ratio

        # TODO: Use GMM to manage samples.
        # Create a queue to store the samples.
        self._samples = queue.Queue(10)
        self._samples.put((init_frame[sa_y_min:sa_y_max, sa_x_min:sa_x_max], label))

    def track(self, frame):
        # Calculate search area.
        x_mid = self._last_bbox[0] + self._last_bbox[2] / 2
        y_mid = self._last_bbox[1] + self._last_bbox[3] / 2
        sa_x_min = max(x_mid - self._last_bbox[2] * self._search_area_size_ratio / 2, 0)
        sa_x_max = max(x_mid + self._last_bbox[2] * self._search_area_size_ratio / 2, frame.shape[1])
        sa_y_min = max(y_mid - self._last_bbox[3] * self._search_area_size_ratio / 2, 0)
        sa_y_max = max(y_mid + self._last_bbox[3] * self._search_area_size_ratio / 2, frame.shape[0])

        # Feed into the network.
        resp_map = self._net(frame[sa_y_min:sa_y_max, sa_x_min:sa_x_max])

        # Find the greatest response.
        y, x = np.unravel_index(np.argmax(resp_map), resp_map.shape)

        # Update the latest bounding box.
        # TODO: Use the network to predict shape of the bounding box.
        # Currently directly use the original shape.
        self._last_bbox[0] = x
        self._last_bbox[1] = y
        self._last_bbox[2] = min(self._last_bbox[2], x * 2, (frame.shape[1] - x) / 2)
        self._last_bbox[3] = min(self._last_bbox[3], y * 2, (frame.shape[0] - y) / 2)

        # Create a new search area for training.
        sa_x_min = max(x_mid - self._last_bbox[2] * self._search_area_size_ratio / 2, 0)
        sa_x_max = max(x_mid + self._last_bbox[2] * self._search_area_size_ratio / 2, frame.shape[1])
        sa_y_min = max(y_mid - self._last_bbox[3] * self._search_area_size_ratio / 2, 0)
        sa_y_max = max(y_mid + self._last_bbox[3] * self._search_area_size_ratio / 2, frame.shape[0])

        # Train the network to fit the current object.
        resp_map = self._net(frame[sa_y_min:sa_y_max, sa_x_min:sa_x_max])
        label = np.zeros(resp_map.shape)
        label[y_mid - sa_y_max, x_mid - sa_x_min] = 1
        loss = self._criterion(resp_map, label)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Return the last bounding box.
        return self._last_bbox
