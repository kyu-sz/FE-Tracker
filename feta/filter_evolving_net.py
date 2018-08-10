from collections import OrderedDict
from itertools import chain

import torch
import torch.nn as nn
import torchvision.models as models


class FilterEvolvingNet(nn.Module):
    def __init__(self):
        self._base_features = models.vgg16(pretrained=True, batch_norm=True).features
        self._evolving_features = nn.Sequential(OrderedDict([
            ('conv3a', nn.Conv2d(256, 1, 3)),
            ('norm3a', nn.BatchNorm2d(5)),
            ('relu3a', nn.ReLU(inplace=True)),
            ('conv3b', nn.Conv2d(256, 1, 3)),
            ('norm3b', nn.BatchNorm2d(5)),
            ('relu3b', nn.ReLU(inplace=True)),
            ('conv3c', nn.Conv2d(256, 1, 3)),
            ('norm3c', nn.BatchNorm2d(5)),
            ('relu3c', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4a', nn.Conv2d(512, 1, 3)),
            ('norm4a', nn.BatchNorm2d(5)),
            ('relu4a', nn.ReLU(inplace=True)),
            ('conv4b', nn.Conv2d(512, 1, 3)),
            ('norm4b', nn.BatchNorm2d(5)),
            ('relu4b', nn.ReLU(inplace=True)),
            ('conv4c', nn.Conv2d(512, 1, 3)),
            ('norm4c', nn.BatchNorm2d(5)),
            ('relu4c', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv5a', nn.Conv2d(512, 1, 3)),
            ('norm5a', nn.BatchNorm2d(5)),
            ('relu5a', nn.ReLU(inplace=True)),
            ('conv5b', nn.Conv2d(512, 1, 3)),
            ('norm5b', nn.BatchNorm2d(5)),
            ('relu5b', nn.ReLU(inplace=True)),
            ('conv5c', nn.Conv2d(512, 1, 3)),
            ('norm5c', nn.BatchNorm2d(5)),
            ('relu5c', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
        self._classifier = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        # TODO: store the response of each filter.
        return self._classifier(self._features(x))