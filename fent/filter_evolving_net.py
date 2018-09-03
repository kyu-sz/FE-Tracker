from collections import OrderedDict

import torch.nn as nn

from . import config


class FilterEvolvingNet(nn.Module):
    def __init__(self):
        super(FilterEvolvingNet, self).__init__()
        self._features = nn.Sequential(OrderedDict([
            ('conv3a', nn.Conv2d(128, 256, 3, padding=3, dilation=3)),
            ('norm3a', nn.BatchNorm2d(256)),
            ('relu3a', nn.ReLU(inplace=True)),
            ('conv3b', nn.Conv2d(256, 256, 3, padding=1)),
            ('norm3b', nn.BatchNorm2d(256)),
            ('relu3b', nn.ReLU(inplace=True)),
            ('conv3c', nn.Conv2d(256, 256, 3, padding=1)),
            ('norm3c', nn.BatchNorm2d(256)),
            ('relu3c', nn.ReLU(inplace=True)),
            ('fc4', nn.Conv2d(256, 256, 7, padding=3)),
            ('relu4', nn.ReLU(inplace=True)),
            ('dropout4', nn.Dropout()),
            ('fc5', nn.Conv2d(256, 256, 1)),
            ('relu5', nn.ReLU(inplace=True)),
            ('dropout5', nn.Dropout()),
        ]))
        self._classifier = nn.Sequential(
            nn.Conv2d(256, len(config.ANCHORS), 1),
            nn.Sigmoid(),
        )
        self._bbox_reg = nn.Conv2d(256, 4 * len(config.ANCHORS), 7, padding=3)

    def forward(self, x):
        # TODO: store the response of each filter.
        features = self._features(x)
        return self._classifier(features), self._bbox_reg(features)
