from typing import Union, List

import numpy as np
import torch
from torchvision import models, transforms


class StaticFeaturesExtractor:
    class FeatureExtractor:
        def extract_features(self, img: Union[List[np.ndarray], np.ndarray]) -> torch.tensor:
            raise NotImplementedError

    class HoGExtractor(FeatureExtractor):
        def __init__(self):
            pass

        def extract_features(self, img: Union[List[np.ndarray], np.ndarray]) -> torch.tensor:
            # TODO: import https://github.com/joaofaro/FHOG
            raise NotImplementedError

    class ColorNameExtractor(FeatureExtractor):
        def __init__(self):
            pass

        def extract_features(self, img: Union[List[np.ndarray], np.ndarray]) -> torch.tensor:
            raise NotImplementedError

    class VGGExtractor(FeatureExtractor):
        def __init__(self, num_layers):
            self._net = models.vgg16_bn(pretrained=True).features[:num_layers]
            self._preprocessor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        def extract_features(self, img: Union[List[np.ndarray], np.ndarray]) -> torch.tensor:
            if type(img) == list:
                return self._net(torch.stack([self._preprocessor(i) for i in img])).detach()
            else:
                return self._net(torch.stack([self._preprocessor(img)])).detach()[0, ...]

    def __init__(self, feature_names: list, output_size: int):
        self._extractors = []
        self._output_size = output_size

        for name in feature_names:
            if name.lower() == 'hog':
                self._extractors.append(self.HoGExtractor())
            elif name.lower() == 'colorname':
                self._extractors.append(self.ColorNameExtractor())
            elif 'conv' in name.lower():
                self._extractors.append(self.VGGExtractor(int(name[4:])))
            else:
                raise NotImplementedError

    def extract_features(self, x) -> torch.tensor:
        return torch.cat([extractor.extract_features(x)
                          for extractor in self._extractors])
