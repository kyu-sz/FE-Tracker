import torch
import torchvision.models as models


class StaticFeatureExtractor:
    class FeatureExtractor:
        def extract_features(self, x) -> torch.tensor:
            raise NotImplementedError

    class HoGExtractor(FeatureExtractor):
        def __init__(self):
            pass

        def extract_features(self, x) -> torch.tensor:
            raise NotImplementedError

    class ColorNameExtractor(FeatureExtractor):
        def __init__(self):
            pass

        def extract_features(self, x) -> torch.tensor:
            raise NotImplementedError

    class VGGExtractor(FeatureExtractor):
        def __init__(self, num_layers):
            self._net = models.vgg16(pretrained=True, batch_norm=True).features[:num_layers]

        def extract_features(self, x) -> torch.tensor:
            raise NotImplementedError

    def __init__(self, feature_names: list, output_size: int):
        self._extractors = []

        for name in feature_names:
            if name.lower() == 'hog':
                self._extractors.append(self.HoGExtractor())
            elif name.lower() == 'colorname':
                self._extractors.append(self.ColorNameExtractor())
            elif 'conv' in name.lower():
                self._extractors.append(self.VGGExtractor(int(name[4:])))
            else:
                raise NotImplementedError
