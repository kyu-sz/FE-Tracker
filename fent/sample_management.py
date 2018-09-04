import random
from typing import Tuple, List

from torch import tensor


class Sample:
    def __init__(self,
                 features: tensor,
                 bbox: Tuple[float, float, float, float],
                 frame_index: int,
                 important: bool = False):
        self.features = features
        self.bbox = bbox
        self.frame_index = frame_index
        self.important = important
        self.confidence = 0


class SampleManager:
    def __init__(self, batch_size: int = 128, max_sample_num: int = 10000):
        assert batch_size <= max_sample_num * 4, \
            'To ensure performance, ' \
            'the maximum number of samples should be at least 4 times equal or greater to the batch size.'
        self.batch_size = batch_size
        self._max_sample_num = max_sample_num
        self._usual_samples = []
        self._important_samples = []

    def add_sample(self, sample: Sample) -> None:
        if sample.frame_index == 0 or sample.important:
            self._important_samples.append(sample)
            if len(self._important_samples) > self._max_sample_num / 2:
                self._important_samples = \
                    sorted(self._important_samples,
                           key=lambda _init_samples: sample.confidence)[:self._max_sample_num / 2]
        else:
            self._usual_samples.append(sample)
            max_usual_sample = self._max_sample_num - len(self._important_samples)
            if len(self._usual_samples) > max_usual_sample:
                num_windows = max_usual_sample / 2
                index_window_size = self._usual_samples[-1].from_frame_index / num_windows
                window_end = index_window_size
                _new_samples = []
                samples_in_window = []
                for i in range(len(self._usual_samples)):
                    if self._usual_samples[i].from_index > window_end:
                        _new_samples.append(max(samples_in_window, key=lambda s: s.confidence))
                        window_end += index_window_size
                        samples_in_window = []
                    if self._usual_samples[i].confidence > 0:
                        samples_in_window.append(self._usual_samples[i])
                    else:
                        _new_samples.append(self._usual_samples[i])
                _new_samples.append(max(samples_in_window, key=lambda s: s.confidence))
                self._usual_samples = _new_samples

    def pick_samples(self) -> List[Sample]:
        if len(self._usual_samples) > 0:
            num_init_samples = max(1, self.batch_size - len(self._usual_samples))
            num_usual_samples = self.batch_size - num_init_samples
            num_new_samples = int(num_usual_samples / 4)
            return \
                random.sample(self._important_samples, num_init_samples) + \
                random.sample(self._usual_samples[:len(self._usual_samples) - num_new_samples],
                              num_usual_samples - num_new_samples) + \
                self._usual_samples[len(self._usual_samples) - num_new_samples:]
        else:
            return random.sample(self._important_samples, self.batch_size)
