import heapq

import numpy as np
from sklearn.mixture.gaussian_mixture import GaussianMixture
from torch import tensor


class SampleManager:
    def __init__(self, num_mixture_components=128, max_sample_num=10000):
        self._mixture = GaussianMixture(num_mixture_components)
        self._max_sample_num = max_sample_num
        self._samples = []
        self._component_sample_idx = None  # sample indices classified to each component

    def update_gmm(self):
        if len(self._samples) > self._max_sample_num and self._component_sample_idx:
            to_del = [t[2] for t in
                      heapq.nlargest(
                          len(self._samples) - self._max_sample_num,
                          [(self._samples[i][1], i) for i in range(len(self._samples))])]
            for i in reversed(to_del):
                del self._samples[i]

        self._mixture.fit(self._samples)
        pred = self._mixture.predict(self._samples)
        self._component_sample_idx = [[]] * self._mixture.n_components
        for idx, p in enumerate(pred):
            self._component_sample_idx[p].append(idx)

    def add_sample(self, features: tensor, bbox: list):
        self._samples.append((features, bbox, 0))
        if self._component_sample_idx:
            self._component_sample_idx[self._mixture.predict(features)].append(len(self._samples) - 1)

    def pick_samples(self) -> list:
        """
        Randonly pick a sample within each component.
        :return: a list of samples; each sample is a tuple of features and confidence score.
        """
        if self._component_sample_idx is None:
            self.update_gmm()

        selected_samples = []
        for c in self._component_sample_idx:
            selected_samples.append(self._samples[c[np.random.randint(0, len(c))]])
        return selected_samples
