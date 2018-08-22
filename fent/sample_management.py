import heapq

import numpy as np
from sklearn.mixture.gaussian_mixture import GaussianMixture
from torch import tensor


class SampleManager:
    def __init__(self, num_mixture_components=128, max_sample_num=10000):
        self._mixture = GaussianMixture(num_mixture_components)
        self._max_sample_num = max_sample_num
        self._samples = []
        self._component_sample_idx = []  # sample indices classified to each component
        self._sample_hardness = []

    def update(self):
        if len(self._samples) > self._max_sample_num and len(self._component_sample_idx) > 0:
            # drop the size of the greatest component to the size of the smallest component.
            component_idx_with_largest_size = np.argmax([len(c) for c in self._component_sample_idx])[0]
            smallest_component_size = min([len(c) for c in self._component_sample_idx])
            del self._samples[[t[1] for t in
                               heapq.nsmallest(
                                   len(self._component_sample_idx[
                                           component_idx_with_largest_size]) - smallest_component_size,
                                   [(self._sample_hardness[i], i) for i in
                                    self._component_sample_idx[component_idx_with_largest_size]])]]

        self._mixture.fit(self._samples)
        pred = self._mixture.predict(self._samples)
        self._component_sample_idx = [[]] * self._mixture.n_components
        for idx, p in enumerate(pred):
            self._component_sample_idx[p].append(idx)

    def add_sample(self, features: tensor):
        self._samples.append(features)

    def pick_sample(self) -> (list, list):
        """
        Pick samples within each component with probability reciprocal to network confidence.
        :return: a tuple of picked samples and their indices in the sample manager.
        """
        selected_samples = []
        selected_indices = []
        for c in self._component_sample_idx:
            prob = np.array([self._sample_hardness[i] for i in c])
            prob_cumsum = np.cumsum(prob)
            rand = np.random.uniform(prob_cumsum[-1])
            idx = np.searchsorted(prob_cumsum, rand)
            selected_samples.append(self._samples[idx])
            selected_indices.append(idx)
        return selected_samples, selected_indices

    def set_sample_hardness(self, sample_indices: list, hardness: list):
        for i, h in zip(sample_indices, hardness):
            self._sample_hardness[i] = h
