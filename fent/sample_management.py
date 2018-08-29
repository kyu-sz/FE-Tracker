import heapq
import threading

import numpy as np
from sklearn.mixture.gaussian_mixture import GaussianMixture
from torch import tensor


class SampleManager:
    def __init__(self, batch_size=128, max_sample_num=10000):
        self._mixture = GaussianMixture(batch_size - 1)
        self._gmm_update_thread = None

        self._max_sample_num = max_sample_num
        self._samples = None
        self._component_sample_idx = None  # sample indices classified to each component
        self._component_lock = threading.Lock()
        self._num_init_samples = 0

    def _update_gmm_sync(self):
        lock_acquired = False
        if len(self._samples) > self._max_sample_num and self._component_sample_idx:
            to_del = {
                t[2] for t in
                heapq.nlargest(
                    len(self._samples) - self._max_sample_num,
                    [(self._samples[i][1], i) for i in range(len(self._samples))])
            }.union(
                set(self._component_sample_idx[np.argmin(self._component_sample_idx)[0]])
            )
            self._component_lock.acquire()
            lock_acquired = True
            for i in sorted([i for i in to_del if i >= self._num_init_samples], reverse=True):
                del self._samples[i]

        self._mixture.fit(self._samples)
        pred = self._mixture.predict(self._samples)

        if not lock_acquired:
            self._component_lock.acquire()
        self._component_sample_idx = [[]] * self._mixture.n_components
        for idx, p in enumerate(pred):
            self._component_sample_idx[p].append(idx)

        self._component_lock.release()

    def update_gmm(self, async=True):
        if async:
            if self._gmm_update_thread is None or not self._gmm_update_thread.is_alive():
                self._gmm_update_thread = threading.Thread(target=self._update_gmm_sync)
                self._gmm_update_thread.start()
        else:
            self._update_gmm_sync()

    def add_init_samples(self, features: list, rel_bbox: list):
        self._num_init_samples = len(features)
        self._samples = [(f, b, 0) for f, b in zip(features, rel_bbox)]
        self.update_gmm(async=False)

    def add_sample(self, features: tensor, rel_bbox: list):
        self._samples.append((features, rel_bbox, 0))
        self._component_sample_idx[self._mixture.predict(features)].append(len(self._samples) - 1)

    def pick_samples(self) -> list:
        """
        Randonly pick a sample within each component.
        :return: a list of samples; each sample is a tuple of features and confidence score.
        """
        if self._component_sample_idx is None:
            self.update_gmm(async=False)

        selected_samples = [self._samples[np.random.randint(0, self._num_init_samples)]]
        self._component_lock.acquire()
        for c in self._component_sample_idx:
            selected_samples.append(self._samples[c[np.random.randint(0, len(c))]])
        self._component_lock.release()
        return selected_samples
