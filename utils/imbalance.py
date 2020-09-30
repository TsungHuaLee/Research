import collections
import random

import torch
    
import torch
import torch.utils.data
import torchvision



class UnderSampler(torch.utils.data.IterableDataset):
    """Dataset wrapper for under-sampling.
    This method is based on rejection sampling.
    Parameters:
        dataset
        desired_dist: The desired class distribution. The keys are the classes whilst the
            values are the desired class percentages. The values must sum up to 1.
        seed: Random seed for reproducibility.
    Attributes:
        actual_dist: The counts of the observed sample labels.
        rng: A random number generator instance.
    References:
        - https://www.wikiwand.com/en/Rejection_sampling
    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, desired_dist: dict,
                 seed: int = None):

        self.dataset = dataset
        self.desired_dist = {c: p / sum(desired_dist.values()) for c, p in desired_dist.items()}
        self.seed = seed

        self.actual_dist = collections.Counter()
        self.rng = random.Random(seed)
        self._pivot = None

    def __iter__(self):

        for x, y in self.dataset:

            self.actual_dist[y] += 1

            # To ease notation
            f = self.desired_dist
            g = self.actual_dist

            # Check if the pivot needs to be changed
            if y != self._pivot:
                self._pivot = max(g.keys(), key=lambda y: f[y] / g[y])
            else:
                yield x, y
                continue

            # Determine the sampling ratio if the observed label is not the pivot
            M = f[self._pivot] / g[self._pivot]
            ratio = f[y] / (M * g[y])

            if ratio < 1 and self.rng.random() < ratio:
                yield x, y

    @classmethod
    def expected_size(cls, n, desired_dist, actual_dist):
        M = max(
            desired_dist.get(k) / actual_dist.get(k)
            for k in set(desired_dist) | set(actual_dist)
        )
        return int(n / M)

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            return dataset[idx]
#             raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples