import numpy as np
import torch
from torchvision import datasets


def _first_n_per_class(targets, n):
    t = np.asarray(targets)
    return np.sort(np.concatenate([np.where(t == c)[0][:n] for c in range(t.max() + 1)]))


def get_tensors(dataset="cifar10", images_per_class=10, data_dir="./data"):
    Dataset = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    train_set = Dataset(data_dir, train=True, download=True)
    test_set = Dataset(data_dir, train=False, download=True)
    idx = _first_n_per_class(train_set.targets, images_per_class)
    to_tensor = lambda d: torch.from_numpy(d).permute(0, 3, 1, 2).float().div_(255.0)
    targets = torch.as_tensor(train_set.targets)
    return (to_tensor(train_set.data[idx]), targets[idx],
            to_tensor(test_set.data), torch.as_tensor(test_set.targets))
