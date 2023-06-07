import torch
import os
import json
from collections import defaultdict
import numpy as np
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from utils import cifar100_preprocessing, femnist_preprocessing


MEAN_STD = {
    "CIFAR100": ((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
    "FEMNIST": ((0.9642384386141873,), (0.15767843198426892,))
}


def load_dataset(dataset, path):
    if dataset == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(
            root=path,
            train=True,
            transform=cifar100_preprocessing(),
            download=True
        )
        testset = torchvision.datasets.CIFAR100(
            root=path,
            train=False,
            transform=cifar100_preprocessing(),
            download=True
        )
    else:
        raise ValueError

    return trainset, testset


def attack_dataloader(
        path,
        dataset,
        batchsize,
        local_data_size,
):

    if dataset != "FEMNIST":
        trainset, testset = load_dataset(dataset, path)
        trainset = torch.utils.data.Subset(trainset, np.arange(local_data_size * (len(trainset) // local_data_size)))
        train_sets = torch.utils.data.random_split(
            trainset,
            [local_data_size] * (len(trainset) // local_data_size)
        )
    else:
        train_data_dir = os.path.join(path, "femnist", 'data', 'train')
        test_data_dir = os.path.join(path, "femnist", 'data', 'test')
        _, _, train_data, test_data = read_data(train_data_dir, test_data_dir)
        train_keys = list(train_data.keys())
        train_keys.sort()
        test_keys = list(test_data.keys())
        test_keys.sort()
        train_sets = [
            TensorDataset(
                torch.stack(
                    [femnist_preprocessing()(x) for x in np.array(train_data[k]['x']).reshape(-1, 28, 28)]
                ).to(dtype=torch.float32)[:local_data_size],
                torch.Tensor(np.array(train_data[k]['y'])).to(dtype=torch.long)[:local_data_size]
            )
            for k in train_keys
        ]

    trainloaders = [
        DataLoader(d, batch_size=batchsize, shuffle=True) for d in train_sets
    ]

    return trainloaders, None, MEAN_STD[dataset]


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)
    data_dir = os.path.expanduser(data_dir)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data
