import torch
import torchvision.transforms as transformers
from torchvision.transforms import ToPILImage
from scipy.optimize import linear_sum_assignment
import random
import matplotlib.pyplot as plt
import os
import json
import numpy as np

__all__ = [
    "random_seed",
    "cifar100_preprocessing",
    "femnist_preprocessing",
    "psnr",
    "save_args",
    "save_figs",
]


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def cifar100_preprocessing():
    transform = transformers.Compose([
        transformers.ToTensor(),
        transformers.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    return transform

def femnist_preprocessing():
    transform = transformers.Compose([
        transformers.ToTensor(),
        transformers.Normalize((0.9642384386141873,), (0.15767843198426892,))
    ])
    return transform


def psnr(data, rec, sort=False):
    assert data.max().item() <= 1.0001 and data.min().item() >= -0.0001
    assert rec.max().item() <= 1.0001 and rec.min().item() >= -0.0001
    cost_matrix = []
    if sort:
        for x_ in rec:
            cost_matrix.append(
                [(x_ - d).square().mean().item() for d in data]
            )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assert np.all(row_ind == np.arange(len(row_ind)))
        data = data[col_ind]
    psnr_list = [10 * np.log10(1 / (d - r).square().mean().item()) for d, r in zip(data, rec)]
    return np.mean(psnr_list)


def save_args(**kwargs):
    if os.path.exists(os.path.join(kwargs["path_to_res"], "args.json")):
        os.remove(os.path.join(kwargs["path_to_res"], "args.json"))

    with open(os.path.join(kwargs["path_to_res"], "args.json"), "w") as f:
        json.dump(kwargs, f, indent=4)


def save_figs(tensors, path, subdir=None, dataset=None):
    def save(imgs, path):
        for name, im in imgs:
            plt.figure()
            plt.imshow(im, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(path, f'{name}.png'), bbox_inches='tight')
            plt.close()
    tensor2image = ToPILImage()
    path = os.path.join(path, subdir)
    os.makedirs(path, exist_ok=True)
    if dataset == "FEMNIST":
        tensors = 1 - tensors
    imgs = [
        [i, tensor2image(tensors[i].detach().cpu().squeeze())] for i in range(len(tensors))
    ]
    save(imgs, path)
