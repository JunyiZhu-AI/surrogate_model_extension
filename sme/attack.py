import copy
import random
import torch
import os
import json
from models import *
from sme.Adversary import IWU
from utils import attack_dataloader, random_seed


def attack(
        path_to_data,
        path_to_res,
        dataset,
        model,
        seed,
        batchsize,
        train_lr,
        epochs,
        alpha,
        eta,
        iters,
        lamb,
        lr_decay,
        beta,
        test_steps,
        k,
):
    os.makedirs(path_to_res, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup = {"device": device, "dtype": torch.float32}
    random_seed(seed)

    # Prepare dataset
    trainloaders, _, mean_std = attack_dataloader(
        path=path_to_data,
        dataset=dataset,
        batchsize=batchsize,
        local_data_size=k,
    )
    # Initialize the network
    if dataset == "CIFAR100":
        classes = 100
    elif dataset == "FEMNIST":
        classes = 62
    else:
        raise ValueError
    net = eval(f"{model}(classes={classes})").to(**setup)

    # Initialize the adversary
    trainloader = random.choice(trainloaders)
    adversary = IWU(
        trainloader=trainloader,
        setup=setup,
        alpha=alpha,
        test_steps=test_steps,
        path_to_res=path_to_res,
        lamb=lamb,
        mean_std=mean_std,
        dataset=dataset,
    )

    # Victim trains local model
    with torch.no_grad():
        net1 = copy.deepcopy(net)
    adversary.net0 = net
    train(
        net=net1,
        trainloader=trainloader,
        epochs=epochs,
        train_lr=train_lr,
        setup=setup
    )
    adversary.net1 = net1

    # Reconstruction
    stats = adversary.reconstruction(
        eta=eta,
        beta=beta,
        iters=iters,
        lr_decay=lr_decay,
        save_figure=True,
    )
    with open(os.path.join(path_to_res, "res.json"), "w") as f:
        json.dump(stats, f, indent=4)


def train(
        net,
        trainloader,
        epochs,
        train_lr,
        setup,
):
    # In evaluation mode, updates to the running statistics of Batch Normalization (if applicable) are halted.
    # This practice follows the work of IG. For more details, please refer to our paper.
    net.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(
        params=net.parameters(),
        lr=train_lr
    )
    for _ in range(epochs):
        for data, label in trainloader:
            optimizer.zero_grad()
            data = data.to(**setup)
            label = label.to(device=setup["device"])

            pred = net(data)
            loss = criterion(input=pred, target=label)

            loss.backward()
            optimizer.step()

