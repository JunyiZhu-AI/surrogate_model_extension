import torch
import os
import copy
from utils import psnr, save_figs
from sme.Adversary.utils import *


class IWU:
    def __init__(
            self,
            trainloader,
            setup,
            alpha,
            test_steps,
            path_to_res,
            mean_std,
            lamb,
            dataset=None,
    ):
        self.alpha = torch.tensor(alpha, requires_grad=True, **setup)
        self.rec_alpha = 0 < self.alpha < 1
        self.setup = setup
        self.net0 = None
        self.net1 = None
        self.test_steps = test_steps
        os.makedirs(path_to_res, exist_ok=True)
        self.path = path_to_res
        self.lamb = lamb
        self.dataset = dataset
        data, labels = [], []
        for img, l in trainloader:
            labels.append(l)
            data.append(img)
        self.data = torch.cat(data).to(**setup)

        # We assume that labels have been restored separately, for details please refer to the paper.
        self.y = torch.cat(labels).to(device=setup["device"])
        # Dummy input.
        self.x = torch.normal(0, 1, size=self.data.shape, requires_grad=True, **setup)

        self.mean = torch.tensor(mean_std[0]).to(**setup).reshape(1, -1, 1, 1)
        self.std = torch.tensor(mean_std[1]).to(**setup).reshape(1, -1, 1, 1)
        # This is a trick (a sort of prior information) adopted from IG.
        prior_boundary(self.x, -self.mean / self.std, (1 - self.mean) / self.std)

    def reconstruction(self, eta, beta, iters, lr_decay, signed_grad=False, save_figure=True):
        # when taking the SME strategy, alpha is set within (0, 1).
        if 0 < self.alpha < 1:
            self.alpha.grad = torch.tensor(0.).to(**self.setup)
        optimizer = torch.optim.Adam(params=[self.x], lr=eta)
        alpha_opti = torch.optim.Adam(params=[self.alpha], lr=beta)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[iters // 2.667,
                                                                     iters // 1.6,
                                                                     iters // 1.142],
                                                         gamma=0.1)
        alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(alpha_opti,
                                                               milestones=[iters // 2.667,
                                                                           iters // 1.6,
                                                                           iters // 1.142],
                                                               gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        # Direction of the weight update.
        w1_w0 = []
        for p0, p1 in zip(self.net0.parameters(), self.net1.parameters()):
            w1_w0.append(p0.data - p1.data)
        norm = compute_norm(w1_w0)
        w1_w0 = [p / norm for p in w1_w0]

        # Construct the model for gradient inversion attack.
        require_grad(self.net0, False)
        require_grad(self.net1, False)
        with torch.no_grad():
            _net = copy.deepcopy(self.net0)
            for p, q, z in zip(self.net0.parameters(), self.net1.parameters(), _net.parameters()):
                z.data = (1 - self.alpha) * p + self.alpha * q

        # Reconstruction
        _net.eval()
        stats = []
        for i in range(iters):
            optimizer.zero_grad()
            alpha_opti.zero_grad(set_to_none=False)
            _net.zero_grad()

            if self.rec_alpha:
                # Update the surrogate model.
                with torch.no_grad():
                    for p, q, z in zip(self.net0.parameters(), self.net1.parameters(), _net.parameters()):
                        z.data = (1 - self.alpha) * p + self.alpha * q
            pred = _net(self.x)
            loss = criterion(input=pred, target=self.y)
            grad = torch.autograd.grad(loss, _net.parameters(), create_graph=True)
            norm = compute_norm(grad)
            grad = [p / norm for p in grad]

            # Compute x's grad.
            cos_loss = 1 - sum([
                p.mul(q).sum() for p, q in zip(w1_w0, grad)
            ])
            loss = cos_loss + self.lamb * total_variation(self.x)
            loss.backward()
            if signed_grad:
                self.x.grad.sign_()

            # Compute alpha's grad.
            if self.rec_alpha:
                with torch.no_grad():
                    for p, q, z in zip(self.net0.parameters(), self.net1.parameters(), _net.parameters()):
                        self.alpha.grad += z.grad.mul(
                            q.data - p.data
                        ).sum()
                if signed_grad:
                    self.alpha.grad.sign_()

            # Update x and alpha.
            optimizer.step()
            alpha_opti.step()
            prior_boundary(self.x, -self.mean / self.std, (1 - self.mean) / self.std)
            prior_boundary(self.alpha, 0, 1)
            if lr_decay:
                scheduler.step()
                alpha_scheduler.step()
            if i % self.test_steps == 0 or i == iters - 1:
                with torch.no_grad():
                    _x = self.x * self.std + self.mean
                    _data = self.data * self.std + self.mean
                measurement = psnr(_data, _x, sort=True)
                print(f"iter: {i}| alpha: {self.alpha.item():.2f}| (1 - cos): {cos_loss.item():.3f}| "
                      f"psnr: {measurement:.3f}")
                stats.append({
                    "iter": i,
                    "alpha": self.alpha.item(),
                    "cos_loss": cos_loss.item(),
                    "psnr": measurement,
                })
                if save_figure:
                    save_figs(tensors=_x, path=self.path, subdir=str(i), dataset=self.dataset)
        if save_figure:
            save_figs(tensors=self.data * self.std + self.mean,
                      path=self.path, subdir="original", dataset=self.dataset)
        return stats
