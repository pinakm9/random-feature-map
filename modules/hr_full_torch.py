import os, sys
from pathlib import Path
from os.path import dirname, realpath

script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')

import torch
import matplotlib.pyplot as plt
import utility as ut
from joblib import Parallel, delayed


class InequalitySampler3:
    """
    A class for sampling from w · x_+ + b < l_plus and w · x_- + b > l_minus
    """
    def __init__(self, x_pm, l_minus, l_plus, dim) -> None:
        self.x_pm = x_pm
        self.l_minus = l_minus
        self.l_plus = l_plus
        self.dim = dim
        self.int_pts = []
        self.weights = torch.tensor([])

    def is_feasible(self, x: torch.Tensor) -> bool:
        w = x[:-1]
        b = x[-1]
        x_plus = self.x_pm(torch.sign(w), True)
        x_minus = self.x_pm(torch.sign(w), False)
        val_plus = x_plus.dot(w) + b
        val_minus = x_minus.dot(w) + b
        return (val_plus < self.l_plus).item() and (val_minus > self.l_minus).item()

    def intersection_with_bisection(self, x0: torch.Tensor, d: torch.Tensor,
                                   tol: float = 1e-2, max_iters: int = 100) -> int:
        self.x0 = x0.clone()
        if self.is_feasible(self.x0):
            self.weights = torch.zeros(1)
            t = 1.0
            # expand until outside feasible
            while self.is_feasible(self.x0 + t * d):
                t *= 2

            l, r = 0.0, t
            itr = 0
            while abs(r - l) > tol and itr < max_iters:
                m = (r + l) / 2.0
                if not self.is_feasible(self.x0 + m * d):
                    r = m
                else:
                    l = m
                itr += 1

            self.weights[0] = l
            return 1
        else:
            self.int_pts = []
            self.weights = torch.tensor([])
            return 0

    def single_sample(self, x: torch.Tensor, steps: int = 10,
                      tol: float = 1e-2, max_iters: int = 100) -> torch.Tensor:
        x = x.clone()
        for _ in range(steps):
            d = torch.randn(self.dim + 1)
            d /= d.norm()
            # ensure feasible step
            while self.intersection_with_bisection(x, d, tol=tol, max_iters=max_iters) == 0:
                d = torch.randn(self.dim + 1)
                d /= d.norm()
            t = torch.rand(1).item() * self.weights[0].item()
            x = x + t * d
        return x


class GoodRowSampler:
    """
    A class for sampling a row R such that m < |R · x + b| < M for x in C (a convex set)
    """
    def __init__(self, m: float, M: float, data) -> None:
        self.m = m
        self.M = M
        self.data = torch.tensor(data, dtype=torch.float32)
        self.dim = self.data.shape[-1]
        mins, maxs = self.data.min(dim=0).values, self.data.max(dim=0).values
        # shape [dim, 2]: [min, max] for each feature
        self.lims = torch.stack((mins, maxs), dim=1)

    def get_vector(self, signs, option: bool) -> torch.Tensor:
        # convert signs to python list of 0/1
        if isinstance(signs, torch.Tensor):
            signs_list = signs.tolist()
        else:
            signs_list = list(signs)
        bits = [0 if s < 0 else 1 for s in signs_list]
        if option:
            vec = [self.lims[d, bits[d]].item() for d in range(self.dim)]
        else:
            vec = [self.lims[d, (bits[d] + 1) % 2].item() for d in range(self.dim)]
        return torch.tensor(vec, dtype=torch.float32)

    def sample_(self, steps: int = 10):
        flag = torch.randint(0, 2, (1,)).item()
        s = torch.randint(0, 2, (self.dim,))
        if flag:
            lims = (self.m, self.M)
        else:
            lims = (-self.M, -self.m)
        b = torch.empty(1).uniform_(lims[0], lims[1]).item()
        sampler = InequalitySampler3(self.get_vector, lims[0], lims[1], self.dim)
        x0 = torch.cat((torch.zeros(self.dim), torch.tensor([b])))
        wb = sampler.single_sample(x=x0, steps=steps)
        return wb[:-1], wb[-1]

    def sample(self, n_sample: int, steps: int = 10):
        rows = torch.zeros((n_sample, self.dim), dtype=torch.float32)
        bs = torch.zeros(n_sample, dtype=torch.float32)
        for n in range(n_sample):
            row, b = self.sample_(steps)
            rows[n] = row
            bs[n] = b
        return rows, bs

    def sample_parallel(self, n_sample: int):
        results = Parallel(n_jobs=-1)(delayed(self.sample_)() for _ in range(n_sample))
        rows = torch.vstack([res[0] for res in results])
        bs = torch.tensor([res[1] for res in results], dtype=torch.float32)
        return rows, bs

