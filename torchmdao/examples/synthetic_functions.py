import torch
from typing import Optional

Tensor = torch.Tensor


class Rosenbrock:
    r"""Rosenbrock synthetic test function.

    d-dimensional function (usually evaluated on `[-5, 10]^d`):

        f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_i) = 0.0`.
    """

    optimal_value = 0.0

    def __init__(self, dim=2) -> None:
        self.dim = dim
        self.bounds = [(-5.0, 10.0) for _ in range(self.dim)]
        self.minimizer = torch.ones(self.dim)

    def __call__(self, X: Tensor) -> Tensor:
        return torch.sum(
            100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2,
            dim=-1,
        )
