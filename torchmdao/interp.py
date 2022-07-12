import torch
import gpytorch
from pdb import set_trace

Tensor = torch.Tensor


class Interp1d(gpytorch.models.ExactGP):
    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        differentiability: int = 1,
        bounds_error: bool = True,
    ):
        # some checks
        assert x.ndim == y.ndim == 2
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == 1
        assert differentiability in [1, 2, 3]
        self.n_functions = y.shape[1]
        self.bounds_error = bounds_error
        # save the min and max values so we can make sure we don't extrapolate
        self.x_min = torch.min(x, dim=0, keepdim=True)[0]
        self.x_max = torch.max(x, dim=0, keepdim=True)[0]
        self.x_ptp = torch.maximum(  # for safe division
            self.x_max - self.x_min, torch.as_tensor(1e-5)
        )
        # define likelihood
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
            num_tasks=self.n_functions,
            has_task_noise=False,  # ensure it only has a single noise term
        )
        likelihood.noise = 1e-5  # type:ignore
        # initialize the model
        super().__init__(train_inputs=x, train_targets=y, likelihood=likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.n_functions])
        )
        self.covar_module = gpytorch.kernels.MaternKernel(
            nu=differentiability - 0.5, batch_shape=torch.Size([self.n_functions])
        )
        # Get into evaluation (predictive posterior) mode
        self.eval()
        self.likelihood.eval()

    def scale_inputs(self, x: Tensor) -> Tensor:
        return (x - self.x_min) / self.x_ptp

    def forward(self, x: Tensor):
        assert x.ndim == 2
        assert x.shape[1] == 1
        x_scaled = self.scale_inputs(x)
        # make sure we're not out of bounds
        if any(x_scaled.max(dim=0)[0] > 1.0):
            raise ValueError("Trying to interpolate above the data bounds.")
        elif any(x_scaled.min(dim=0)[0] < 0.0):
            raise ValueError("Trying to interpolate below the data bounds.")
        # now make the prediction
        mean_x = self.mean_module(x_scaled)
        covar_x = self.covar_module(x_scaled)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

    def __call__(self, x: Tensor) -> Tensor:
        """
        only return the posterior mean.
        """
        return super().__call__(x).mean
