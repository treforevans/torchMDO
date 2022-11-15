import torch
import gpytorch
from torch_interpolations import RegularGridInterpolator
from pdb import set_trace
from typing import Optional

Tensor = torch.Tensor


class InterpND(gpytorch.models.ExactGP):
    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        differentiability: int = 1,
        bounds_error: bool = True,
        num_optim_iters: Optional[int] = None,
    ):
        # some checks
        assert x.ndim == y.ndim == 2
        assert x.shape[0] == y.shape[0]
        assert differentiability in [1, 2, 3]
        self.n_inputs = x.shape[1]
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
            noise_constraint=gpytorch.constraints.Interval(
                lower_bound=1e-6, upper_bound=1e-5
            ),
            num_tasks=self.n_functions,
            has_task_noise=False,  # ensure it only has a single noise term
        )
        likelihood.noise = 0.99e-5  # type:ignore
        # initialize the model
        super().__init__(train_inputs=x, train_targets=y, likelihood=likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.n_functions])
        )
        self.covar_module = gpytorch.kernels.MaternKernel(
            nu=differentiability - 0.5, batch_shape=torch.Size([self.n_functions])
        )
        # perform hyperparam optimiztion if nessessary
        if num_optim_iters is not None:
            self.optimize_hyperparameters(x=x, y=y, num_iterations=num_optim_iters)
        # Get into evaluation (predictive posterior) mode
        self.eval()
        self.likelihood.eval()

    def scale_inputs(self, x: Tensor) -> Tensor:
        """ scale the inputs to the range [0, 1] """
        return (x - self.x_min) / self.x_ptp

    def forward(self, x: Tensor):
        assert x.ndim == 2
        assert x.shape[1] == self.n_inputs
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

    def optimize_hyperparameters(
        self, x: Tensor, y: Tensor, num_iterations: int = 100
    ) -> None:
        """ Find optimal model hyperparameters """
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(num_iterations):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.forward(x)
            # Calc loss and backprop gradients
            loss = -mll(output, y)
            loss.backward()
            print(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3g"
                % (
                    i + 1,
                    num_iterations,
                    loss.item(),
                    self.covar_module.lengthscale.item(),
                    self.likelihood.noise.item(),
                )
            )
            optimizer.step()


class Interp1D(InterpND):
    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        differentiability: int = 1,
        bounds_error: bool = True,
    ):
        assert x.shape[1] == 1
        self.differentiability = differentiability
        if differentiability != 1:
            super().__init__(
                x=x, y=y, differentiability=differentiability, bounds_error=bounds_error
            )
        else:
            # we are performing linear interpolation in 1D
            # some checks
            assert x.ndim == y.ndim == 2
            assert x.shape[0] == y.shape[0]
            self.n_inputs = 1
            # create a list of interpolators, one for each output in y
            # TODO: computations would be far faster if we rewrite this to did the interpolation in one call
            self.interp_list = [
                RegularGridInterpolator(points=[x.squeeze(dim=1)], values=yi,)
                for yi in y.T
            ]
            self.x_min = x.min()
            self.x_max = x.max()

    def forward(self, x: Tensor):
        if self.differentiability != 1:
            return super().forward(x=x)
        else:
            if x.ndim == 1:
                x = x.unsqueeze(dim=1)
            else:
                assert x.ndim == 2
            assert x.shape[1] == self.n_inputs
            # make sure we're not out of bounds
            if x.max() > self.x_max:
                raise ValueError("Trying to interpolate above the data bounds.")
            elif x.min() < self.x_min:
                raise ValueError("Trying to interpolate below the data bounds.")
            # now make the prediction
            return torch.cat(
                [
                    interp([x.squeeze(dim=1)]).unsqueeze(dim=1)
                    for interp in self.interp_list
                ],
                dim=1,
            )

    def __call__(self, x: Tensor) -> Tensor:
        """
        only return the posterior mean if we are using a GP.
        """
        if self.differentiability != 1:
            return super().__call__(x)
        else:
            return self.forward(x)
