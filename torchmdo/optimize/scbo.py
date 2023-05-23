import torch
from numpy import ndarray, maximum
from logging import getLogger
from typing import Union, Optional, List, Tuple, cast, Dict, Callable
from warnings import warn, filterwarnings
from pdb import set_trace
from bdb import BdbQuit
from functools import cached_property
from .input_output import (
    DesignVariable,
    Output,
    Constraint,
    Minimize,
    Maximize,
    NearestFeasible,
)
from ..model import Model
from .optimizer import Optimizer
from dataclasses import dataclass, asdict
import math
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.exceptions.errors import BotorchError, ModelFittingError
from traceback import print_exc
import json

logger = getLogger(__name__)
Tensor = torch.Tensor
as_tensor = torch.as_tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
max_cholesky_size = float("inf")  # Always use Cholesky
filterwarnings("ignore")


@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = cast(int, None)  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(()) * torch.inf
    best_x: Tensor = cast(Tensor, None)
    restart_triggered: bool = False
    ftol: float = 1e-4

    def __post_init__(self):
        if self.failure_tolerance is None:
            self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
            )
        if self.best_x is not None:
            assert self.best_x.shape == (self.dim,)


def update_tr_length(state: ScboState) -> ScboState:
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


def update_state(
    state: ScboState, X_next: Tensor, Y_next: Tensor, C_next: Tensor
) -> ScboState:
    """Method used to update the TuRBO state after each step of optimization.

    Success and failure counters are updated according to the objective values 
    (Y_next) and constraint values (C_next) of the batch of candidate points 
    evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver any one of the 
    new candidate points improves upon the incumbent best point. The key difference 
    for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). If exactly one of the two points being compared 
    violates a constraint, the other valid point is automatically considered to be better. 
    If both points violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)
    """

    # Determine which candidates meet the constraints (are valid)
    bool_tensor = C_next <= 0
    bool_tensor = torch.all(bool_tensor, dim=-1)
    Valid_Y_next = Y_next[bool_tensor]
    Valid_C_next = C_next[bool_tensor]
    Valid_X_next = X_next[bool_tensor]
    if Valid_Y_next.numel() == 0:  # if none of the candidates are valid
        # pick the point with minimum sum of violated constraints
        sum_violation = torch.maximum(torch.zeros(()), C_next).sum(dim=-1)
        incumbent_violation = torch.maximum(
            torch.zeros(()), state.best_constraint_values
        ).sum()
        min_violation = sum_violation.min()
        # if the minimum voilation candidate is smaller than the violation of the incumbent
        # Note this will only be true if the incumbent does not meet all constraints
        if min_violation < incumbent_violation:
            # count a success and update the current best point and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            # new best is min violator
            """
            TODO: really we could choose any point from the pareto front of violation and objective,
            for instance, we could instead select to be the point with best objective that decreases the violation.
            Perhaps we could include a flag that would switch between these two choices.
            The suggested selection would favor finding a feasible point more slowly but
            losing as little performance as possible.
            """
            state.best_value = Y_next[sum_violation.argmin()].item()
            state.best_constraint_values = C_next[sum_violation.argmin()]
            state.best_x = X_next[sum_violation.argmin()]
        else:
            # otherwise, count a failure
            state.success_counter = 0
            state.failure_counter += 1
    else:  # if at least one valid candidate was suggested,
        # throw out all invalid candidates
        # (a valid candidate is always better than an invalid one)

        # Case 1: if the best valid candidate found has a higher objective value that
        # incumbent best count a success, the obj valuse has been improved
        # TODO: should probably still update the value even if improved by less than ftol. Just don't call it a success
        improved_obj = torch.max(Valid_Y_next) > state.best_value + state.ftol
        # Case 2: if incumbent best violates constraints
        # count a success, we now have suggested a point which is valid and thus better
        obtained_validity = torch.any(state.best_constraint_values > 0)
        if improved_obj or obtained_validity:  # If Case 1 or Case 2
            # count a success and update the best value and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = torch.max(Valid_Y_next).item()
            state.best_constraint_values = Valid_C_next[Valid_Y_next.argmax()]
            state.best_x = Valid_X_next[Valid_Y_next.argmax()]
        else:
            # otherwise, count a failure
            state.success_counter = 0
            state.failure_counter += 1

    # Finally, update the length of the trust region according to the
    # updated success and failure counters
    state = update_tr_length(state)
    return state


def get_initial_points(
    x0: Tensor, n_pts: int, scale: float, lb: Tensor, ub: Tensor, seed=0
) -> Tensor:
    """generate a set of random initial datapoints that we will use to kick-off optimization."""
    assert n_pts > 1
    x0 = x0.reshape((1, -1))
    assert torch.all(x0 >= lb.unsqueeze(0)), "initial point out of bounds"
    assert torch.all(x0 <= ub.unsqueeze(0)), "initial point out of bounds"
    # first generate samples in the range [-1, +1]
    sobol = SobolEngine(dimension=x0.shape[1], scramble=True, seed=seed)
    X_perturb = torch.vstack(
        [
            torch.zeros_like(x0),  # add a zero perturbation so x0 will be included
            2 * sobol.draw(n=n_pts - 1).to(dtype=dtype, device=device) - 1.0,
        ]
    )
    # now add the perturbation to x0 and scale
    X_init = x0 + scale * X_perturb
    # ensure all points are in bounds
    X_init = torch.clamp(X_init, lb.unsqueeze(0), ub.unsqueeze(0))
    return X_init


def generate_batch(
    state: ScboState,
    objective_model,  # GP model
    X: Tensor,  # Evaluated points on the domain [0, 1]^d
    Y: Tensor,  # Function values
    batch_size: int,
    n_candidates: int,  # Number of candidates for Thompson sampling
    constraint_model,
    sobol: SobolEngine,
    lb: Tensor,
    ub: Tensor,
) -> Tensor:
    """
    Generating a batch of candidates for SCBO

    Just as in the TuRBO Tutorial (https://botorch.org/tutorials/turbo_1), we'll define
    a method generate_batch to generate a new batch of candidate points within the TuRBO 
    trust region using Thompson sampling.

    The key difference here from TuRBO is that, instead of using MaxPosteriorSampling 
    to simply grab the candidates within the trust region with the maximum posterior 
    values, we use ConstrainedMaxPosteriorSampling to instead grab the candidates 
    within the trust region with the maximum posterior values subject to the constraint 
    that the posteriors for the constraint models must be less than 
    or equal to 0.

    We use additional GPs ('constraint models') to model each black-box constraint, 
    and throw out all candidates for which the sampled value for these 
    constraint models is greater than 0. In the special case when 
    all of the candidates are predicted to be constraint violators, we select the 
    candidate with the minimum predicted violation. (See 
    `botorch.generation.sampling.ConstrainedMaxPosteriorSampling` for implementation 
    details).
    """
    assert torch.all(torch.isfinite(Y))

    # Create the TR bounds
    x_center = state.best_x.clone()
    tr_lb = x_center - state.length / 2.0
    tr_ub = x_center + state.length / 2.0

    # enforce bounds
    assert torch.all(X >= lb.unsqueeze(0))
    assert torch.all(X <= ub.unsqueeze(0))
    tr_lb = torch.clamp(tr_lb, lb, ub)
    tr_ub = torch.clamp(tr_ub, lb, ub)

    # Thompson Sampling w/ Constraints (SCBO)
    dim = X.shape[-1]
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points using Constrained Max Posterior Sampling
    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model=objective_model, constraint_model=constraint_model, replacement=False
    )
    with torch.no_grad():
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

    return X_next


def get_fitted_model(X: Tensor, Y: Tensor):
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(
            nu=2.5, ard_num_dims=X.shape[1], lengthscale_constraint=Interval(0.005, 4.0)
        )
    )
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        fit_gpytorch_mll(mll)

    return model


def dump_state(state: ScboState, filename: str) -> None:
    """write state to disc in human-readable json format"""
    with torch.no_grad():
        # convert to dict
        state_dict = asdict(state)
        # convert tensors to lists
        for key, val in state_dict.items():
            if isinstance(val, Tensor):
                state_dict[key] = val.tolist()
        # write to disc
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(obj=state_dict, fp=file, ensure_ascii=False, indent=4)


def load_state(filename: str) -> ScboState:
    """read state json file from disc"""
    with torch.no_grad():
        # read from disc
        with open(filename, "r", encoding="utf-8") as file:
            state_dict = json.load(fp=file)
        # convert lists to tensors
        for key, val in state_dict.items():
            if isinstance(val, list):
                state_dict[key] = as_tensor(val)
        # reset counters
        state_dict["success_counter"] = 0
        state_dict["failure_counter"] = 0
        # reset the best value and best constraint values because we will recompute these
        state_dict["best_value"] = -float("inf")
        state_dict["best_constraint_values"] = torch.ones(()) * torch.inf
    return ScboState(**state_dict)


class SCBO(Optimizer):
    def constraint_fun(
        self, x: Union[ndarray, Tensor], jacobian_computation=False
    ) -> Tensor:
        """return the constrains such that valid is <= 0"""
        # get the raw constraint values
        constraints = super().constraint_fun(
            x=x, jacobian_computation=jacobian_computation
        )
        # transform so that <= 0 is valid
        constraint_lb, constraint_ub = self.constraint_bounds_tensor
        bool_lb = torch.isfinite(constraint_lb)
        bool_ub = torch.isfinite(constraint_ub)
        assert not torch.any(
            bool_lb == bool_ub
        ), "simultaneous upper and lower bounds are not supported."
        constraints_transformed = torch.zeros_like(constraints)
        constraints_transformed[bool_lb] = constraint_lb[bool_lb] - constraints[bool_lb]
        constraints_transformed[bool_ub] = constraints[bool_ub] - constraint_ub[bool_ub]
        return constraints_transformed

    def optimize(
        self,
        batch_size: int = 50,
        initial_doe_size: int = 50,
        initial_scale=1.0,
        max_gp_size=512,
        max_candidates: int = 1000,
        seed=0,
        state_init: Union[None, ScboState, str] = None,
        state_dump: Optional[str] = None,
    ) -> ScboState:
        torch.manual_seed(seed)
        # compute the current iterate first. This is nessessary since we need to know
        # the dimensions of all the outputs.
        self.compute()
        x0 = self.variables_tensor.detach()
        dim = x0.numel()
        if state_init is None:
            state = ScboState(dim=dim, batch_size=batch_size)
        else:
            if isinstance(state_init, str):  # load from disc
                state_init = load_state(filename=state_init)
            assert state_init.dim == dim
            assert state_init.batch_size == batch_size
            state = state_init
            x0 = (
                state.best_x.clone().detach()
            )  # over-ride x0 with the best value so far
        lb, ub = self.variable_bounds_tensor
        # specify number of candidates to use
        # SCBO actually uses min(5000, max(2000, 200 * dim)) candidate points by default.
        N_CANDIDATES = min(max_candidates, max(2000, 200 * dim))
        sobol = SobolEngine(dim, scramble=True, seed=seed)

        # start the iterations
        train_X, train_Y, train_C = None, None, None
        num_func_evals = 0
        try:
            while not state.restart_triggered:  # Run until converges
                if train_X is None or train_Y is None or train_C is None:
                    # generate initial candidates
                    logger.info(
                        "Performing initial DOE with %d evaluations in %d dims."
                        % (initial_doe_size, dim)
                    )
                    X_next = get_initial_points(
                        x0=x0, n_pts=initial_doe_size, scale=initial_scale, lb=lb, ub=ub
                    )
                else:
                    # Fit GP models for objective and constraints
                    objective_model = get_fitted_model(train_X, train_Y)
                    constraint_models = [
                        get_fitted_model(train_X, C.unsqueeze(-1)) for C in train_C.T
                    ]

                    # Generate a batch of candidates
                    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                        X_next = generate_batch(
                            state=state,
                            objective_model=objective_model,
                            X=train_X,
                            Y=train_Y,
                            batch_size=batch_size,
                            n_candidates=N_CANDIDATES,
                            constraint_model=ModelListGP(*constraint_models),
                            sobol=sobol,
                            lb=lb,
                            ub=ub,
                        )

                # Evaluate both the objective and constraints for the selected candidaates
                with torch.no_grad():
                    Y_next = cast(List[Tensor], [])
                    C_next = cast(List[Tensor], [])
                    success_indices = cast(List[int], [])
                    for i, x in enumerate(X_next):
                        try:
                            Y_next.append(
                                -self.objective_fun(x)
                            )  # neg since maximizing
                            C_next.append(self.constraint_fun(x))
                        except KeyboardInterrupt:
                            raise
                        except:
                            logger.error("Objective or constraint failed.")
                            print_exc()
                            # make sure that neither objective or constraint were added
                            Y_next = Y_next[:i]
                            C_next = C_next[:i]
                        else:
                            success_indices.append(i)
                    if len(success_indices) < X_next.shape[0]:
                        # then there was a failed run
                        assert (
                            len(success_indices) > 0
                        ), "no points in the batch succeeded"
                        X_next = X_next[success_indices]
                        assert X_next.shape[0] == len(Y_next) == len(C_next)

                    Y_next = torch.tensor(Y_next, dtype=dtype, device=device).unsqueeze(
                        -1
                    )
                    C_next = torch.stack(C_next, dim=0)

                # Update state
                num_func_evals += X_next.shape[0]
                state = update_state(
                    state=state, X_next=X_next, Y_next=Y_next, C_next=C_next
                )

                # Append data. Note that we append all data, even points that violate
                # the constraints. This is so our constraint models can learn more
                # about the constraint functions and gain confidence in where violations occur.
                if train_X is None or train_Y is None or train_C is None:
                    train_X = X_next
                    train_Y = Y_next
                    train_C = C_next
                else:
                    train_X = torch.cat((train_X, X_next), dim=0)
                    train_Y = torch.cat((train_Y, Y_next), dim=0)
                    train_C = torch.cat((train_C, C_next), dim=0)

                # reduce the size of the training dataset if becoming too large
                if train_X.shape[0] > max_gp_size:
                    # discard the oldest data
                    train_X = train_X[-max_gp_size:]
                    train_Y = train_Y[-max_gp_size:]
                    train_C = train_C[-max_gp_size:]
                # run a couple of testing sanity checks
                assert train_X.shape[0] <= max_gp_size, "dataset size exceeds limit"
                assert torch.all(train_X[-1] == X_next[-1]), "most recent data lost"

                # Print current status. Note that state.best_value is always the best
                # objective value found so far which meets the constraints, or in the case
                # that no points have been found yet which meet the constraints, it is the
                # objective value of the point with the minimum constraint violation.
                logger.info(
                    f"num_eval: {num_func_evals:d}, "
                    f"fun: {state.best_value:.6g}, "
                    f"constr_violation: {state.best_constraint_values.max():.2g}, "
                    f"tr_length: {state.length:.2g}"
                )

                # save state to file
                if state_dump is not None:
                    dump_state(state=state, filename=state_dump)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt raised. Cleaning up...")
            self.reset_state()  # reset because computation could have ended prematurely
        except (BotorchError, ModelFittingError):
            logger.error("Botorch or GP error during optimization.")
            print_exc()
            logger.info("Trying to clean up and exit...")
        except BdbQuit:
            raise  # this is just the debugger exiting
        except:
            logger.error("Unknown error during optimization!")
            print_exc()
            set_trace()
            logger.info("Trying to clean up and exit...")
            self.reset_state()  # reset because we don't know what went wrong
        else:
            logger.info("Optimization completed.")
        assert isinstance(state.best_x, Tensor), "no best point found yet"
        # set the parameters internally
        self.variables_tensor = state.best_x.clone().detach()
        return state
