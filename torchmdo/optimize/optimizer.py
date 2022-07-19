import torch
from numpy import ndarray
from scipy.optimize import (
    minimize,
    NonlinearConstraint,
    OptimizeResult,
    check_grad,
)
from scipy.sparse import csr_matrix
from logging import getLogger
from typing import Union, Optional, List, Tuple, cast, Dict
from warnings import warn
from pdb import set_trace
from functools import cached_property
from .input_output import (
    DesignVariable,
    Output,
    Constraint,
    Objective,
    Minimize,
    Maximize,
)
from ..model import ComputeObject

logger = getLogger(__name__)
Tensor = torch.Tensor
as_tensor = torch.as_tensor


class Optimizer:
    """
    Object used to perform non-linear constrained optimization.

    Args:
        initial_design_variables: list of design variables to be used in the 
            optimization problem.
        objective: objective to be used in the optimization problem.
        constraints: list of constraints to be used in the optimization problem.
        compute_object: object that computes all outputs 
            when `compute_object.compute()` is called.
        vectorize_constraint_jac: if true then the computation of the constraint 
            jacobian will be vectorized which may help a lot when many 
            constraints are present. Note, however, that this is an experimental 
            feature. Default: `False`.
    """

    def __init__(
        self,
        initial_design_variables: List[DesignVariable],
        objective: Union[Minimize, Maximize],
        constraints: List[Constraint],
        compute_object: ComputeObject,
        vectorize_constraint_jac: bool = False,
    ):
        self.vectorize_constraint_jac = bool(vectorize_constraint_jac)
        # check compute_object
        assert isinstance(compute_object, ComputeObject)
        self.compute_object = compute_object
        # check outputs
        assert isinstance(objective, (Minimize, Maximize))
        for out in constraints:
            assert isinstance(out, Constraint)
        self.outputs = cast(List[Output], [objective, *constraints])
        self.objective_index = 0  # is always the first one
        # check to make sure that the compute object has all of the design variables attributes
        for idv in initial_design_variables:
            assert isinstance(idv, DesignVariable)
            if not hasattr(self.compute_object, idv.name):
                raise KeyError(
                    "Name '%s' of the design variable does not match an attribute in the ComputeObject"
                    % idv.name
                )
            # if the initial value of the design variable is not set then extract it
            # from the compute object
            if idv.value is None:
                assert getattr(self.compute_object, idv.name) is not None, (
                    "No initial value found for design variable '%s'" % idv.name
                )
                idv.extract_val(compute_object=self.compute_object)
        self.initial_design_variables = initial_design_variables
        # set parameters in the compute object to the initial design variables
        self.variables_object = self.initial_design_variables
        # do a sanity check to make sure the reconstructed desgin variables are the same
        for (original, reconstructed) in zip(
            self.initial_design_variables, self.variables_object
        ):
            try:
                assert original.name == reconstructed.name
                assert torch.all(original.value_tensor == reconstructed.value_tensor)
            except:
                logger.error(
                    "design variable internalization failed for %s." % original.name
                )
                raise
        # initialize an tensor to save the variable values from the last iteration so we
        # can check whether or not they have changed
        self._last_variables = None

    @property
    def variables_object(self) -> List[DesignVariable]:
        """
        create a copy of the initial design variables with a change made to the value
        """
        design_variables = [
            idv.replace(value=as_tensor(getattr(self.compute_object, idv.name)))
            for idv in self.initial_design_variables
        ]
        return design_variables

    @variables_object.setter
    def variables_object(self, design_variables: List[DesignVariable]) -> None:
        """
        Set the design variables to the specified values in the ComputeObject.
        """
        for dv in design_variables:
            # make sure design variables are of the correct type
            assert isinstance(dv, DesignVariable)
            setattr(self.compute_object, dv.name, dv.value_tensor)
        # reset state since variables changed
        self.reset_state()

    @property
    def variables_tensor(self) -> Tensor:
        """
        returns the variables as a 1d tensor
        """
        return torch.cat([torch.ravel(dv.value_tensor) for dv in self.variables_object])

    @variables_tensor.setter
    def variables_tensor(self, value: Tensor):
        """
        setter for variables_tensor property
        """
        if (
            self._last_variables is None
            or torch.any(value != self._last_variables)
            or (not self._last_variables.requires_grad and value.requires_grad)
        ):
            # if any value changed, or if the previous computation was done without
            # `requires_grad` and it is now required,
            # then update the compute object with the new variables
            i_cur = 0  # current variable index
            for idv in self.initial_design_variables:
                setattr(
                    self.compute_object,
                    idv.name,
                    torch.reshape(value[i_cur : (i_cur + idv.numel)], idv.shape),
                )
                i_cur += idv.numel  # update the position of the index
            assert (
                i_cur == value.numel()
            ), "sanity check: did not use all design variables"
            # reset the state since the variables have changed
            self.reset_state()
            # save parameters for next iteration
            # NOTE: we do not detach this attribute since we will need to gradients to
            # flow through.
            self._last_variables = value
        else:
            pass  # nothing changed so pass

    @property
    def variable_bounds_tensor(self) -> Tuple[Tensor, Tensor]:
        """
        get bounds for all optimization variables
        """
        lower = torch.cat(
            [idv.lower_tensor.reshape(-1) for idv in self.initial_design_variables]
        )
        upper = torch.cat(
            [idv.upper_tensor.reshape(-1) for idv in self.initial_design_variables]
        )
        return lower, upper

    @property
    def constraint_bounds_tensor(self) -> Tuple[Tensor, Tensor]:
        """
        get bounds for all constrained outputs
        """
        lower = torch.cat([out.lower_tensor.reshape(-1) for out in self.outputs])
        upper = torch.cat([out.upper_tensor.reshape(-1) for out in self.outputs])
        return lower[self.constrained_output_mask], upper[self.constrained_output_mask]

    @cached_property
    def constrained_output_mask(self) -> Tensor:
        """
        Get a boolean tensor indicating which outputs are constrained.
        It is cached since this will never change throughout an optimization.
        """
        # get the lower and upper bounds for the ouputs
        lower = torch.cat([out.lower_tensor.reshape(-1) for out in self.outputs])
        upper = torch.cat([out.upper_tensor.reshape(-1) for out in self.outputs])
        # determine which are constrained (has a finite lower and/or upper bound)
        return torch.logical_or(torch.isfinite(lower), torch.isfinite(upper))

    @cached_property
    def num_constraints(self) -> int:
        """ Returns the number of constraints present. """
        return int(torch.count_nonzero(self.constrained_output_mask))

    @cached_property
    def constraints_are_all_linear(self) -> bool:
        """ Returns true if all the constraints are linear. """
        return all(out.linear for out in self.outputs if out.is_constrained)

    def reset_state(self):
        """ resets outputs and internal state """
        # remove all output values
        for out in self.outputs:
            out.value = None
        # reset the _last_variables tensor
        self._last_variables = None

    def compute(self, **kwargs) -> None:
        """
        run compute to get all outputs (objective and constraints) for the current
        design variable setting.
        """
        # check if the computation has already been completed
        if self.outputs[self.objective_index].value is not None:
            # if the objective is already computed then don't need to recompute anything
            return
        else:
            # run the objective and constraint functions
            self.compute_object.compute(**kwargs)

            # extract the outputs
            for out in self.outputs:
                out.extract_val(compute_object=self.compute_object)

    def objective_fun(self, x: Union[ndarray, Tensor]) -> Tensor:
        """ return the objective """
        x = as_tensor(x)
        x.requires_grad = True
        self.variables_tensor = x
        self.compute()
        objective = self.outputs[self.objective_index].value_tensor
        # run a couple of quick checks
        assert objective.numel() == 1, "objective must be a scalar"
        assert torch.all(
            torch.isfinite(objective)
        ), "Non-finite value encountered in the objective = %s" % str(
            objective.detach()
        )
        # determine if we need to negate the objective
        if isinstance(self.outputs[self.objective_index], Maximize):
            objective = -objective  # negate
        # return the objective and detach it so we can convert it to numpy
        return objective.reshape(()).detach()

    def objective_grad(self, x: Union[ndarray, Tensor]) -> Tensor:
        """
        Compute the gradient of the objective with respect to the input x
        """
        x = as_tensor(x)
        x.requires_grad = True
        self.variables_tensor = x
        self.compute()
        # extract the objective
        objective = self.outputs[self.objective_index].value_tensor
        # compute the objective gradient
        # NOTE: the gradient is actually with respect to _last_variables which may
        # or may not be the same tensor as x depending on whether `objective_fun`
        # was called first at the same point.
        objective.backward()
        assert self._last_variables is not None, "sanity check"
        objective_grad = self._last_variables.grad
        assert (
            objective_grad is not None
        ), "objective gradient failed, something wrong with comptuational graph."
        assert torch.all(torch.isfinite(objective_grad)), (
            "Non-finite value encountered in the objective gradient = %s"
            % str(objective_grad.detach())
        )
        # determine if we need to negate the objective
        if isinstance(self.outputs[self.objective_index], Maximize):
            objective_grad = -objective_grad  # negate
        return objective_grad

    def constraint_fun(
        self, x: Union[ndarray, Tensor], jacobian_computation=False
    ) -> Tensor:
        """ return the constraints """
        x = as_tensor(x)
        if jacobian_computation:
            # explicitly reset the state whether we have computed at this x or not
            # this ensures that all computations are performed
            self.reset_state()
        else:
            x.requires_grad = True
        # set the values internally to x
        self.variables_tensor = x
        self.compute()
        # get all the outputs as a concatenated 1d vector
        all_outputs = torch.cat([out.value_tensor.reshape(-1) for out in self.outputs])
        # extract any constrained outputs
        constraints = all_outputs[self.constrained_output_mask]
        # run a quick check
        assert torch.all(torch.isfinite(constraints)), (
            "Non-finite value encountered in the constraints = %s"
            % str(constraints.detach())
        )
        # return, determining whether to detach or not based on whether the jacobian
        # is being computed
        return constraints if jacobian_computation else constraints.detach()

    def constraint_jac(self, x: Union[ndarray, Tensor]) -> Tensor:
        """
        Compute the jacobian of the constraints with respect to the design variables.

        Note that whereas the objective gradient is computed by pytorch's standard
        implicit approach the constraint jacobian is computed by pytorch's functional
        API.
        """
        x = as_tensor(x)
        # compute the jacobian which is shape (num. outputs, num. design variables)
        # NOTE: we need to manually reset the state for each forward call to ensure all computations
        # are actually being performed. Also resetting of this state doesn't seem to
        # effect the number of evaluations because of the order in which the optimizer
        # calls the method which appears to be:
        # - objective_grad
        # - objective_fun - this re-uses the previous computation from objective_grad
        # - constraint_jac
        # - constraint_fun - this re-uses the previous computation from constraint_jac
        constraint_jac = torch.autograd.functional.jacobian(
            func=lambda xx: self.constraint_fun(x=xx, jacobian_computation=True),
            inputs=x,
            create_graph=False,
            strict=False,
            vectorize=self.vectorize_constraint_jac,
        )
        # run a quick check
        assert torch.all(torch.isfinite(constraint_jac)), (
            "Non-finite value encountered in the constraint jacobian:\n%s"
            % str(constraint_jac.detach())
        )
        return constraint_jac

    def optimize(
        self,
        maxiter: int = 1000,
        display_step: int = 50,
        keep_feasible: bool = False,
        use_finite_diff: bool = False,
        **optimizer_options
    ) -> Optional[OptimizeResult]:
        """
        Optimize the objective, subject to constraints

        Args:
            maxiter: maximum number of optimization iterations
            display_step: number of optimization iterations between printing a logging
                message to monitor the procedure.
            keep_feasible: whether the optimizer should attempt to keep the optimization
                trajectory in a feasible region. Default: `False`.
            use_finite_diff: if True, will approximate gradients using finite
                differences rather than using pytorch's automatic differentiation.
                This will result in less accurate gradients and slower computation
                but can be useful for debugging.
            optimizer_options: additional optimization options from scipy's `trust-constr`
                implementation. See 
                `here <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`_.
        """
        self.display_step = display_step
        # compute the current iterate first. This is nessessary since we need to know the dimensions of all the outputs
        self.compute()
        logger.info(
            "Initial objective: %.3f." % self.outputs[self.objective_index].value_tensor
        )
        logger.info("Beginning optimization.")
        x0 = self.variables_tensor.detach()

        # setup constraints
        if self.num_constraints > 0:
            constraint_lb, constraint_ub = self.constraint_bounds_tensor
            constraints = NonlinearConstraint(
                fun=self.constraint_fun,
                jac=self.constraint_jac if not use_finite_diff else "2-point",
                lb=constraint_lb,
                ub=constraint_ub,
                keep_feasible=keep_feasible,
                **(  # specify a Hessian of zero if all the constraints are linear
                    dict(hess=lambda x, v: csr_matrix((x0.numel(),) * 2))
                    if self.constraints_are_all_linear
                    else {}
                )
            )
        else:
            # there are no constraints
            constraints = None

        # run the optimization
        try:
            res = minimize(
                fun=self.objective_fun,
                jac=self.objective_grad if not use_finite_diff else None,
                hess=csr_matrix((x0.numel(),) * 2)
                if self.outputs[self.objective_index].linear
                else None,
                x0=x0,
                bounds=[bound for bound in zip(*self.variable_bounds_tensor)],
                method="trust-constr",
                constraints=constraints,
                options=dict(maxiter=maxiter, **optimizer_options),
                callback=(
                    (lambda xk, res: self.callback(xk, res))
                    if display_step < maxiter
                    else None
                ),
            )
        except (KeyboardInterrupt):
            logger.info("Keyboard interrupt raised. Cleaning up...")
            return
        except:
            logger.error("Error during optimization!")
            raise
        else:
            logger.info("Optimization completed.")
            logger.info(
                "Function Evals: %d. Exit status: %s. Objective: %.4g. Constraint Violation: %.3g"
                % (res["nfev"], res["status"], res["fun"], res["constr_violation"])
            )
            logger.info("Execution time: %.1fs" % res["execution_time"])
            print(res["message"])
            # set the parameters internally
            self.variables_tensor = as_tensor(res["x"])
            return res

    def callback(self, xk, res):
        if res["niter"] == 1 or res["niter"] % self.display_step == 0:
            logger.info(
                "niter: %04d, fun: %.4f, constr_violation: %.3g, execution_time: %.1fs"
                % (
                    res["niter"],
                    -res["fun"]
                    if isinstance(self.outputs[self.objective_index], Maximize)
                    else res["fun"],
                    res["constr_violation"],
                    res["execution_time"],
                )
            )
        return False

    def check_grad(self) -> Tuple[Tensor, Tensor]:
        """ 
        verifies that the objective gradient and constraint jacobian are correct
        at the current setting of the design variables.

        Returns:
            A scalar tensor that gives the error in the objective gradient and
            a 1D tensor that gives the error in each row of the the constraint 
            jacobian, respectively.
        """
        # check the objective gradient
        objective_grad_error = check_grad(
            func=lambda x: self.objective_fun(x).numpy(),
            grad=lambda x: self.objective_grad(x).numpy(),
            x0=self.variables_tensor.detach().numpy(),
        )
        # check the constraint gradients
        # do this by looping through each constraint and the respective row of the jacobian
        constraints_jac_error = [
            check_grad(
                func=lambda x: self.constraint_fun(x)[i].numpy(),
                grad=lambda x: self.constraint_jac(x)[i, :].numpy(),
                x0=self.variables_tensor.detach().numpy(),
            )
            for i in range(self.num_constraints)
        ]
        return as_tensor(objective_grad_error), as_tensor(constraints_jac_error)

    def reset_design_variables(self) -> None:
        """ reset the design variables to their initial value. """
        self.variables_object = self.initial_design_variables

    def __str__(self):
        string = ""
        # print all design variables
        string += "*" * 80 + "\n"
        string += "DESIGN VARIABLES:   (value, lower, upper, active)\n"
        string += "*" * 80 + "\n"
        design_variables = self.variables_object
        for dv in design_variables:
            string += dv.__str__() + "\n"

        # print all outputs
        string += "*" * 80 + "\n"
        string += "OUTPUTS:            (value, lower, upper, active)\n"
        string += "*" * 80 + "\n"
        for out in self.outputs:
            string += out.__str__() + "\n"
        return string

