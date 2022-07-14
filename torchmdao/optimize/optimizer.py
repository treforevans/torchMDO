import torch
from scipy.optimize import minimize, NonlinearConstraint, OptimizeResult
from logging import getLogger
from typing import Union, Optional, List, Tuple
from warnings import warn
from pdb import set_trace
from functools import cached_property
from .input_output import DesignVariable, Output
from ..model import ComputeObject

logger = getLogger(__name__)
Tensor = torch.Tensor
as_tensor = torch.as_tensor


class Optimizer:
    def __init__(
        self,
        initial_design_variables: List[DesignVariable],
        outputs: List[Output],
        compute_object: ComputeObject,
        objective_index: int = 0,
    ):
        """
        Inputs:
            design_variables : list of DesignVariable objects
            outputs : list of Outputs objects
            compute_object : ComputeObject what computes all outputs 
                when `compute_object.compute()` is called
            objective_index : index of objective in outputs
        """
        # check compute_object
        assert isinstance(compute_object, ComputeObject)
        self.compute_object = compute_object
        # check outputs
        assert len(outputs) >= 1
        for out in outputs:
            assert isinstance(out, Output)
        self.outputs = outputs
        # check objective index
        assert isinstance(objective_index, int)
        assert objective_index in torch.arange(
            len(self.outputs)
        ), "objective_index not in range."
        self.objective_index = objective_index
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
        # reset state since variable changed
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
        if self._last_variables is None or torch.any(value != self._last_variables):
            # if any value changed then update the compute object with the new variables
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
            self._last_variables = value.detach().clone()
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
        return int(torch.count_nonzero(self.constrained_output_mask))

    def reset_state(self):
        """ resets outputs and internal state """
        # remove all output values
        for out in self.outputs:
            out.value = None

    def compute(self, **kwargs) -> None:
        """
        run compute to get all outputs (objective and constraints) for the current
        design variable setting.
        """
        # check if the computation has already been completed completed
        if self.outputs[self.objective_index].value is not None:
            # if the objective is already computed then don't need to recompute anything
            return

        # run the objective and constraint functions
        self.compute_object.compute(**kwargs)

        # extract the outputs
        for out in self.outputs:
            out.extract_val(compute_object=self.compute_object)

    def objective_fun(self, x) -> Tensor:
        """ return the objective """
        x = as_tensor(x)
        self.variables_tensor = x
        self.compute()
        objective = self.outputs[self.objective_index].value_tensor
        assert objective.numel() == 1, "objective must be a scalar"
        return objective.reshape(())

    def constraint_fun(self, x) -> Tensor:
        """ return the constraints """
        x = as_tensor(x)
        self.variables_tensor = x
        self.compute()
        # extract all the outputs as a concatenated 1d vector
        all_outputs = torch.cat([out.value_tensor.reshape(-1) for out in self.outputs])
        # return any constrained outputs
        return all_outputs[self.constrained_output_mask]

    def optimize(
        self, maxiter=1000, display_step=50, keep_feasible=False, **trust_constr_options
    ) -> Optional[OptimizeResult]:
        """
        optimizing the objective, subject to constraints

        Inputs:
            max_iters : int
                maximum number of optimization iterations
            **trust_constr_options : from https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html
        """
        self.display_step = display_step
        # compute the current iterate first. This is nessessary since we need to know the dimensions of all the outputs
        self.compute()
        logger.info(
            "Initial objective: %.3f." % self.outputs[self.objective_index].value_tensor
        )
        logger.info("Beginning optimization.")

        # setup constraints
        if self.num_constraints > 0:
            constraint_lb, constraint_ub = self.constraint_bounds_tensor
            constraints = NonlinearConstraint(
                fun=self.constraint_fun,
                lb=constraint_lb,
                ub=constraint_ub,
                keep_feasible=keep_feasible,
            )
        else:
            # there are no constraints
            constraints = None

        # run the optimization
        try:
            res = minimize(
                fun=self.objective_fun,
                x0=self.variables_tensor.clone(),
                bounds=[bound for bound in zip(*self.variable_bounds_tensor)],
                method="trust-constr",
                constraints=constraints,
                options=dict(maxiter=maxiter, **trust_constr_options),
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

    def plot(self, **kwargs):
        self.reset_state()
        self.compute(to_plot=True, **kwargs)

    def callback(self, xk, res):
        if res["niter"] == 1 or res["niter"] % self.display_step == 0:
            logger.info(
                "niter: %04d, fun: %.4f, constr_violation: %.3g, execution_time: %.1fs"
                % (
                    res["niter"],
                    res["fun"],
                    res["constr_violation"],
                    res["execution_time"],
                )
            )
        return False

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

