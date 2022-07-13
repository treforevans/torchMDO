import torch
from numpy import prod
from scipy.optimize import minimize, NonlinearConstraint
from logging import getLogger
from typing import Union, Optional, List, Tuple
from warnings import warn
from functools import cached_property
from .input_output import DesignVariable, Output
from .model import ComputeObject

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
            compute_object : ComputeObject what computes all outputs when `compute_object.compute()` is called
            objective_index : index of objective in outputs
        """
        # check compute_object
        assert isinstance(compute_object, ComputeObject)
        self.compute_object = compute_object
        # check outputs
        for out in outputs:
            assert isinstance(out, Output)
        self.outputs = outputs
        # check objective index
        assert isinstance(objective_index, int)
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
        # initialize an array to save the variable values from the last iteration so we
        # can check whether or not they have changed
        self._last_variables = None

    @property
    def variables_object(self) -> List[DesignVariable]:
        design_variables = []
        for idv in self.initial_design_variables:
            # get variable value if present
            design_variables.append(
                DesignVariable(
                    name=idv.name,
                    value=as_tensor(getattr(self.compute_object, idv.name)),
                    lower=idv.lower,
                    upper=idv.upper,
                )
            )
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
    def variables_array(self) -> Tensor:
        """
        returns the variables as a 1d array
        """
        return torch.cat([torch.ravel(dv.value_tensor) for dv in self.variables_object])

    @variables_array.setter
    def variables_array(self, value: Tensor):
        """
        setter for variables_array property
        """
        if self._last_variables is None or torch.any(value != self._last_variables):
            # if any value changed
            i_cur = 0  # current variable index
            for idv in self.initial_design_variables:
                dv_shape = idv.value_tensor.size()
                dv_size = prod(dv_shape)
                setattr(
                    self.compute_object,
                    idv.name,
                    torch.reshape(value[i_cur : (i_cur + dv_size)], dv_shape),
                )
                i_cur += dv_size  # update the position of the index
            assert i_cur == value.size, "sanity check: did not use all design variables"
            self.reset_state()
            # save parameters for next iteration
            self._last_variables = value.detach().clone()
        else:
            pass  # nothing changed so pass

    @property
    def variable_bounds_array(self) -> Tuple[Tensor, Tensor]:
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
    def constraint_bounds_array(self) -> Tuple[Tensor, Tensor]:
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

    def reset_state(self):
        """ resets outputs and internal state """
        # remove all output values
        for out in self.outputs:
            out.value = None

    def compute(self, to_plot=False, **kwargs) -> None:
        """
        run compute to get all outputs (objective and constraints) for the current
        design variable setting.
        """
        # check if computations completed
        if self.outputs[self.objective_index].value is not None:
            # if the objective is already computed then don't need to recompute anything
            return

        # run the objective and constraint functions
        self.compute_object.compute(to_plot=to_plot, **kwargs)

        # extract the outputs
        for out in self.outputs:
            out.extract_val(compute_object=self.compute_object)

    def objective_fun(self, x) -> Tensor:
        """ return the objective """
        self.variables_array = x
        self.compute()
        return self.outputs[self.objective_index].value_tensor

    def constraint_fun(self, x) -> Tensor:
        """ return the constraints """
        self.variables_array = x
        self.compute()
        # return any constrained outputs
        all_outputs = torch.cat([out.value_tensor.reshape(-1) for out in self.outputs])
        return all_outputs[self.constrained_output_mask]

    def optimize(
        self, maxiter=1000, display_step=50, keep_feasible=False, **trust_constr_options
    ):
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
        constraint_lb, constraint_ub = self.constraint_bounds_array
        constraints = NonlinearConstraint(
            fun=self.constraint_fun,
            lb=constraint_lb,
            ub=constraint_ub,
            keep_feasible=keep_feasible,
        )

        # run the optimization
        try:
            res = minimize(
                fun=self.objective_fun,
                x0=self.variables_array.clone(),
                bounds=[bound for bound in zip(*self.variable_bounds_array)],
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
            self.variables_array = res["x"]
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

