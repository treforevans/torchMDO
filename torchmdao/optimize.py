import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from traceback import format_exc
from logging import getLogger
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Optional, List
from warnings import warn

logger = getLogger(__name__)


@dataclass
class DesignVariable:
    name: str
    val: Union[float, np.ndarray, list]
    lower: Optional[float] = None
    upper: Optional[float] = None
    finite_diff_step: float = 1e-6  # stepsize used for finite diff calcs. Called `finite_diff_rel_step` in the optimizer but seems to function as expected.

    def __post_init__(self,):
        assert isinstance(self.name, str)
        assert isinstance(self.val, (float, np.ndarray, list))
        assert isinstance(self.finite_diff_step, float)

    def stringify(self) -> str:
        """
        Returns a string version of the class that can be immediately used to reproduce
        it.
        """
        return super().__str__()

    def __str__(self) -> str:
        """
        For printing in a pretty way.
        For non-pretty printing in a way that can be copied to reset the 
        design variable, call `stringify`
        """
        string = self.name + ": \n"  # ": (value, lower, upper, active)\n"
        lvu = np.zeros((np.size(self.val), 4))
        lvu[:, 0] = np.ravel(self.val)
        lvu[:, 1] = self.lower if self.lower is not None else -np.inf
        lvu[:, 2] = self.upper if self.upper is not None else np.inf
        lvu[:, 3] = [  # determine if bound is active
            (0 if v > l else -1)
            + (  # check if lower bound active
                0 if v < u else 1
            )  # check if upper bound active. Will cancel if it's equality constraint
            for (v, l, u) in lvu[:, :3]
        ]
        string += np.array2string(
            lvu, formatter={"float_kind": lambda x: "%.3f" % x}, separator="\t"
        )
        return string


class Output:
    def __init__(self, name: str, lower: float = -np.inf, upper: float = np.inf):
        """ Aircraft output."""
        assert isinstance(name, str)
        assert lower is not None, "use -np.inf"
        assert upper is not None, "use np.inf"
        self.name = name
        self.lower = lower
        self.upper = upper
        self.val = None  # value can be set later

    def __str__(self):
        string = self.name + ": \n"  # ": (value, lower, upper, active)\n"
        lvu = np.zeros(
            (max(max(np.size(self.lower), np.size(self.upper)), np.size(self.val)), 4)
        )
        lvu[:, 0] = np.ravel(self.val)
        lvu[:, 1] = self.lower
        lvu[:, 2] = self.upper
        lvu[:, 3] = [  # determine if bound is active
            (0 if self.val is None or l is None or v > l else -1)
            + (  # check if lower bound active
                0 if self.val is None or u is None or v < u else 1
            )  # check if upper bound active
            for (v, l, u) in lvu[:, :3]
        ]
        string += np.array2string(
            lvu, formatter={"float_kind": lambda x: "%.3f" % x}, separator="\t"
        )
        return string


class ComputeObject(ABC):
    @abstractmethod
    def compute(self, to_plot=False):
        pass


class Optimizer:
    def __init__(
        self,
        initial_design_variables: List[DesignVariable],
        outputs: List[Output],
        obj_constr: ComputeObject,
        objective_index: int = 0,
    ):
        """
        Inputs:
            design_variables : list of DesignVariable objects
            outputs : list of Outputs objects
            obj_constr : ComputeObject what computes all outputs when `obj_constr.compute()` is called
            objective_index : index of objective in outputs
        """
        self.initial_design_variables = initial_design_variables
        # check obj_constr
        assert isinstance(obj_constr, ComputeObject)
        self.obj_constr = obj_constr
        # check outputs
        assert isinstance(objective_index, int)
        for out in outputs:
            assert isinstance(out, Output)
            assert isinstance(out.name, str)
        self.objective_index = objective_index
        self.outputs = outputs
        # internalize parameters
        self.variables_object = self.initial_design_variables
        # do a sanity check to make sure the reconstructed desgin variables are the same
        for (original, reconstructed) in zip(
            self.initial_design_variables, self.variables_object
        ):
            try:
                assert original.name == reconstructed.name
                assert np.array_equal(original.val, reconstructed.val)
                assert np.array_equal(original.lower, reconstructed.lower)
                assert np.array_equal(original.upper, reconstructed.upper)
            except:
                print("design variable internalization failed for %s." % original.name)
                raise
        # initialize an array to save the variable values from the last iteration so we
        # can check whether or not they have changed
        self._last_variables = None

    @property
    def variables_object(self)->List[DesignVariable]:
        design_variables = []
        for idv in self.initial_design_variables:
            # get variable value if present
            design_variables.append(
                DesignVariable(
                    name=idv.name,
                    val=getattr(self.obj_constr, idv.name),
                    lower=idv.lower,
                    upper=idv.upper,
                    finite_diff_step=idv.finite_diff_step,
                )
            )
        return design_variables

    @variables_object.setter
    def variables_object(self, design_variables):
        for dv in design_variables:
            # make sure design variables are of the correct type
            assert isinstance(dv, DesignVariable)
            setattr(self.obj_constr, dv.name, dv.val)
        # reset state since variable changed
        self.reset_state()

    @property
    def variables_array(self):
        """
        returns the variables as a 1d array
        """
        return np.concatenate([np.ravel(dv.val) for dv in self.variables_object])

    @variables_array.setter
    def variables_array(self, value):
        """
        setter for variables_array property
        """
        if self._last_variables is None or np.any(
            value != self._last_variables
        ):  # if any value changed
            i_cur = 0  # current variable index
            for idv in self.initial_design_variables:
                dv_size = np.size(idv.val)
                setattr(
                    self.obj_constr,
                    idv.name,
                    np.reshape(value[i_cur : (i_cur + dv_size)], np.shape(idv.val)),
                )
                i_cur += dv_size  # update the position of the index
            assert i_cur == value.size, "sanity check: did not use all design variables"
            self.reset_state()
            self._last_variables = value.copy()  # save parameters for next iter
        else:
            pass  # nothing changed so pass

    @property
    def finite_diff_step_array(self):
        """
        returns the finite diff step size for each variables as a 1d array
        """
        finite_diff_step = np.concatenate(
            [  # create a big 1d array
                np.ravel(
                    [idv.finite_diff_step,] * np.size(idv.val)
                    if np.size(idv.finite_diff_step) == 1
                    else idv.finite_diff_step
                )  # broadcast to correct size if nessessary
                for idv in self.initial_design_variables  # loop through all design variables
            ]
        )
        return finite_diff_step

    @property
    def variable_bounds_array(self):
        """
        get bounds for all optimization variables
        """
        lower, upper = [
            np.concatenate(
                [  # create a big 1d array
                    np.ravel(
                        [getattr(idv, key),] * np.size(idv.val)
                        if np.size(getattr(idv, key)) == 1
                        else getattr(idv, key)
                    )  # broadcast to correct size if nessessary
                    for idv in self.initial_design_variables  # loop through all design variables
                ]
            )
            for key in ["lower", "upper"]
        ]
        return lower, upper

    @property
    def constraint_bounds_array(self):
        """
        get bounds for all constrained outputs
        """
        lower, upper = [
            np.concatenate(
                [  # create a big 1d array
                    np.ravel(
                        [getattr(out, key),]
                        * np.size(getattr(self.obj_constr, out.name))
                        if np.size(getattr(out, key)) == 1
                        else getattr(out, key)
                    )  # broadcast to correct size if nessessary
                    for out in self.outputs  # loop through all outputs
                    if (
                        np.isfinite(out.lower) or np.isfinite(out.upper)
                    )  # filter to include only outputs that are constrained
                ]
            )
            for key in ["lower", "upper"]
        ]
        return lower, upper

    def reset_state(self):
        """ resets outputs and internal state """
        # remove all output values
        for out in self.outputs:
            out.val = None

    def compute(self, to_plot=False, **kwargs):
        """
        run objective and constraint functions
        """
        # check if computations completed
        if self.outputs[self.objective_index].val is not None:
            # if the objective is already computed then don't need to recompute anything
            return

        # run the objective and constraint functions
        self.obj_constr.compute(to_plot=to_plot, **kwargs)

        # extract the outputs
        for out in self.outputs:
            out.val = getattr(self.obj_constr, out.name)

    def objective_fun(self, x):
        self.variables_array = x
        self.compute()
        return self.outputs[self.objective_index].val

    def constraint_fun(self, x):
        self.variables_array = x
        self.compute()
        return np.concatenate(
            [
                np.ravel(out.val)
                for out in self.outputs
                if (np.isfinite(out.lower) or np.isfinite(out.upper))
            ]
        )  # return any constrained output

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
        logger.info("Initial objective: %.3f." % self.outputs[self.objective_index].val)
        logger.info("Beginning optimization.")

        # setup constraints
        finite_diff_step = self.finite_diff_step_array
        constraint_lb, constraint_ub = self.constraint_bounds_array
        constraints = NonlinearConstraint(
            fun=self.constraint_fun,
            lb=constraint_lb,
            ub=constraint_ub,
            finite_diff_rel_step=finite_diff_step,
            keep_feasible=keep_feasible,
        )

        # run the optimization
        try:
            res = minimize(
                fun=self.objective_fun,
                x0=self.variables_array.copy(),
                bounds=[bound for bound in zip(*self.variable_bounds_array)],
                method="trust-constr",
                constraints=constraints,
                options=dict(
                    maxiter=maxiter,
                    finite_diff_rel_step=finite_diff_step,
                    **trust_constr_options
                ),
                callback=(
                    (lambda xk, res: self.callback(xk, res))
                    if np.isfinite(display_step)
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

