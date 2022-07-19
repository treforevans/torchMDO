import torch
import numpy
from dataclasses import dataclass, replace
from typing import Union, Optional, List
from functools import cached_property
from warnings import warn
from ..utils import is_broadcastable
from ..model import Model

Tensor = torch.Tensor
as_tensor = torch.as_tensor


@dataclass
class InputOutput:
    name: str
    value: Optional[Tensor] = None
    lower: Optional[Tensor] = None
    upper: Optional[Tensor] = None

    def __post_init__(self,):
        # check the types and the bounds
        assert isinstance(self.name, str)
        if self.value is not None:
            # ensure value is a tensor
            self.value = as_tensor(self.value)
        # ensure the types are none or Tensor
        self.value = None if self.value is None else as_tensor(self.value)
        self.lower = None if self.lower is None else as_tensor(self.lower)
        self.upper = None if self.upper is None else as_tensor(self.upper)
        # check the broadcasting of the bounds
        for bound in [self.lower, self.upper]:
            if bound is not None:
                assert self.value is None or is_broadcastable(
                    self.value.shape, bound.shape
                ), "lower or upper bound cannot be broadcast to shape of value"
        # make sure the bounds are not infeasible
        if self.lower is not None and self.upper is not None:
            assert torch.all(
                self.lower <= self.upper
            ), "Lower bound is greater than upper bound."

    def extract_val(self, model: Model):
        """
        extract value from a compute object

        Args:
            model: object to extract value from.
        """
        self.value = as_tensor(getattr(model, self.name))

    @property
    def value_tensor(self) -> Tensor:
        """return value, guaranteeing that it will be a tensor. """
        if self.value is None:
            raise RuntimeError("value has not yet been set.")
        return self.value

    @cached_property
    def lower_tensor(self) -> Tensor:
        """
        Return the bound, guaranteeing that it will be a tensor and broadcasting it
        to the size of value.
        Cached since this won't change throughout optimization.
        """
        if self.lower is not None:
            bound = self.lower.expand(self.shape)
        else:
            bound = as_tensor(-torch.inf).expand(self.shape)
        return bound

    @cached_property
    def upper_tensor(self) -> Tensor:
        """
        Return the bound, guaranteeing that it will be a tensor and broadcasting it
        to the size of value.
        Cached since this won't change throughout optimization.
        """
        if self.upper is not None:
            bound = self.upper.expand(self.shape)
        else:
            bound = as_tensor(torch.inf).expand(self.shape)
        return bound

    @cached_property
    def shape(self) -> torch.Size:
        """ get the shape of value """
        return self.value_tensor.size()

    @cached_property
    def numel(self) -> int:
        """ get the numel of value """
        return self.value_tensor.numel()

    @cached_property
    def is_constrained(self) -> bool:
        """ Returns true if there is some constraint present. """
        for bound in [self.lower, self.upper]:
            if bound is not None and torch.any(torch.isfinite(bound)):
                return True
        return False

    def replace(self, **changes):
        """
        Returns a shallow copy of itself with any changes made.
        """
        return replace(self, **changes)

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
        numel = max(
            max(numpy.size(self.lower), numpy.size(self.upper)), numpy.size(self.value),  # type: ignore
        )
        lvu = numpy.zeros((numel, 4,))
        lvu[:, 0] = torch.nan if self.value is None else torch.ravel(self.value)
        lvu[:, 1] = self.lower if self.lower is not None else -torch.inf
        lvu[:, 2] = self.upper if self.upper is not None else torch.inf
        # determine if lower or upper bound is active
        # Will cancel if it's equality constraint
        lvu[:, 3] = [
            (0 if numpy.isnan(v) or v > l else -1)
            + (0 if numpy.isnan(v) or v < u else 1)
            for (v, l, u) in lvu[:, :3]
        ]
        string += numpy.array2string(
            lvu, formatter={"float_kind": lambda x: "%.3f" % x}, separator="\t"
        )
        return string


@dataclass
class Output(InputOutput):
    """ Base class for outputs (objectives or constraints). """

    linear: bool = False  # set true if this output is linear in the input

    def __post_init__(self,):
        super().__post_init__()
        if self.value is not None:
            warn("Initial value provided for Objective or Constraint will be ignored.")


@dataclass
class Constraint(Output):
    """ 
    Specifies an equality or inequality constraint to be used for optimization.
    
    Args:
        name: the name of an attribute in a :class:`~.Model` that will be 
            used as a constraint for optimization.
        lower: if specified, this output will be constrained
            as an inequality constraint such that the output value must be 
            greater than or equal to `lower`.
            If `upper` is also specified and `lower == upper` then this will 
            be an equality constraint.
        upper: if specified, this output will be constrained
            as an inequality constraint such that the output value must be 
            less than or equal to `upper`.
            If `lower` is also specified and `lower == upper` then this will 
            be an equality constraint.
        linear: if True then this constraint will be treated as
            linear in the design variables
            (as specified by :class:`~.DesignVariable`).
            Default: `False`.
    """

    pass


@dataclass
class Objective(Output):
    """ Base class for objectives. """

    pass


@dataclass
class Minimize(Objective):
    """ 
    Specifies an objective to be minimized during optimization.
    
    Args:
        name: the name of an attribute in a :class:`~.Model` that will be 
            minimized during optimization.
        linear: if True then this objective will be treated as
            linear in the design variables
            (as specified by :class:`~.DesignVariable`).
            Default: `False`.
    """

    pass


@dataclass
class Maximize(Objective):
    """ 
    Specifies an objective to be maximized during optimization.
    
    Args:
        name: the name of an attribute in a :class:`~.Model` that will be 
            maximized during optimization.
        linear: if True then this objective will be treated as
            linear in the design variables
            (as specified by :class:`~.DesignVariable`).
            Default: `False`.
    """

    pass


@dataclass
class DesignVariable(InputOutput):
    """
    Specifies a design variable, along with input bounds for that variable, if any.
    
    Args:
        name: the name of an attribute in a :class:`~.Model` that will 
            be used as a design variable for optimization.
        value: if specified, this value will be used as an
            initial value of this design variable for optimization.
            If not specified, then the value of the attribute stored in 
            the :class:`~.Model` with the same `name` will be used as the 
            initial value of this design variable for optimization.
        lower: if specified, this design variable will
            have a lower bound given by this value.
        upper: if specified, this design variable will
            have an upper bound given by this value.
    """

    def stringify(self) -> str:
        """
        Returns a string version of the class that can be immediately used to reproduce
        it. For instance, this can be used to save the state of the design variable as 
        a string after optimization has completed.
        """
        return super().stringify()
