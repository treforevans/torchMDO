import torch
import numpy
from dataclasses import dataclass
from typing import Union, Optional, List
from ..utils import is_broadcastable
from .model import ComputeObject

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
        assert self.value is None or isinstance(self.value, Tensor)
        for bound in [self.lower, self.upper]:
            if bound is not None:
                assert isinstance(bound, Tensor), "lower or upper bound is not a Tensor"
                assert self.value is None or is_broadcastable(
                    self.value.shape, bound.shape
                ), "lower or upper bound cannot be broadcast to shape of value"

    def extract_val(self, compute_object: ComputeObject):
        """extract value from a compute object"""
        self.value = as_tensor(getattr(compute_object, self.name))

    @property
    def value_tensor(self) -> Tensor:
        """return value, guaranteeing that it will be a tensor. """
        if self.value is None:
            raise RuntimeError("value has not yet been set.")
        return self.value

    @property
    def lower_tensor(self) -> Tensor:
        """
        Return the bound, guaranteeing that it will be a tensor and broadcasting it
        to the size of value.
        """
        bound = torch.zeros_like(self.value_tensor)
        if self.lower is not None:
            bound[:] = self.lower
        else:
            bound[:] = -torch.inf
        return bound

    @property
    def upper_tensor(self) -> Tensor:
        """
        Return the bound, guaranteeing that it will be a tensor and broadcasting it
        to the size of value.
        """
        bound = torch.zeros_like(self.value_tensor)
        if self.upper is not None:
            bound[:] = self.upper
        else:
            bound[:] = -torch.inf
        return bound

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
    """ Specifies the output of a model which may be an objective or constraint."""

    pass


@dataclass
class DesignVariable(InputOutput):
    """
    Stores a design variable, along with input bounds for that variable.
    """

    pass

