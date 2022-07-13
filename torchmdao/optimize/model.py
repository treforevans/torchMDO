import torch
import numpy
from scipy.optimize import minimize, NonlinearConstraint
from ..utils import is_broadcastable
from traceback import format_exc
from logging import getLogger
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Optional, List
from warnings import warn

logger = getLogger(__name__)
Tensor = torch.Tensor
as_tensor = torch.as_tensor


class ComputeObject(ABC):
    @abstractmethod
    def compute(self, to_plot=False):
        pass
