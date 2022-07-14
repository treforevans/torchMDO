import torch
from abc import ABC, abstractmethod

Tensor = torch.Tensor
as_tensor = torch.as_tensor


class ComputeObject(ABC):
    @abstractmethod
    def compute(self) -> None:
        """
        Compute and set all attributes that will be used as outputs for the model
        such as objectives and constraints.
        This method can take additional keyword arguments if desired.
        """
        pass
