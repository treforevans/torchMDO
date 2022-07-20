from . import fea, lifting_line, optimize, model
from .model import Model
from .optimize import Optimizer, DesignVariable, Constraint, Minimize, Maximize
import torch
from sys import stdout
from logging import basicConfig, INFO

# set the default type to double
torch.set_default_dtype(torch.float64)

# setup logging configuration
basicConfig(
    stream=stdout,
    level=INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="[ %H:%M:%S ]",
)


__version__ = "0.1.2"
