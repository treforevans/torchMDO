# torchMDO

Multidisciplinary design optimization made fast with PyTorch and modern automatic differentiation.

At its heart, torchMDO is a library of optimizers and tools that allow you to build out large-scale
models to assess a design in PyTorch (with its Numpy-like syntax) and to optimize the design extremely quickly by taking
advantage of its automatic differentiation capabilities as well as its GPU acceleration.

Also, if you have a model that has previously been built in Python, you can convert it to PyTorch (which is
typically straightforward if it was originally implemented in Numpy) and
you can immediately plug it into torchMDO.

<!--
Article about converting from numpy that may be worthwhile:
https://pytorch.org/blog/torch-linalg-autograd/
-->

## Installation

Install using pip:

```bash
# minimal install:
pip install torchmdo
# or to be able to run the tutorials:
pip install torchmdo[examples]
```

<!--
For development, run
```bash
pip install -e .[examples,dev]
```
-->

## Tutorials


### Aerodynamic optimization of a wing planform

<img 
  align="right" 
  style="width: 500px; height: auto; object-fit: contain" 
  hspace="10" 
  src="examples/wing_aerodynamic_optimization.gif">

[IPython notebook](./examples/wing_aerodynamic_optimization.ipynb)

In this simple example, we consider a 50-dimensional nonlinear constrained optimization problem to optimize the shape
of a wing to minimize induced drag, subject to a wing-area equality constraint.
We also compare the performance of modern automatic differentiation to the use of (classical) finite-difference
methods.

<!--
## Performance

The gradient computation time will be proportional to the number of constraints but is effectively
constant in the number of design variables, making it ideal for the optimization of high-dimensional
optimziation problems.
In contrast, gradient computations using finite difference methods scale independently with the number of
constraints but scale proportionally to the number of design variables, making them poorly suited to
high-dimensional real-world problems.
Finite-difference gradients are also approximated and can suffer from loss of precision.
-->

<!--
## Other useful packages
I've listed here some other useful packages that might be helpful to build out a
model in PyTorch.

- [`torch_interpolations`](https://github.com/sbarratt/torch_interpolations)
- [`torch_cg`](https://github.com/sbarratt/torch_cg)
- [`deq`](https://github.com/locuslab/deq) for differentiating through non-linear solvers
- [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) for differentiating through ODE solvers
- -->
