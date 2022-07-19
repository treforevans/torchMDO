# torchMDO
Multidisciplinary design optimization made fast with PyTorch and modern automatic differentiation.

At its heart, torchmdo is a library of optimizers and tools that allow you to build out large-scale 
model to assess a design in PyTorch (with its Numpy-like syntax) and to optimize the design extremely quickly by taking
advantage of the automatic differentiation capabilities provided by PyTorch.

- [torchMDO](#torchmdo)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Examples](#examples)
    - [Aerodynamic optimization of a wing planform](#aerodynamic-optimization-of-a-wing-planform)
  - [Performance](#performance)
  - [Other useful packages](#other-useful-packages)

## Installation

For development, run

```bash
pip install -e .[full]
```

Confirm setup by running tests as follows:

```bash
pytest
```

## Usage

_TODO: put a simple example here to show the basic usage of the package._
_Maybe create a 2D constrained optimization problem with a_
_Rosenbrock objective (such as [this](https://bit.ly/3AVCIUY))._

## Examples

### Aerodynamic optimization of a wing planform

<img 
  align="right" 
  style="width: 500px; height: auto; object-fit: contain" 
  hspace="10" 
  src="examples/wing_aerodynamic_optimization.gif">

[IPython notebook](./examples/wing_aerodynamic_optimization.ipynb)

In this example we consider a 50-dimensional nonlinear constrained optimization problem to optimize the shape 
of a wing to minimize induced drag, subject to a wing-area equality constraint.
We also compare the performance of modern automatic differentiation to the use of (classical) finite-difference
methods.

## Performance

The gradient computation time will be proportional to the number of constraints but is effectively
constant in the number of design variables, making it ideal for the optimization of high-dimensional
optimziation problems.
In contrast, gradient computations using finite difference methods scale independently with the number of
constraints but scale proportionally to the number of design variables, making them poorly suited to
high-dimensional real-world problems.
Finite-difference gradients are also approximated and can suffer from loss of precision.

## Other useful packages

- [`torch_interpolations`](https://github.com/sbarratt/torch_interpolations)
- [`torch_cg`](https://github.com/sbarratt/torch_cg)
- [`deq`](https://github.com/locuslab/deq) for differentiating through non-linear solvers
- [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) for differentiating through ODE solvers
