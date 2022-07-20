## Docs

- [ ] put a simple example in the readme to show the basic usage of the package.
      Maybe create a 2D constrained optimization problem with a
      Rosenbrock objective (such as [this](https://bit.ly/3AVCIUY)).
- [ ] Add readme to docs, may need to convert to rst
- [ ] Nbsphinx to include the ipython notebooks
- [x] Figure out how to only add some class methods to sphinx

## Optimizer

- [x] add an `Objective` class with `Minimize`, `Maximize` subclasses that can be used to directly specify the objective.
- [x] implement a switch to turn pytorch gradients on or off for testing/comparison in `optimize`.
- [ ] add a way to specify a callback function (e.g. for plotting)
- [ ] add example to optimizer docstring
- [ ] write a test with a linear objective (linear constraints are used in the wing design example)
- [ ] consider adding other objectives, although these are non-standard:
  - [ ] consider adding a `Target` objective whose goal is to minimize the least-squares distance from a target value
  - [ ] consider adding a `FindFeasible` objective

## Examples

- [x] aerodynamic planform optimization of a wing -> elliptical
- [ ] aerostructural optimization of a wing
- [ ] consider aero shape optimization from doublet or other potential flow
- [ ] implement constrained problems from [this paper](https://arxiv.org/abs/2002.08526) in appendix G such as:
  - 60D rover trajectory planning
  - 124D vehicle design with 68 constraints
- [ ] consider some models such as a lithium battery or single diode solar cell from https://github.com/paulcon/as-data-sets
      and formulate an optimization problem.
- [ ] checkout some models from openmdao that could be re-written in pytorch
