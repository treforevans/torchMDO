## Docs

- [ ] put a simple example in the readme to show the basic usage of the package.
      Maybe create a 2D constrained optimization problem with a
      Rosenbrock objective (such as [this](https://bit.ly/3AVCIUY)).
- [x] Add readme to docs, may need to convert to rst
- [x] Nbsphinx to include the ipython notebooks
- [x] Figure out how to only add some class methods to sphinx
- [ ] setup github actions CI
- [ ] add some badges: documentation, code coverage, build

## Optimizer

- [x] add an `Objective` class with `Minimize`, `Maximize` subclasses that can be used to directly specify the objective.
- [x] implement a switch to turn pytorch gradients on or off for testing/comparison in `optimize`.
- [x] add a way to specify a callback function (e.g. for plotting)
- [ ] add details about the optimizer used in the `Optimizer` docstring.
- [ ] add example to optimizer docstring?
- [ ] write a test with a linear objective (linear constraints are used in the wing design example)
- [ ] consider adding the following objectives. Perhaps the best thing to do would be to inherit the optimizer class?
  - [ ] consider adding a `Target` objective whose goal is to minimize the least-squares distance from a target value
  - [ ] consider adding a `NearestFeasible` objective which could be quadratic.

## Examples

- [x] aerodynamic planform optimization of a wing -> elliptical
- [ ] aerostructural optimization of a wing: maybe don't model twisting but can set a tip deflection constraint
      and can set lift requirements based on the weight of a structure.
- [ ] generate adversarial examples: take an image-net pre-trained model, minimize the distance from the original image why
      enforcing that the class-conditional probability of the correct class is below a threshold. This would
      use the FindFeasible objective.
- [ ] consider aero shape optimization from doublet or other potential flow
- [ ] implement constrained problems from [this paper](https://arxiv.org/abs/2002.08526) in appendix G such as:
  - 60D rover trajectory planning
  - 124D vehicle design with 68 constraints
- [ ] consider some models such as a lithium battery or single diode solar cell from https://github.com/paulcon/as-data-sets
      and formulate an optimization problem.
- [ ] checkout some models from openmdao that could be re-written in pytorch
