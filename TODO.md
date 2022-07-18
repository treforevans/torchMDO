## Docs
- [ ] Add readme to docs, may need to convert to rst
- [ ] Nbsphinx to include the ipython notebooks
- [x] Figure out how to only add some class methods to sphinx

## Optimizer
- [x] add an `Objective` class with `Minimize`, `Maximize` subclasses that can be used to directly specify the objective.
- [ ] implement a switch to turn pytorch gradients on or off for testing/comparison.
- [ ] write a test with a linear objective (linear constraints are used in the wing design example)

## Examples
- [x] aerodynamic planform optimization of a wing -> elliptical
- [ ] aerostructural optimization of a wing
- [ ] implement constrained problems from [this paper](https://arxiv.org/abs/2002.08526) in appendix G such as:
  - 60D rover trajectory planning
  - 124D vehicle design with 68 constraints
- [ ] re-write a model such as a lithium battery or single diode solar cell from https://github.com/paulcon/as-data-sets
      to pytorch and formulate an optimization problem.
- [ ] checkout some models from openmdao that could be re-writen in pytorch
