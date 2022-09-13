torchMDO
========

Multidisciplinary design optimization made fast with PyTorch and modern automatic differentiation.

At its heart, torchMDO is a library of optimizers and tools that allow you to build out large-scale
models to assess a design in PyTorch (with its Numpy-like syntax) and to optimize the design extremely quickly by taking
advantage of its automatic differentiation capabilities as well as its GPU acceleration.

Also, if you have a model that has previously been built in Python, you can convert it to PyTorch (which is
typically straightforward if it was originally implemented in Numpy) and
you can immediately plug it into torchMDO.

Examples
--------

`Aerodynamic shape optimization of a wing's planform <https://torchmdo.readthedocs.io/en/latest/examples/wing_aerodynamic_optimization.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <img 
    align="right" 
    style="width: 500px; height: auto; object-fit: contain" 
    hspace="10" 
    src="https://github.com/treforevans/torchMDO/raw/main/examples/wing_aerodynamic_optimization.gif">

In this simple example, we consider a 50-dimensional nonlinear constrained optimization problem to optimize the shape
of a wing to minimize induced drag, subject to a wing-area equality constraint.
We also compare the performance of modern automatic differentiation to the use of (classical) finite-difference
methods.

Installation
-------------
Install using pip::

    pip install torchmdo # minimal install
    pip install torchmdo[examples] # to be able to run the examples

To upgrade to the latest (unstable) version, run::

    pip install --upgrade git+https://github.com/treforevans/torchmdo.git

Documentation
-------------

Online documentation:
    https://torchmdo.readthedocs.io/

Source code repository (and issue tracker):
    https://github.com/treforevans/torchmdo/

License:
    `AGPL-3.0-or-later <https://github.com/treforevans/torchMDO/blob/main/LICENSE>`_
    --
    please `contact <mailto:trefor@infera.ai>`_ for inquiries about licensing.

