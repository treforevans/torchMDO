torchMDO
========

Multidisciplinary design optimization made fast with PyTorch and modern automatic differentiation.

At its heart, torchMDO is a library of optimizers and tools that allow you to build out large-scale
models to assess a design in PyTorch (with its Numpy-like syntax) and to optimize the design extremely quickly by taking
advantage of its automatic differentiation capabilities as well as its GPU acceleration.

Also, if you have a model that has previously been built in Python, you can convert it to PyTorch (which is
typically straightforward if it was originally implemented in Numpy) and
you can immediately plug it into torchMDO.

Online documentation (and examples):
    https://torchmdo.readthedocs.io/

Installation:
    Install using pip::

        pip install torchmdo # minimal install
        pip install torchmdo[examples] # to be able to run the examples

    To upgrade to the latest (unstable) version, run::

        pip install --upgrade git+https://github.com/treforevans/torchmdo.git

Source code repository (and issue tracker):
    https://github.com/treforevans/torchmdo/

License:
    AGPL-3.0-or-later -- see the file ``LICENSE`` for details.
