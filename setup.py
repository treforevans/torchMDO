from setuptools import setup
import itertools
import os.path
import codecs


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


install_requires = [
    "gpytorch~=1.7.0",
    "numpy~=1.23.0",
    "scipy~=1.9.0rc1",
    "torch~=1.12.0",
]
extras_require = dict(
    examples=["matplotlib",],
    dev=["black", "twine", "pytest",],
    docs=[
        "ipython",
        "ipykernel",
        "nbsphinx",
        "sphinx~=5.0.2",
        "sphinx_rtd_theme",
        "sphinx-autodoc-typehints",
    ],
)
extras_require["all"] = list(itertools.chain.from_iterable(extras_require.values()))

setup(
    name="torchmdo",
    version=get_version(os.path.join("torchmdo", "__init__.py")),
    description="Multidisciplinary design optimization made fast with PyTorch and modern automatic differentiation.",
    author="Trefor Evans",
    author_email="trefor@infera.ai",
    packages=["torchmdo"],
    zip_safe=True,
    license="AGPL-3.0-or-later",
    install_reqires=install_requires,
    extras_require=extras_require,
    keywords=[
        "optimization",
        "MDO",
        "engineering",
        "design",
        "pytorch",
        "aerodynamics",
        "FEA",
    ],
)
