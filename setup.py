from setuptools import setup
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


setup(
    name="torchmdo",
    version=get_version(os.path.join("torchmdo", "__init__.py")),
    description="Multidisciplinary design optimization made fast with PyTorch and modern automatic differentiation.",
    author="Trefor Evans",
    author_email="trefor@infera.ai",
    packages=["torchmdo"],
    zip_safe=True,
    install_requires=[
        "numpy~=1.23.0",
        "scipy~=1.9.0rc1",
        "gpytorch~=1.7.0",
        "torch~=1.12.0",
    ],
    license="AGPL-3.0-or-later",
    extras_require=dict(
        full=[
            "matplotlib",
            "pytest",
            "sphinx~=5.0.2",
            "sphinx_rtd_theme",
            "sphinx-autodoc-typehints",
        ]
    ),
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
