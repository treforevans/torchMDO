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
    name="torchmdao",
    version=get_version(os.path.join("torchmdao", "__init__.py")),
    description="Multidisciplinary design optimization tools in PyTorch.",
    author="Trefor Evans",
    author_email="trefor@infera.ai",
    packages=["torchmdao"],
    zip_safe=True,
    install_requires=[
        "numpy~=1.23.0",
        "scipy",
        "matplotlib",
        "pytest",
        "gpytorch~=1.7.0",
        "torch~=1.12.0",
    ],
)
