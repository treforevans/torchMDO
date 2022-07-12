from setuptools import setup
import os.path
import codecs


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    """
    get the version of the package.
    From: https://packaging.python.org/guides/single-sourcing-package-version/
    """
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


VERSION_PATH = os.path.join("torchmdao", "__init__.py")

setup(
    name="torchmdao",
    version=get_version(VERSION_PATH),
    description="Multidisciplinary design optimization tools in PyTorch.",
    author="Trefor Evans",
    author_email="trefor@infera.ai",
    packages=["torchmdao"],
    zip_safe=True,
    install_requires=["numpy", "scipy", "matplotlib", "pytest", "gpytorch", "torch",],
)
