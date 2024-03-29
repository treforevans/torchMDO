from setuptools import setup, find_packages
import itertools
import os.path
import codecs
from pathlib import Path

# define functions to read the package version
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


# read the contents of the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

# define essential requirements
install_requires = [
    "gpytorch>=1.7.0",
    "numpy>=1.23.0",
    "scipy>=1.9.0rc1",
    "torch>=1.12.0",
    "torch_interpolations @ git+https://github.com/sbarratt/torch_interpolations.git",
]
# define requirements for various uses
extras_require = dict(
    examples=["matplotlib",],
    dev=["black", "twine", "pytest", "check-manifest"],
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
# install_requires seems to get ignored when running `pip install -e .` so add to extras
extras_require["all"] += install_requires

# save the github repo
url_github = "https://github.com/treforevans/torchmdo"

# run setup
setup(
    name="torchmdo",
    version=get_version(os.path.join("torchmdo", "__init__.py")),
    description="Multidisciplinary design optimization made fast with PyTorch and modern automatic differentiation.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Trefor Evans",
    author_email="trefor@infera.ai",
    url=url_github,
    project_urls={
        "Documentation": "https://torchmdo.readthedocs.io/",
        "Source": url_github,
    },
    packages=find_packages(exclude=["test", "test.*"]),
    zip_safe=True,
    license="AGPL-3.0-or-later",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    install_reqires=install_requires,
    extras_require=extras_require,
    keywords=["Optimization", "Engineering", "Design", "Aerodynamics", "FEA",],
    test_suite="test",
)

# write a requirements.txt file to be read by anything building documentation
with open((this_directory / "docs" / "requirements.txt"), "w") as the_file:
    the_file.write(
        "# This file was automatically generated by setup.py, do not edit.\n"
    )
    # add my current package as a requirement (from github)
    the_file.write("git+%s.git\n" % url_github)
    # add all other requirements for documentation
    for requirement in install_requires + extras_require["docs"]:
        the_file.write("%s\n" % requirement)


# To push a new version to Pypi:
"""
```
python -m build
python -m twine upload --skip-existing dist/*

```
"""
