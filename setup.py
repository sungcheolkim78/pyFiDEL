from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyFiDEL",
    version="0.1.0",
    description="python implementation of the Fermi-Dirac ensemble learning (FiDEL) method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sungcheolkim78/pyFiDEL",
    author="Sung-Cheol Kim",
    author_email="sungcheol.kim78@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="classifier auc ensemble binary",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.7",
    install_requires=["pandas", "numpy", "matplotlib", "sklearn", "seaborn"],
    package_data={
        "": ["*.txt"],
        "pyFiDEL": ["data/*.gz"],
    },
    entry_points={
        "console_scripts": [
            "pyFiDEL=pyFiDEL:main",
        ],
    },
)
