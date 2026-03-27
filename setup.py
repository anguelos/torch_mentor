from pathlib import Path
from setuptools import setup, find_packages

_version: dict = {}
exec(Path("mentor/version.py").read_text(), _version)

setup(
    name="torch-mentor",
    url="https://github.com/anguelos/torch_mentor",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    version=_version["__version__"],
    license="MIT",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        "torch", "tqdm", "torchvision", "matplotlib", "seaborn", "tensorboard", "fargv"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "mypy",
            "types-tqdm",
            "ruff",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
            "sphinx-copybutton",
            "nbsphinx",
            "readthedocs-sphinx-ext",
        ],
    },
    entry_points={
        "console_scripts": [
            "mtr_checkpoint=mentor.reporting:main_checkpoint",
            "mtr_plot_file_hist=mentor.reporting:main_plot_file_hist",
        ],
    },
)
