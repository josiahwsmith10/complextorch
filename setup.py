import re
from pathlib import Path

from setuptools import setup, find_packages


def _read_version() -> str:
    init_py = Path(__file__).parent / "complextorch" / "__init__.py"
    match = re.search(
        r'^__version__\s*=\s*"([^"]+)"', init_py.read_text(), re.MULTILINE
    )
    if not match:
        raise RuntimeError("Cannot find __version__ in complextorch/__init__.py")
    return match.group(1)


setup(
    name="complextorch",
    version=_read_version(),
    author="Josiah W. Smith",
    author_email="josiah.radar@gmail.com",
    description="A lightweight complex-valued neural network package built on PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://www.github.com/josiahwsmith10/complextorch",
    packages=find_packages(include=["complextorch", "complextorch.*"]),
    include_package_data=True,
    project_urls={
        "Bug Tracker": "https://github.com/josiahwsmith10/complextorch/issues",
        "Documentation": "https://complextorch.readthedocs.io/en/latest/index.html",
        "GitHub": "https://github.com/josiahwsmith10/complextorch",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    python_requires=">=3.6",
    install_requires=[
        "numpy>=2.2.0",
        "setuptools>=68.2.2",
        "torch>=1.11.0",
        "deprecated>=1.2.18",
    ],
)
