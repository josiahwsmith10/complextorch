from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name = "complextorch",
    version = "0.1.0",
    author = "Josiah W. Smith",
    author_email = "josiah.radar@gmail.com",
    description = "A lightweight complex-valued neural network package built on PyTorch",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url="https://www.github.com/josiahwsmith10/complextorch",
    packages=find_packages(),
    project_urls = {
        "Bug Tracker": "package issues URL",
        "Documentation": "https://complextorch.readthedocs.io/en/latest/index.html",
        "GitHub": "https://www.github.com/josiahwsmith10/complextorch",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "./"},
    python_requires = ">=3.6"
)