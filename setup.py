import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "complextorch",
    version = "1.0.0",
    author = "Josiah W. Smith",
    author_email = "josiah.radar@gmail.com",
    description = "A lightweight complex-valued neural network package built on PyTorch",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "package URL",
    packages=["complextorch"],
    project_urls = {
        "Bug Tracker": "package issues URL",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "complextorch"},
    python_requires = ">=3.6"
)