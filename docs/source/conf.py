# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import re
from pathlib import Path

project = "complextorch"
copyright = "2025, Josiah W. Smith"
author = "Josiah W. Smith"


def _read_version() -> str:
    init_py = Path(__file__).resolve().parents[2] / "complextorch" / "__init__.py"
    match = re.search(
        r'^__version__\s*=\s*"([^"]+)"', init_py.read_text(), re.MULTILINE
    )
    if not match:
        raise RuntimeError("Cannot find __version__ in complextorch/__init__.py")
    return match.group(1)


release = _read_version()
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
