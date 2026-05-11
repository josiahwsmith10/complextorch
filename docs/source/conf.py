# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "complextorch"
copyright = "2026, Josiah W. Smith"
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

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxext.opengraph",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Source files ------------------------------------------------------------

# myst-nb owns .md and .ipynb (it subsumes myst-parser for Markdown).
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- MyST / MyST-NB ----------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

nb_execution_mode = "auto"
nb_execution_timeout = 120
nb_execution_allow_errors = False
nb_execution_excludepatterns = []

# -- Autoapi -----------------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = [str(Path(__file__).resolve().parents[2] / "complextorch")]
autoapi_root = "api"
autoapi_keep_files = False
# We wire the API tree into our own toctree in index.md, so don't let
# autoapi inject a second top-level entry.
autoapi_add_toctree_entry = False
autoapi_python_class_content = "both"  # merge class + __init__ docstrings
autoapi_member_order = "groupwise"
# ``imported-members`` would cause every re-exported symbol to appear twice
# (once in the source module, once in the re-exporting __init__). Leaving
# it off relies on users following the canonical source module link.
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_ignore = ["*/_build/*", "*/tests/*"]

# -- Napoleon (docstring parsing) --------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_rtype = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_baseurl = "https://josiahwsmith10.github.io/complextorch/"
html_title = f"complextorch {release}"

html_context = {
    "github_user": "josiahwsmith10",
    "github_repo": "complextorch",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_theme_options = {
    "github_url": "https://github.com/josiahwsmith10/complextorch",
    "use_edit_page_button": True,
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "show_nav_level": 2,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/complextorch/",
            "icon": "fa-brands fa-python",
        },
    ],
    "switcher": {
        # sphinx-multiversion puts each version's static assets under
        # /<version>/_static/, so the switcher manifest lives at
        # /latest/_static/switcher.json (we alias main -> latest in CI).
        "json_url": "https://josiahwsmith10.github.io/complextorch/latest/_static/switcher.json",
        "version_match": release,
    },
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright", "sphinx-version"],
    "footer_end": ["theme-version"],
}

# -- OpenGraph & sitemap -----------------------------------------------------

ogp_site_url = html_baseurl
ogp_image = None  # add a social card later if a logo is created
sitemap_url_scheme = "{link}"

# -- sphinx-multiversion -----------------------------------------------------

# Whitelist any 2.x.y release. Older tags from before the migration to this
# conf.py / dependency set remain accessible via PyPI but are not re-rendered
# here. Bump the major when cutting a 3.x.
smv_tag_whitelist = r"^2\.\d+\.\d+$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = None
smv_released_pattern = r"^refs/tags/.*$"
smv_outputdir_format = "{ref.name}"
smv_prefer_remote_refs = False
