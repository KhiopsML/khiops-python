"""pyKhiops Sphinx configuration file"""
import os
import sys
from datetime import datetime

# Add the root of the repository and the samples directory to sys.path
# so Sphinx can find both pykhiops and the samples scripts
sys.path.append("..")
sys.path.append("./samples")

import pykhiops

project = "pyKhiops"
copyright = f"2018-{datetime.today().year}, Orange"
author = "Orange Innovation"

# The full version, including alpha/beta/rc tags
release = pykhiops.__version__

# Be strict about any broken references
nitpicky = True

# To avoid using qualifiers like :class: to reference objects within the same context
default_role = "obj"

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
]

## Numpydoc extension config
numpydoc_show_class_members = False

## Autodoc extension config
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "private-members": False,
    "show-inheritance": True,
    "special-members": False,
}

# Intersphinx extension config
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

## Autosummary extension config
autosummary_generate = True
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and directories to
# ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML Theme
html_theme = "furo"

# HTML static pages
html_static_path = []

# Suppress warning about sklearn code (`X` or `y`) included via intersphinx
def suppress_sklearn_warnings(app, env, node, contnode):
    if (node.rawsource == "`X`" or node.rawsource == "`y`") and node.source.endswith(
        "Mixin.score"
    ):
        return contnode
    return None


def setup(app):
    app.connect("missing-reference", suppress_sklearn_warnings)
