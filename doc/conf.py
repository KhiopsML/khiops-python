"""pyKhiops Sphinx configuration file"""
import os
import sys
from datetime import datetime

import numpydoc

# Add the root of the repository and the samples directory to sys.path
# so Sphinx can find both pykhiops and the samples scripts
sys.path.append("..")
sys.path.append("../pykhiops/samples")
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
exclude_patterns = ["_templates", "_build", "Thumbs.db", ".DS_Store"]

# HTML Theme
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#FF7900",
        "color-brand-content": "#F16E00",
        "color-highlighted-background": "#FFD200",
        "color-admonition-title--note": "#FF7900",
        "color-admonition-title-background--note": "#FFF0E2",
        "font-stack": "Helvetica Neue, sans-serif",
    },
}
html_title = f"<h6>{project} {release}</h6>"
html_logo = "./khiops_logo.png"

# HTML static pages
html_static_path = []


# Suppress warnings:
# - about sklearn code (`X` or `y`) included via intersphinx
# - about some literals included via the tutorials transformation
def suppress_sklearn_warnings(app, env, node, contnode):
    def sklearn_not_found_variable(node):
        return (
            node.rawsource == "`X`" or node.rawsource == "`y`"
        ) and node.source.endswith("Mixin.score")

    def tutorial_literal(node):
        return (
            node.rawsource == "`ProbClassIris-setosa`"
            or node.rawsource == "`ProbClassIris-versicolor`"
            or node.rawsource == "`ProbClassIris-virginica`"
        )

    if sklearn_not_found_variable(node) or tutorial_literal(node):
        return contnode
    return None


def setup(app):
    app.connect("missing-reference", suppress_sklearn_warnings)
