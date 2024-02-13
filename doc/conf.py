"""Khiops Python Sphinx configuration file"""
import os
import sys
from datetime import datetime

import numpydoc

# Add the root of the repository and the samples directory to sys.path
# so Sphinx can find both khiops and the samples scripts
sys.path.append("..")
sys.path.append("../khiops/samples")
import khiops

project = "Khiops"
copyright = f"2018-{datetime.today().year}, Orange"
author = "The Khiops Team"

# The full version, including alpha/beta/rc tags
release = khiops.__version__

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
    "inherited-members": False,
    "private-members": False,
    "show-inheritance": True,
    "special-members": False,
}

## Intersphinx extension config
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

## Autosummary extension config
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and directories to
# ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_templates", "_build", "Thumbs.db", ".DS_Store"]

# HTML Theme
# Theme colors and fonts come from https://brand.orange.com
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#FF7900",
        "color-brand-content": "#F16E00",
        "color-sidebar-background": "#FFFFFF",
        "color-highlighted-background": "#FFD200",
        "color-admonition-title--note": "#FF7900",
        "color-admonition-title-background--note": "#FFF0E2",
        "font-stack": "Helvetica Neue, Helvetica, sans-serif",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FF7900",
        "color-brand-content": "#F16E00",
        "color-sidebar-background": "#000000",
        "color-highlighted-background": "#FFD200",
        "color-admonition-title--note": "#FF7900",
        "color-admonition-title-background--note": "#CC6100",
        "font-stack": "Helvetica Neue, Helvetica, sans-serif",
    },
    "source_repository": "https://github.com/khiopsml/khiops/",
    # Sets the Github Icon (the SVG is embedded, copied from furo's repo)
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/khiopsml/khiops",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
html_title = f"<h6><center>{project} {release}</center></h6>"

# HTML static assets
html_static_path = ["_static"]
html_logo = "_static/images/orange_small_logo.png"
html_css_files = ["css/custom.css"]


# Callback to Suppress warnings:
# - about sklearn code (`X` or `y`) included via intersphinx
# - about some literals included via the tutorials transformation
def suppress_sklearn_warnings(app, env, node, contnode):
    def sklearn_not_found_variable(node):
        return (
            node.rawsource == "`X`"
            or node.rawsource == "`y`"
            or node.rawsource == '`"default"`'
            or node.rawsource == '`"pandas"`'
        ) and node.attributes["py:module"] == "khiops.sklearn.estimators"

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
