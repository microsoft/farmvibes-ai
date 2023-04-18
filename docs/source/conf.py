# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FarmVibes.AI"
copyright = "2023, Microsoft"
author = "Microsoft"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.mermaid",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_member_order = "groupwise"
myst_heading_anchors = 3
typehints_use_rtype = False
typehints_defaults = "comma"

sys.path.insert(0, os.path.abspath("../../src"))

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
