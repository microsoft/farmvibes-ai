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
    "sphinxcontrib.openapi",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_member_order = "groupwise"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_inherit_docstrings = False
autoclass_content = "class"
myst_heading_anchors = 3
typehints_document_rtype = False
typehints_use_rtype = False
typehints_defaults = "comma"
always_use_bars_union = True
napoleon_use_rtype = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False

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
