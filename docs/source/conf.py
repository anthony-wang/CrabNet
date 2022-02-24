#!/usr/bin/env python3
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from os.path import dirname, abspath, join
import sys
import sphinx_rtd_theme  # noqa

try:
    # mat_discover is installed
    import crabnet
except ImportError:
    # mat_discover is run from its source checkout
    current_dir = dirname(__file__)
    mypath = abspath(join(current_dir, "..", "..", "crabnet"))
    mypath2 = abspath(join(current_dir, "..", ".."))
    sys.path.insert(0, mypath)
    sys.path.insert(0, mypath2)
    print("path added: ", mypath)
    print("path added: ", mypath2)
    import crabnet

    # try:
    #     import mat_discover
    # except ImportError:
    #     warn("You might need to run `conda install flit; flit install --pth-file`")


# -- Project information -----------------------------------------------------

project = "crabnet"
copyright = "2022, Anthony Wang, Steven Kauwe, Sterling Baird, Andrew Falkowski"
author = "Anthony Wang, Steven Kauwe, Sterling Baird, Andrew Falkowski"

# The full version, including alpha/beta/rc tags
version = ".".join(crabnet.__version__.split(".")[:2])
release = crabnet.__version__


# -- General configuration ---------------------------------------------------

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directive-option-automodule-special-members https://stackoverflow.com/a/61732050/13697228
autodoc_default_options = {"special-members": "__init__"}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",  # both NumPy and Google docstring support
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    # "sphinx.ext.linkcode", #
    # https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html, e.g.
    # non-GitHub links
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list = []

# https://sublime-and-sphinx-guide.readthedocs.io/en/latest/code_blocks.html
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "logo.png"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
}

html_extra_path = ["googlea441f484329a8f75.html"]

autodoc_mock_imports = ["torch"]

# https://github.com/sphinx-doc/sphinx/issues/7000#issuecomment-677916705
source_suffix = [".rst", ".md"]
