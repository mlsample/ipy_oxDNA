import sys
import os
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('../ipy_oxdna'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme'
]
html_theme = 'sphinx_rtd_theme'
autodoc_member_order = 'bysource'
html_logo = os.path.abspath('../src/oxDNA.png')
html_domain_indices = True


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ipy_oxdna'
copyright = '2023, Matthew Sample'
author = 'Matthew Sample'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
