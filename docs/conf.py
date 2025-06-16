# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QuGrad'
copyright = '2025, Christopher K. Long'
author = 'Christopher K. Long'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode',
              'sphinx.ext.duration',
              'sphinx.ext.mathjax',
              'sphinx_math_dollar',
              'myst_parser',
              'stubdoc',
              'sphinx_tabs.tabs',
              'sphinx_codeautolink',
              'sphinx.ext.intersphinx']
myst_enable_extensions = ["dollarmath"]

autosummary_generate = True # Turn on sphinx.ext.autosummary
autosummary_imported_members = True
# autodoc_inherit_docstrings = True
autodoc_member_order = "groupwise"
napoleon_include_init_with_doc = True
napoleon_numpy_docstring = True
viewcode_line_numbers = True
myst_heading_anchors = 5
suppress_warnings = ["myst.header"]
module_names = ["qugrad.systems._systems"]
sphinx_tabs_disable_tab_closing = True
toc_object_entries_show_parents = 'hide'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

mathjax3_config = {
    'loader': {'load': ['[tex]/mathtools', '[tex]/physics']},
    'tex': {
        'packages': {'[+]': ['mathtools', 'physics']},
        'inlineMath': [['\\(', '\\)']],
        'displayMath': [['\\[', '\\]']],
    }
}

intersphinx_mapping = {
    "py_ste": (
        "https://PySTE.readthedocs.io/en/latest",
        None,
    ),
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

import os
import sys
import subprocess
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))


DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.check_call(f"cd {DIR}; cffconvert -f apalike > docs/citation/citation_files/citation.txt", shell=True)
subprocess.check_call(f"cd {DIR}; cffconvert -f bibtex > docs/citation/citation_files/citation.bib", shell=True)
subprocess.check_call(f"cd {DIR}; cffconvert -f ris > docs/citation/citation_files/citation.ris", shell=True)
subprocess.check_call(f"cd {DIR}; cffconvert -f codemeta > docs/citation/citation_files/citation_codemeta.json", shell=True)
subprocess.check_call(f"cd {DIR}; cffconvert -f endnote > docs/citation/citation_files/citation.enw", shell=True)