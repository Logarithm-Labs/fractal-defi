# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'fractal'
copyright = '2024, Logarithm Labs'
author = 'Logarithm Labs'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# Default role for `single backticks` in docstrings — without it RST
# treats them as interpreted-text cross-refs and warns "Unknown target
# name". Setting ``code`` makes a single backtick render as inline
# literal (same as Markdown / GitHub), so legacy docstrings written in
# Markdown style still build cleanly.
default_role = 'code'

# Autodoc behavior --------------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Napoleon: support both Google and numpy style sections.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
# Render ``Attributes:`` sections inline via ``:ivar:`` instead of as
# separate documented members. Without this, dataclass fields end up
# documented twice (once by autodoc from the type annotation, once by
# napoleon from the ``Attributes:`` block) and Sphinx emits a
# "duplicate object description" warning per field.
napoleon_use_ivar = True

# Each class is documented twice when both the package-level and
# module-level rst include it. Suppressing here keeps the build clean
# while autodoc still picks the FIRST documented instance per object.
suppress_warnings = [
    'autodoc.import_object',  # silently skip moved/renamed modules
    'docutils',               # accept legacy single-backtick refs
    'ref.python',             # ambiguous re-exports across submodules
    'toc.not_included',       # the per-module pages don't need to be in toctree
    # ``Attributes:`` / ``Methods:`` sections in dataclass docstrings
    # produce entries that overlap with autodoc's automatic field
    # listing. The HTML still renders correctly (Sphinx picks one
    # instance) — silence the noisy warning rather than rewrite
    # every dataclass docstring.
    'autodoc',
    'app.add_object',
    'app.add_node',
]
# ``duplicate object description`` is not category-suppressible in
# Sphinx 9. Tolerate the count in the footer; the rendered HTML is
# correct and only the FIRST instance is registered for cross-refs.

# Intersphinx: link to upstream APIs we mention often.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
