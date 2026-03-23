import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project   = "mentor"
author    = "mentor contributors"
copyright = f"2024, {author}"
release   = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "nbsphinx",
]

# Napoleon — NumPy docstrings only
napoleon_numpy_docstring   = True
napoleon_google_docstring  = False
napoleon_use_param         = True
napoleon_use_rtype         = True

# autodoc — types are documented in NumPy sections; suppress annotation injection
autodoc_member_order       = "bysource"
autodoc_typehints          = "none"
autoclass_content          = "both"
autodoc_default_options    = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "special-members":  "__init__, __repr__, __str__",
}

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch":  ("https://pytorch.org/docs/stable", None),
}

# MyST
myst_enable_extensions = ["colon_fence", "deflist"]

html_theme         = "sphinx_rtd_theme"
html_static_path   = ["_static"]
html_title         = "mentor"
html_show_sourcelink = True


# single-page HTML
singlehtml_sidebars = {"**": ["globaltoc.html", "relations.html"]}

# LaTeX / PDF
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
}
latex_documents = [
    ("index", "mentor.tex", "mentor Documentation", author, "manual"),
]
