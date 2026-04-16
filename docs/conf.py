"""Sphinx configuration for mentor documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

_version: dict = {}
exec(open("../mentor/version.py").read(), _version)

project = "mentor"
copyright = "anguelos"
author = "anguelos"
release = _version["__version__"]
version = release

extensions = [
    # "sphinx.ext.autodoc", This causes bad rst rendering of docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = ["colon_fence"]

# ---------------------------------------------------------------------------
# intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# ---------------------------------------------------------------------------
# autodoc / autosummary
# ---------------------------------------------------------------------------
#autosummary_generate = True  # This causes duplicate entries in the API reference and badly rendered docstrings, so we will generate the autosummary tables manually for now
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# ---------------------------------------------------------------------------
# napoleon (numpy docstrings)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# ---------------------------------------------------------------------------
# linkcode — point [source] links to GitHub
# ---------------------------------------------------------------------------
_GITHUB_ROOT = "https://github.com/anguelos/torch_mentor/blob/main"


def linkcode_resolve(domain, info):
    """Map a Python object to its GitHub source URL."""
    if domain != "py" or not info["module"]:
        return None
    import importlib
    import inspect

    try:
        mod = importlib.import_module(info["module"])
    except ImportError:
        return None

    obj = mod
    for part in (info.get("fullname") or "").split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    # Unwrap decorated / bound objects
    obj = inspect.unwrap(obj)

    try:
        src_file = inspect.getfile(obj)
        lines, start = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None

    # Make path relative to the repo root
    try:
        rel = os.path.relpath(src_file, os.path.join(os.path.dirname(__file__), ".."))
    except ValueError:
        return None

    end = start + len(lines) - 1
    return f"{_GITHUB_ROOT}/{rel}#L{start}-L{end}"


# ---------------------------------------------------------------------------
# copybutton
# ---------------------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
