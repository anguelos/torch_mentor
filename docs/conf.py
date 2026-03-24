import os
import sys
import inspect
import importlib

sys.path.insert(0, os.path.abspath(".."))

# Mock heavy dependencies so autodoc can import mentor on RTD without PyTorch installed
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "tqdm",
    "matplotlib",
    "seaborn",
    "tensorboard",
    "fargv",
]

project   = "torch-mentor"
author    = "mentor contributors"
copyright = f"2024, {author}"
release   = "0.2.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",      # replaces viewcode: emits [source] links to GitHub
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
html_title         = "torch-mentor"
html_show_sourcelink = True

# RTD "Edit on GitHub" button and per-object GitHub source links
html_context = {
    "display_github": True,
    "github_user":    "anguelos",
    "github_repo":    "torch_mentor",
    "github_version": "main",
    "conf_py_path":   "/docs/",
}


def linkcode_resolve(domain, info):
    """Return a GitHub URL pointing to the source of a Python object."""
    if domain != "py":
        return None
    module = info.get("module")
    fullname = info.get("fullname")
    if not module:
        return None
    try:
        mod = importlib.import_module(module)
        obj = mod
        for part in fullname.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        # Unwrap decorated functions / classmethods so inspect can find the source
        try:
            obj = inspect.unwrap(obj)
        except Exception:
            pass
        try:
            src_file = inspect.getfile(obj)
            lines, start = inspect.getsourcelines(obj)
            anchor = f"#L{start}-L{start + len(lines) - 1}"
        except (TypeError, OSError):
            src_file = inspect.getfile(mod)
            anchor = ""
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        rel = os.path.relpath(src_file, root)
        if rel.startswith(".."):
            return None
        return f"https://github.com/anguelos/torch_mentor/blob/main/{rel}{anchor}"
    except Exception:
        return None


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
