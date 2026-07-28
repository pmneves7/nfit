project = "nfit"
author = "Paul M. Neves"
copyright = "2026, Paul M. Neves"
release = "0.29.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
html_theme = "furo"
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 4
