project = "nfit"
author = "Paul M. Neves"
copyright = "2026, Paul M. Neves"
release = "0.5.2"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
master_doc = "index"
html_theme = "furo"
myst_enable_extensions = ["dollarmath", "amsmath"]
nb_execution_mode = "off"
