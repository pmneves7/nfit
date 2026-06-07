project = "metallix"
author = "metallix contributors"
copyright = "2026, metallix contributors"
release = "0.0.1"

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

