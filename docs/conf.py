project = "nfit"
author = "Paul M. Neves"
copyright = "2026, Paul M. Neves"
release = "0.80.3"

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

html_static_path = ["_static"]
html_logo = "_static/nfit-logo.svg"
html_favicon = "_static/nfit-icon.png"
html_css_files = ["branding.css"]
