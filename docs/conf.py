project = "nfit"
author = "Paul M. Neves"
copyright = "2026, Paul M. Neves"
release = "0.93.7"

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

# SVG embeds the equation glyphs, so local help does not fetch web fonts.
mathjax_path = "mathjax/tex-svg-full.js"
mathjax3_config = {"svg": {"fontCache": "local"}}

html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "nfit-logo.svg",
    "dark_logo": "nfit-logo-dark.svg",
}
html_favicon = "_static/nfit-icon.png"
