"""Collect public scripting modules, optional CPU kernels, and release metadata."""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

hiddenimports = collect_submodules("nfit")
# Figure export selects these backends dynamically from the output suffix.
hiddenimports += [
    "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_ps",
    "matplotlib.backends.backend_pdf",
]
datas = collect_data_files("nfit") + copy_metadata("nfit")
# Numba's cache locators need real source files in the installed bundle.
module_collection_mode = {"nfit": "py", "numba": "pyz+py"}
