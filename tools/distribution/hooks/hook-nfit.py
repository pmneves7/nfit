"""Collect public scripting modules, optional CPU kernels, and release metadata."""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

hiddenimports = collect_submodules("nfit")
datas = collect_data_files("nfit") + copy_metadata("nfit")
# Numba's cache locators need real source files in the installed bundle.
module_collection_mode = {"nfit": "py", "numba": "pyz+py"}
