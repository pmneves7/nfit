"""Keep JIT, plotting caches, and windowless logs outside the signed app bundle."""
import os
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "nfit"
elif sys.platform == "darwin":
    root = Path.home() / "Library/Caches/nfit"
else:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nfit"
try:
    root.mkdir(parents=True, exist_ok=True)
except OSError:
    root = Path(tempfile.gettempdir()) / "nfit-cache"
    root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(root / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(root / "matplotlib"))
if sys.stdout is None or sys.stderr is None:
    log = (root / "application.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
