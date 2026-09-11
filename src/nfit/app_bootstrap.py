"""Entry point for standalone installers, scripted work, and frozen workers."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


def smoke_test() -> dict:
    """Exercise the bundled GUI, local help, data archive, optimizer, and JIT."""
    import tempfile

    import h5py
    import numpy as np
    from PySide6 import QtWidgets
    from scipy.optimize import least_squares

    try:
        from startup_splash import StartupSplash
    except ModuleNotFoundError:  # Source-tree smoke tests.
        from tools.distribution.startup_splash import StartupSplash

    from .app_distribution import application_version, local_help_index, platform_key
    from .project_gui import _qt_app
    from .rebin import rebin_nd

    app = _qt_app()
    assert not app.windowIcon().isNull()
    splash = StartupSplash()
    splash_logo = splash.findChild(QtWidgets.QLabel, "startup_logo")
    assert splash_logo is not None and splash_logo.pixmap() is not None
    assert not splash_logo.pixmap().isNull()
    splash.close()
    index = local_help_index()
    assert index is not None and index.is_file()
    assert (index.parent / "_static/mathjax/tex-svg-full.js").is_file()
    with tempfile.TemporaryDirectory() as directory:
        with h5py.File(Path(directory) / "smoke.h5", "w") as stream:
            stream["signal"] = np.arange(8.0)
        with h5py.File(Path(directory) / "smoke.h5", "r") as stream:
            assert stream["signal"][:].sum() == 28
    assert np.allclose(least_squares(lambda x: x - 2, [0.0]).x, [2.0])
    arguments = dict(data=np.array([1., 2., 3.]), coords=np.array([[0.1], [0.4], [0.8]]),
                     data_errs=np.ones(3), lower=[0.], upper=[1.], num_bins=[2])
    reference = rebin_nd(**arguments, backend="numpy")
    compiled = rebin_nd(**arguments, backend="numba")
    assert compiled.resolved_backend == "numba"
    np.testing.assert_allclose(reference.binned_data, compiled.binned_data)
    import pyvista as pv
    from pyvistaqt import QtInteractor

    assert QtInteractor is not None and pv.Sphere().n_points > 0
    return {
        "version": application_version(),
        "platform": platform_key(),
        "help": str(index),
        "splash": "ok",
        "status": "ok",
    }


def main(argv: list[str] | None = None, *, startup_splash=None) -> int:
    parser = argparse.ArgumentParser(description="nfit desktop application")
    parser.add_argument("--smoke-test", action="store_true", help="Verify bundled runtime components without opening a project.")
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--run-script", type=Path, help="Run an exported Python script using the bundled nfit runtime.")
    parser.add_argument("--benchmark-worker", nargs=4, metavar=("SNAPSHOT", "OUTPUT", "MIB", "WORKERS"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.benchmark_worker:
        from .performance_benchmark import _trial

        snapshot, output, mib, workers = args.benchmark_worker
        _trial(snapshot, output, int(mib), int(workers))
        return 0
    if args.run_script:
        sys.argv = [str(args.run_script)]
        runpy.run_path(str(args.run_script), run_name="__main__")
        return 0
    if args.smoke_test:
        result = smoke_test()
        if args.smoke_output:
            args.smoke_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if sys.stdout is not None:
            print(json.dumps(result))
        return 0
    if startup_splash is not None:
        startup_splash.show_status("Loading scientific and graphical components…")
    from .project_gui import main as launch

    if startup_splash is not None:
        startup_splash.show_status("Building the nfit project explorer…")
    return launch()
