"""Exercise the extracted Linux application's Qt canvas and VTK renderer.

Run with ``nfit --run-script tools/distribution/linux_runtime_smoke.py`` under
an X11 desktop or Xvfb. This complements the application's numerical smoke
test by requiring rendered pixels from its bundled Qt/VTK dependencies.
"""

from __future__ import annotations

import json


def main() -> None:
    import numpy as np
    import pyvista as pv
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6 import QtWidgets
    from pyvistaqt import QtInteractor

    from nfit.app_bootstrap import smoke_test
    from nfit.qt_pyvista import configure_pyvista_interactor

    result = smoke_test()
    application = QtWidgets.QApplication.instance()
    assert application is not None
    figure = Figure(figsize=(2, 2))
    figure.add_subplot().plot([0, 1, 2], [0, 1, 0])
    canvas = FigureCanvasQTAgg(figure)
    canvas.draw()
    assert np.asarray(canvas.buffer_rgba())[..., :3].std() > 0
    canvas.close()

    window = QtWidgets.QMainWindow()
    plotter = QtInteractor(window, off_screen=True, auto_update=False)
    configure_pyvista_interactor(plotter)
    try:
        window.setCentralWidget(plotter.interactor)
        window.resize(320, 240)
        window.show()
        application.processEvents()
        plotter.set_background("white")
        plotter.add_mesh(pv.Sphere(), color="royalblue")
        plotter.reset_camera()
        plotter.render()
        pixels = np.asarray(plotter.screenshot(return_img=True))
        assert pixels.ndim == 3 and pixels.shape[-1] >= 3
        assert pixels[..., :3].std() > 0, "VTK returned a blank image"
    finally:
        plotter.close()
        window.close()
    result.update(qt_canvas="ok", vtk_render="ok")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
