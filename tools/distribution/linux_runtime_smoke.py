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
    from PySide6 import QtCore, QtWidgets
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
    plotter = QtInteractor(window, off_screen=False, auto_update=False)
    configure_pyvista_interactor(plotter)
    try:
        window.setCentralWidget(plotter.interactor)
        window.resize(320, 240)
        window.show()
        application.processEvents()
        plotter.set_background("white")
        plotter.add_mesh(pv.Sphere(), color="royalblue")
        plotter.reset_camera()
        # An exposed Qt surface and its first paint are asynchronous under Xvfb.
        # Keep the event loop running until the scene has produced a real frame.
        loop = QtCore.QEventLoop()
        elapsed = QtCore.QElapsedTimer()
        elapsed.start()
        pixels = None
        rendered = False
        capture_error = None

        def capture_frame() -> None:
            nonlocal pixels, capture_error, rendered
            if window.windowHandle().isExposed():
                try:
                    plotter.render()
                    pixels = np.asarray(plotter.screenshot(return_img=True))
                    capture_error = None
                    if (
                        pixels.ndim == 3
                        and pixels.shape[-1] >= 3
                        and np.ptp(pixels[..., :3], axis=(0, 1)).max() > 0
                    ):
                        rendered = True
                        loop.quit()
                        return
                except RuntimeError as error:
                    capture_error = str(error)
            if elapsed.elapsed() >= 5000:
                loop.quit()

        timer = QtCore.QTimer()
        timer.setInterval(50)
        timer.timeout.connect(capture_frame)
        timer.start()
        loop.exec()
        timer.stop()
        if not rendered:
            print(plotter.ren_win.ReportCapabilities(), flush=True)
            if pixels is not None:
                from PIL import Image

                Image.fromarray(pixels).save("vtk-smoke-failure.png")
            raise AssertionError(
                f"VTK returned no rendered scene within 5 seconds: {capture_error}"
            )
    finally:
        plotter.close()
        window.close()
    result.update(qt_canvas="ok", vtk_render="ok")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
