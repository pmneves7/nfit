from __future__ import annotations

import math

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from nfit.figure_export import (
    DEFAULT_FIGURE_DPI,
    figure_export_filters,
    figure_export_script,
    save_figure,
)


def _figure() -> Figure:
    figure = Figure(figsize=(2, 1.5), dpi=80)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot([0, 1], [2, 3])
    axes.set_xlim(-0.25, 1.25)
    axes.set_ylim(1.5, 3.5)
    return figure


@pytest.mark.parametrize("suffix", ["png", "tif", "jpg", "svg", "eps", "pdf"])
def test_save_figure_writes_supported_formats(tmp_path, suffix):
    figure = _figure()
    output = tmp_path / f"plot.{suffix}"

    returned = save_figure(figure, output, dpi=120)

    assert returned == output
    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize("suffix", ["png", "tiff"])
def test_raster_export_uses_requested_resolution_and_dpi(tmp_path, suffix):
    output = save_figure(_figure(), tmp_path / f"plot.{suffix}", dpi=200)

    with Image.open(output) as image:
        assert image.size == (400, 300)
        assert image.info["dpi"] == pytest.approx((200, 200), rel=0.01)


def test_save_figure_defaults_to_png_and_preserves_figure_state(tmp_path):
    figure = _figure()
    original_size = figure.get_size_inches().copy()
    original_dpi = figure.dpi
    original_limits = [
        (axes.get_xlim(), axes.get_ylim()) for axes in figure.axes
    ]

    output = save_figure(figure, tmp_path / "plot", dpi=90)

    assert output == tmp_path / "plot.png"
    assert figure.get_size_inches().tolist() == original_size.tolist()
    assert figure.dpi == original_dpi
    assert [(axes.get_xlim(), axes.get_ylim()) for axes in figure.axes] == (
        original_limits
    )
    assert DEFAULT_FIGURE_DPI == 600


@pytest.mark.parametrize("dpi", [0, -1, math.inf, math.nan, True, "300"])
def test_save_figure_rejects_invalid_dpi_before_writing(tmp_path, dpi):
    output = tmp_path / "plot.png"

    with pytest.raises(ValueError, match="finite positive"):
        save_figure(_figure(), output, dpi=dpi)

    assert not output.exists()


def test_save_figure_rejects_unsupported_suffix(tmp_path):
    output = tmp_path / "plot.not-a-format"

    with pytest.raises(ValueError, match="unsupported figure format"):
        save_figure(_figure(), output)

    assert not output.exists()


def test_figure_export_filters_use_backend_groups_and_exclude_internal_formats():
    figure = _figure()
    filters = figure_export_filters(figure)
    labels = list(filters)

    assert labels[0].startswith("Portable Network Graphics")
    assert any("*.jpeg" in label and "*.jpg" in label for label in labels)
    assert any("*.tif" in label and "*.tiff" in label for label in labels)
    assert all("Raw RGBA" not in label and "PGF" not in label for label in labels)
    assert filters[labels[0]] == ".png"


def test_figure_export_script_saves_real_figure_before_show(tmp_path):
    output = tmp_path / "café's plot.png"
    script = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.figure(figsize=(1, 1))\n"
        "plt.plot([0, 1], [1, 2])\n"
        "plt.show()\n"
    )

    exported = figure_export_script(script, output)
    assert exported.index("save_figure(plt.gcf()") < exported.rindex("plt.show()")
    exec(compile(exported, "plot_script.py", "exec"), {})

    assert output.is_file()
    assert output.stat().st_size > 0


def test_figure_export_script_without_path_is_unchanged():
    script = "import matplotlib.pyplot as plt\nplt.show()\n"

    assert figure_export_script(script, None) == script
