from pathlib import Path

import pytest


@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer
    from tests.plotting_test_data import tiny_mdhisto_data

    value = QtMDHistoSliceViewer(tiny_mdhisto_data(), x_dim=3, y_dim=2)
    yield value
    value.window.close()


def test_save_button_matches_neighbors_and_uses_public_export(viewer, monkeypatch, tmp_path):
    from nfit import qt_figure_export as gui
    from nfit.figure_export import DEFAULT_FIGURE_DPI

    layout = viewer.copy_figure_button.parentWidget().layout()
    assert layout.getItemPosition(layout.indexOf(viewer.copy_figure_button)) == (5, 0, 1, 2)
    assert layout.getItemPosition(layout.indexOf(viewer.save_figure_button)) == (5, 2, 1, 2)
    assert viewer.save_figure_button.sizePolicy() == viewer.save_data_button.sizePolicy()
    assert str(DEFAULT_FIGURE_DPI) in viewer.save_figure_button.toolTip()
    assert viewer.save_figure_button.text() == "Save figure"
    destination = tmp_path / "figure.svg"
    calls = []
    choices = []

    def choose(*args, **kwargs):
        choices.append((args, kwargs))
        return str(destination), ""

    def export(figure, path):
        calls.append((figure, path))
        return Path(path)

    monkeypatch.setattr(gui, "get_save_file_name", choose)
    monkeypatch.setattr(gui, "save_figure", export)
    viewer.save_figure_button.click()
    assert calls == [(viewer.figure, destination)]
    assert viewer._figure_export_path == destination
    assert choices[0][1]["require_selected_filter"] is True
    for suffix in ("png", "eps", "tiff", "svg", "jpg"):
        assert "*." + suffix in choices[0][0][3]
    script = viewer.figure_script()
    assert f"save_figure(plt.gcf(), {str(destination)!r}, dpi={DEFAULT_FIGURE_DPI})" in script


def test_selected_format_adds_extension_and_saves_actual_vector(viewer, monkeypatch, tmp_path):
    from nfit import qt_figure_export as gui
    from nfit.figure_export import figure_export_filters

    svg_filter = next(label for label, suffix in figure_export_filters(viewer.figure).items() if suffix == ".svg")
    destination = tmp_path / "vector figure"
    monkeypatch.setattr(gui, "get_save_file_name", lambda *_args, **_kwargs: (str(destination), svg_filter))
    result = viewer.save_figure()
    assert result == destination.with_suffix(".svg")
    assert "<svg" in result.read_text()


def test_cancel_keeps_previous_export_and_does_not_write(viewer, monkeypatch):
    from nfit import qt_figure_export as gui

    previous = Path("previous.png")
    viewer._figure_export_path = previous
    monkeypatch.setattr(gui, "get_save_file_name", lambda *_args, **_kwargs: ("", ""))
    monkeypatch.setattr(gui, "save_figure", lambda *_args: pytest.fail("cancel exported a file"))
    assert viewer.save_figure() is None
    assert viewer._figure_export_path == previous


def test_added_extension_confirms_overwrite_before_writing(viewer, monkeypatch, tmp_path):
    from PySide6 import QtWidgets

    from nfit import qt_figure_export as gui

    destination = tmp_path / "existing.png"
    destination.write_bytes(b"original")
    monkeypatch.setattr(gui, "get_save_file_name", lambda *_args, **_kwargs: (str(destination.with_suffix("")), ""))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *_args: QtWidgets.QMessageBox.StandardButton.No)
    assert viewer.save_figure() is None
    assert destination.read_bytes() == b"original"


def test_export_error_is_reported_and_does_not_record_destination(viewer, monkeypatch, tmp_path):
    from PySide6 import QtWidgets

    from nfit import qt_figure_export as gui

    destination = tmp_path / "invalid.xyz"
    warnings = []
    monkeypatch.setattr(gui, "get_save_file_name", lambda *_args, **_kwargs: (str(destination), ""))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args))
    assert viewer.save_figure() is None
    assert not destination.exists()
    assert viewer._figure_export_path is None
    assert len(warnings) == 1
    assert "Could not save figure" in warnings[0][2]
