import numpy as np

from nfit.backgrounds import subtract_aligned_background
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.pipeline import DatasetEntry
from nfit.plotting_core import MDHistoSliceViewer, _mdhisto_channel_array
from nfit.project_masks import _mdhisto_with_nfit_masks


def _data(signal: float, error: float) -> MDHistoData:
    shape = (2, 2, 3)
    return MDHistoData(
        axes=(
            MDHistoAxis("H", np.arange(3.0), "rlu", "momentum"),
            MDHistoAxis("K", np.arange(3.0), "rlu", "momentum"),
            MDHistoAxis("E", np.arange(4.0), "meV", "energy"),
        ),
        signal=np.full(shape, signal),
        errors=np.full(shape, error),
        mask=np.zeros(shape, dtype=bool),
        num_events=np.ones(shape),
        metadata={
            "signal_unit": "counts",
            "signal_quantity_type": "scattering_intensity",
        },
    )


def test_background_diagnostics_are_available_and_integrate_like_signal():
    original = _data(10.0, 2.0)
    result = subtract_aligned_background(original, _data(3.0, 1.0))

    background = MDHistoSliceViewer(
        result, x_dim=2, y_dim=0, channel="background", integrate=True
    )
    background.integrate_checks[1] = True
    background.selections[1] = (0.0, 2.0)
    background_view = background.slice_arrays()
    np.testing.assert_allclose(background_view["background"], 6.0)
    np.testing.assert_allclose(background_view["background_errors"], np.sqrt(2.0))

    unsubtracted = MDHistoSliceViewer(
        result, x_dim=2, y_dim=0, channel="unsubtracted", integrate=True
    )
    unsubtracted.integrate_checks[1] = True
    unsubtracted.selections[1] = (0.0, 2.0)
    original_view = unsubtracted.slice_arrays()
    np.testing.assert_allclose(original_view["unsubtracted"], 20.0)
    np.testing.assert_allclose(
        original_view["unsubtracted_errors"], np.sqrt(8.0)
    )
    assert "Background" in background._channel_label()
    assert "Unsubtracted" in unsubtracted._channel_label()


def test_normal_slice_does_not_reconstruct_unsubtracted(monkeypatch):
    result = subtract_aligned_background(_data(10.0, 2.0), _data(3.0, 1.0))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("background diagnostic was computed eagerly")

    monkeypatch.setattr("nfit.plotting_core.background_channel", fail_if_called)
    view = MDHistoSliceViewer(result, x_dim=2, y_dim=0).slice_arrays()
    np.testing.assert_allclose(view["signal"], 7.0)
    assert "background" not in view
    assert "unsubtracted" not in view


def test_unsubtracted_view_restores_valid_original_where_background_was_missing():
    original = _data(10.0, 2.0)
    background = _data(3.0, 1.0).mutable_copy()
    background.mask[0, 1, 2] = True
    result = subtract_aligned_background(original, background)
    result = _mdhisto_with_nfit_masks(DatasetEntry("difference", result))

    primary = MDHistoSliceViewer(result, x_dim=2, y_dim=0).slice_arrays()
    assert np.isnan(primary["signal"][0, 2])
    restored = MDHistoSliceViewer(
        result, x_dim=2, y_dim=0, channel="unsubtracted"
    ).slice_arrays()
    assert restored["unsubtracted"][0, 2] == 10.0
    assert restored["unsubtracted_errors"][0, 2] == 2.0


def test_ordinary_background_auxiliary_channel_needs_no_provenance_marker():
    data = _data(4.0, 1.0).with_updates(
        auxiliary_channels={
            "background": MDHistoChannel(
                np.full((2, 2, 3), 9.0),
                np.full((2, 2, 3), 0.5),
                "Imported background",
                "counts",
                "scattering_intensity",
            )
        }
    )
    viewer = MDHistoSliceViewer(data, x_dim=2, y_dim=0, channel="background")
    view = viewer.slice_arrays()
    np.testing.assert_allclose(view["background"], 9.0)
    assert "unsubtracted" not in viewer.CHANNELS


def test_channel_array_applies_selection_before_materializing_optional_mask():
    data = _data(4.0, 1.0)
    selected = _mdhisto_channel_array(data, "signal", (0, slice(None), 1))
    np.testing.assert_allclose(selected, [4.0, 4.0])
    absent_mask = _mdhisto_channel_array(data, "nfit_mask", (0, slice(None), 1))
    assert absent_mask.shape == (2,)
    np.testing.assert_array_equal(absent_mask, False)


def test_qt_switches_to_unsubtracted_and_exports_displayed_selection(tmp_path):
    import pytest

    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    result = subtract_aligned_background(_data(10.0, 2.0), _data(3.0, 1.0))
    viewer = QtMDHistoSliceViewer(result, x_dim=2, y_dim=0)
    viewer.channel_combo.setCurrentText("unsubtracted")
    assert viewer.model.channel == "unsubtracted"
    np.testing.assert_allclose(
        viewer.model._display_values(viewer._current_slice), 10.0
    )
    assert not viewer.show_fit_check.isEnabled()

    destination = tmp_path / "unsubtracted.csv"
    viewer._csv_export_path = lambda *_args: str(destination)
    viewer.save_displayed_data()
    exported = np.genfromtxt(destination, delimiter=",", names=True)
    np.testing.assert_allclose(exported["I"], 10.0)
    np.testing.assert_allclose(exported["dI"], 2.0)
    viewer.window.close()


def test_grouped_waterfall_keeps_virtual_unsubtracted_channel():
    import pytest

    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    axes = (
        MDHistoAxis("H", np.arange(2.0), "rlu", "momentum"),
        MDHistoAxis("K", np.arange(2.0), "rlu", "momentum"),
        MDHistoAxis("E", np.arange(4.0), "meV", "energy"),
    )

    def line(value):
        return MDHistoData(
            axes=axes,
            signal=np.full((1, 1, 3), value),
            errors=np.ones((1, 1, 3)),
            mask=np.zeros((1, 1, 3), dtype=bool),
            num_events=np.ones((1, 1, 3)),
            metadata={"binning_name": "Default"},
        )

    datasets = [
        subtract_aligned_background(line(10.0), line(2.0)),
        subtract_aligned_background(line(12.0), line(3.0)),
    ]
    viewer = QtMDHistoSliceViewer(
        datasets,
        dataset_names=["first", "second"],
        dataset_group_keys=["group", "group"],
        channel="unsubtracted",
    )
    assert viewer._waterfall_1d_source_indices() == [0, 1]
    viewer.window.close()
