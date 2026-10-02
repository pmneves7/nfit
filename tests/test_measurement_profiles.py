"""Matched final cuts, preserved statistics, and GUI/script/export parity."""

import json

import numpy as np
import pytest

from nfit import histogram_box_profiles, save_measurement_profile_csv
from nfit.histogram_statistics import (
    EVENT_SIGNAL_NUMERATOR,
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
    event_statistics_channels,
    normalized_event_statistics,
    selected_event_statistics,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_statistics import estimate_measurement_bin
from nfit.plotting_core import MDHistoSliceViewer, coarsen_mdhisto_view, smooth_mdhisto_view
from nfit.qt_box_cut_viewers import CutViewerContext, _profile_dataset


def count_data(c=(4, 0), n=(1, 9), *, mask=(False, False)):
    c, n = np.array(c, dtype=float)[:, None], np.array(n, dtype=float)[:, None]
    signal, variance = normalized_event_statistics(c, c, n)
    return MDHistoData(
        axes=(MDHistoAxis("y", np.arange(len(c)+1), "meV", "energy"),
              MDHistoAxis("x", np.array([0., 1.]), "r.l.u.", "momentum")),
        signal=signal, errors=np.sqrt(variance), mask=np.array(mask)[:, None],
        num_events=np.where(c > 0, 3, 0),
        metadata={EVENT_STATISTICS_KEY: dict(EVENT_STATISTICS_METADATA, policy="independent_copies"),
                  "zero_event_bins_are_measured": True, "num_events_semantics": "event_contributions",
                  "signal_unit": "U"},
        auxiliary_channels=event_statistics_channels(c, c, n),
    )


def prepare(data):
    return MDHistoSliceViewer(data, x_dim=1, y_dim=0).slice_arrays()


def profiles(view, **kwargs):
    return histogram_box_profiles(view, view["signal"], view["errors"],
        (view["x_edges"][0], view["x_edges"][-1], view["y_edges"][0], view["y_edges"][-1]), **kwargs)


def test_count_profiles_match_reference_and_keep_measured_zeros():
    result = profiles(prepare(count_data()))
    reference = estimate_measurement_bin(result.x_measurement.contract, [4, 0], [4, 0], exposure=[1, 9])
    np.testing.assert_allclose(result.x[1:], [[reference.value], [reference.standard_error]])
    np.testing.assert_allclose(selected_event_statistics(result.x_measurement.data), [[4], [4], [10]])
    assert result.value_label == "Pooled intensity"
    assert result.x_measurement.data.num_events.item() == 3  # Retained, not inferred from C.
    assert result.y[1][1] == result.y[2][1] == 0
    assert not result.y_measurement.data.mask[1]
    assert result.x_measurement.data.metadata[EVENT_STATISTICS_KEY]["policy"] == "independent_copies"
    assert not result.x_measurement.data.signal.flags.writeable


@pytest.mark.parametrize("mask,exposure,expected", [
    ((False, True), (1, 9), (4, 2, 1)),
    ((False, False), (1, 0), (4, 2, 1)),
])
def test_mask_and_unexposed_cells_remove_all_statistics(mask, exposure, expected):
    result = profiles(prepare(count_data(n=exposure, mask=mask))).x_measurement
    assert (result.data.signal.item(), result.data.errors.item(),
            result.data.auxiliary_channels[NORMALIZATION_DENOMINATOR].values.item()) == expected


def test_public_finite_mask_and_explicit_unmask_policy():
    view = prepare(count_data())
    view["mask"] = np.array([[False], [True]])
    assert profiles(view).x[1].item() == 4
    view["masks_applied"] = False
    assert profiles(view).x[1].item() == .4


def test_all_missing_and_fringe_coverage_are_not_zero_or_full_support():
    view = prepare(count_data(mask=(True, True)))
    missing = profiles(view).x_measurement.data
    assert missing.mask.item() and np.isnan(missing.signal.item())
    view = prepare(count_data())
    view["coverage_fraction"] = np.array([[.1], [.3]])
    kept = profiles(view, coverage_threshold=.1).x_measurement.data
    assert kept.auxiliary_channels["coverage_fraction"].values.item() == pytest.approx(.2)
    assert kept.signal.item() == .4
    rejected = profiles(view, coverage_threshold=.21).x_measurement.data
    assert rejected.mask.item() and np.isnan(profiles(view, coverage_threshold=.21).x[1].item())
    assert selected_event_statistics(rejected) is not None
    assert rejected.auxiliary_channels[NORMALIZATION_DENOMINATOR].values.item() == 10


def test_coarsen_then_profile_equals_direct_for_same_selected_cells():
    view = prepare(count_data())
    direct = profiles(view).x_measurement.data
    staged = profiles(coarsen_mdhisto_view(view, y_step=2)).x_measurement.data
    for first, second in zip(selected_event_statistics(direct), selected_event_statistics(staged), strict=True):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(direct.signal, staged.signal)
    np.testing.assert_allclose(direct.errors, staged.errors)


def test_rotated_bins_use_selected_pixel_reference():
    from nfit.box_cuts import box_coordinates, rotated_box_profiles
    view = prepare(count_data())
    result = profiles(view, angle=20)
    u, _ = box_coordinates(view["x_centers"][None, :], view["y_centers"][:, None], (0, 1, 0, 2), 20)
    payload = result.x_measurement.data
    for index in range(payload.shape[0]):
        bins = np.clip(np.searchsorted(payload.axes[0].values, u, side="right")-1, 0, payload.shape[0]-1)
        selected = result.selected & (bins == index)
        c, v, n = [view[name][selected] for name in (EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR, NORMALIZATION_DENOMINATOR)]
        reference = estimate_measurement_bin(result.x_measurement.contract, c, v, exposure=n)
        assert payload.signal[index] == pytest.approx(reference.value, nan_ok=True)
        assert payload.errors[index] == pytest.approx(reference.standard_error, nan_ok=True)
    zero = rotated_box_profiles(view, view["signal"], view["errors"], (0, 1, 0, 2), 0)
    np.testing.assert_allclose(zero.x[1:], profiles(view).x[1:])


def test_continuous_precision_and_extreme_unit_scale_match_reference():
    view = prepare(count_data())
    view.pop(EVENT_STATISTICS_KEY)
    view["signal"], view["errors"] = np.array([[2.], [10.]]), np.array([[1.], [3.]])
    result = profiles(view).x_measurement
    reference = estimate_measurement_bin(result.contract, [2, 10], [1, 9])
    assert result.data.signal.item() == pytest.approx(reference.value)
    assert result.data.errors.item()**2 == pytest.approx(reference.variance)
    view["errors"] *= 1e-155
    small = profiles(view).x_measurement.data
    assert small.signal.item() == pytest.approx(2.8)
    assert small.errors.item() == pytest.approx(np.sqrt(.9)*1e-155, rel=1e-12, abs=0)


def test_channels_smoothing_and_stale_statistics_do_not_claim_count_pooling():
    view = prepare(count_data())
    assert profiles(view, channel="errors").x_measurement.contract.kind == "continuous"
    smoothed = smooth_mdhisto_view(view, sigma_y=1)
    assert profiles(smoothed).x_measurement.contract.kind == "continuous"
    stale = dict(view, signal=view["signal"] * 2)
    assert profiles(stale).x_measurement.contract.kind == "continuous"
    declared = profiles(view).x_measurement.data
    # Smoothing a retained cut also invalidates its unsmoothed declaration.
    live = _profile_dataset(CutViewerContext(count_data(), 1, 0, str, "Sample"),
        "x", profiles(view).x, profiles(view).x_measurement)
    smoothed = smooth_mdhisto_view(prepare(live), sigma_x=1)
    assert "measurement_contract" not in smoothed
    assert declared.metadata["measurement_contract"]["kind"] == "counting"


def test_declared_contract_preserved_or_rejected_explicitly():
    view = prepare(count_data())
    declared = MeasurementContract(kind="counting", estimator="exposure_pool", quantity="cross section",
        value_units="mb", exposure_units="calibrated exposure", missing="reject")
    view["measurement_contract"] = declared.to_dict()
    assert profiles(view).x_measurement.contract == declared
    view["mask"] = np.array([[False], [True]])
    with pytest.raises(ValueError, match="rejects missing"):
        profiles(view)
    for field, value in (("dependence", "shared_sources"), ("normalizer", "uncertain")):
        view["measurement_contract"] = declared.to_dict() | {field: value}
        with pytest.raises(ValueError, match="source replay"):
            profiles(view)


def test_model_predictions_use_data_exposure_with_explicit_prediction_role():
    view = prepare(count_data())
    result = histogram_box_profiles(view, np.array([[2.], [10.]]), view["errors"], (0, 1, 0, 2),
        reference_values=view["signal"]).x_measurement
    assert result.data.signal.item() == 9.2
    assert result.data.errors.item() == .2
    assert result.data.metadata["profile_role"] == "model_prediction"
    assert EVENT_STATISTICS_KEY not in result.data.metadata
    assert result.data.num_events.item() == 0


def test_profile_archive_live_view_and_csv_retain_exact_statistics(tmp_path):
    data = count_data()
    result = profiles(prepare(data)).x_measurement
    context = CutViewerContext(data, 1, 0, lambda dim: str(dim), "Sample")
    live = _profile_dataset(context, "x", result.arrays, result)
    np.testing.assert_allclose(selected_event_statistics(live), np.array([[4], [4], [10]])[:, None, :])
    live_view = MDHistoSliceViewer(live, x_dim=1, y_dim=0).slice_arrays()
    assert profiles(live_view).x[1].item() == .4
    path = save_measurement_profile_csv(tmp_path / "cut.csv", result, coordinate_name="x", coordinate_unit="r.l.u.")
    exported = np.genfromtxt(path, names=True, delimiter=",")
    assert exported["intensity"] == .4 and exported["uncertainty"] == .2
    assert exported["event_signal_numerator"] == 4 and exported["normalization_denominator"] == 10
    declaration = json.loads(path.with_suffix(".csv.json").read_text())
    assert MeasurementContract.from_dict(declaration["measurement_contract"]) == result.contract
    assert declaration["coordinate"]["unit"] == "r.l.u."


@pytest.mark.parametrize("angle", [0., 20.])
def test_qt_and_static_count_cuts_export_the_prepared_result(angle, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from nfit.plotting_core import plot_mdhisto_slice
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    original = count_data()
    channels = {name: type(channel)(np.repeat(channel.values, 2, axis=1), label=channel.label, unit=channel.unit)
                for name, channel in original.auxiliary_channels.items()}
    data = original.with_updates(
        axes=(original.axes[0], MDHistoAxis("x", np.arange(3), "r.l.u.", "momentum")),
        signal=np.repeat(original.signal, 2, axis=1), errors=np.repeat(original.errors, 2, axis=1),
        mask=np.repeat(original.mask, 2, axis=1), num_events=np.repeat(original.num_events, 2, axis=1),
        auxiliary_channels=channels)
    viewer = QtMDHistoSliceViewer(data, x_dim=1, y_dim=0)
    assert "exposure" in viewer.hist_axes_check.toolTip()
    assert "JSON" in viewer.save_x_cut_button.toolTip()
    viewer.hist_axes_check.setChecked(True)
    viewer.show_box_check.setChecked(True)
    viewer._roi_angle = angle
    viewer._set_roi_extents((0, 2, 0, 2), update_cuts=True, draw=False)
    expected = profiles(viewer._current_slice, angle=angle)
    np.testing.assert_allclose(viewer._current_x_cut[1:], expected.x[1:], equal_nan=True)
    figure = plot_mdhisto_slice(data, x_dim=1, y_dim=0, show_histogram_axes=True,
        roi_extents=(0, 2, 0, 2), roi_angle=angle)
    np.testing.assert_allclose(figure.axes[3].lines[0].get_ydata(), expected.x[1], equal_nan=True)
    path = tmp_path / "qt.csv"
    monkeypatch.setattr(viewer, "_csv_export_path", lambda *_args: str(path))
    viewer.save_x_cut()
    exported = np.atleast_1d(np.genfromtxt(path, names=True, delimiter=","))
    np.testing.assert_allclose(exported["intensity"], expected.x[1], equal_nan=True)
    assert path.with_suffix(".csv.json").exists()
    viewer.window.close()
    plt.close(figure)


def test_profile_dataset_archive_round_trip(tmp_path):
    from nfit.pipeline import DatasetEntry
    from nfit.project_dataset_io import _load_nfit_dataset_file, save_dataset_file
    original = profiles(prepare(count_data())).x_measurement.data
    path = tmp_path / "profile.npz"
    save_dataset_file(DatasetEntry("Cut", original), path, use_view=False)
    restored, _ = _load_nfit_dataset_file(path)
    assert all(restored.metadata[key] == value for key, value in original.metadata.items())
    for source, loaded in zip(selected_event_statistics(original), selected_event_statistics(restored), strict=True):
        np.testing.assert_array_equal(source, loaded)
