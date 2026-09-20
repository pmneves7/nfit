from __future__ import annotations

import pytest

from nfit.viewer_data import (
    DeferredViewerDatasets,
    ViewerDatasetDescriptor,
    ViewerLoadCancelled,
)
from tests.plotting_test_data import tiny_1d_mdhisto_data, tiny_mdhisto_data


def _catalog(loader, *, initial_index=0, revision=None):
    return DeferredViewerDatasets(
        [
            ViewerDatasetDescriptor(
                "fine",
                source_dataset_name="scan",
                binning_name="Fine",
                binning_id="fine-id",
                group_key="group",
                crystal_context={"spacegroup": "P 1"},
            ),
            ViewerDatasetDescriptor(
                "coarse",
                source_dataset_name="scan",
                binning_name="Coarse",
                group_key="group",
            ),
        ],
        loader,
        initial_index=initial_index,
        revision=revision,
    )


def test_deferred_viewer_datasets_catalog_and_identity_do_not_load_payloads():
    calls = []
    datasets = _catalog(lambda index: calls.append(index) or tiny_mdhisto_data())

    assert len(datasets) == 2
    assert datasets.descriptors[1].binning_name == "Coarse"
    assert datasets.descriptors[0].binning_id == "fine-id"
    assert datasets.cached_indices == ()
    assert datasets.cache_identity()[1] == ()
    assert calls == []

    assert datasets[1] is datasets[1]
    assert calls == [1]
    assert datasets.cached_indices == (1,)


@pytest.mark.parametrize("deferred", [False, True])
def test_viewer_rejects_mismatched_explicit_crystal_contexts(monkeypatch, deferred):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _catalog(lambda _index: tiny_mdhisto_data()) if deferred else tiny_mdhisto_data()
    with pytest.raises(ValueError, match="crystal_contexts length"):
        QtMDHistoSliceViewer(data, crystal_contexts=[])


def test_qt_viewer_uses_deferred_initial_index_and_catalog_without_eager_loading(
    monkeypatch,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    calls = []
    payloads = [tiny_mdhisto_data(), tiny_mdhisto_data()]
    datasets = _catalog(lambda index: calls.append(index) or payloads[index], initial_index=1)

    viewer = QtMDHistoSliceViewer(datasets, x_dim=3, y_dim=2)

    assert calls == [1]
    assert viewer.datasets is datasets
    assert viewer.dataset_index == 1
    assert viewer.dataset_combo.currentText() == "scan"
    assert viewer.binning_combo.currentText() == "Coarse"

    viewer.binning_combo.setCurrentIndex(0)
    assert calls == [1, 0]
    assert viewer.dataset_index == 0


def test_qt_viewer_deferred_replacement_only_reconnects_cached_payloads(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    old_calls = []
    new_calls = []
    old = _catalog(lambda index: old_calls.append(index) or tiny_mdhisto_data())
    new = _catalog(lambda index: new_calls.append(index) or tiny_mdhisto_data(), revision="new")
    viewer = QtMDHistoSliceViewer(old, x_dim=3, y_dim=2)

    viewer.replace_datasets(new, selected_dataset_name="scan", selected_binning_name="Fine")

    assert old_calls == [0]
    assert new_calls == [0]
    assert viewer.datasets is new
    assert viewer._dataset_states[1] is None


def test_qt_viewer_failed_deferred_switch_preserves_selection(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    def load(index):
        if index == 1:
            raise RuntimeError("cancelled")
        return tiny_mdhisto_data()

    viewer = QtMDHistoSliceViewer(_catalog(load), x_dim=3, y_dim=2)
    viewer.binning_combo.setCurrentIndex(1)

    assert viewer.dataset_index == 0
    assert viewer.binning_combo.currentText() == "Fine"
    assert "cancelled" in viewer._last_dataset_load_error


def test_qt_viewer_cancelled_deferred_switch_is_silent(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    def load(index):
        if index == 1:
            raise ViewerLoadCancelled
        return tiny_mdhisto_data()

    viewer = QtMDHistoSliceViewer(_catalog(load), x_dim=3, y_dim=2)
    viewer.binning_combo.setCurrentIndex(1)

    assert viewer.dataset_index == 0
    assert viewer.binning_combo.currentText() == "Fine"
    assert not hasattr(viewer, "_last_dataset_load_error")


def test_qt_viewer_refresh_retains_unloaded_view_state_and_rebinds_new_payload(
    monkeypatch,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    old_calls = []
    new_calls = []
    old_payloads = [tiny_mdhisto_data(), tiny_mdhisto_data()]
    new_payloads = [
        tiny_mdhisto_data().with_updates(signal=tiny_mdhisto_data().signal + 10.0),
        tiny_mdhisto_data().with_updates(signal=tiny_mdhisto_data().signal + 20.0),
    ]
    old = _catalog(lambda index: old_calls.append(index) or old_payloads[index])
    new = _catalog(lambda index: new_calls.append(index) or new_payloads[index])
    viewer = QtMDHistoSliceViewer(old, x_dim=3, y_dim=2)
    viewer.binning_combo.setCurrentIndex(1)
    viewer.channel_combo.setCurrentText("errors")
    viewer.x_combo.setCurrentIndex(1)
    expected_x = viewer.x_combo.currentText()
    viewer.binning_combo.setCurrentIndex(0)

    viewer.replace_datasets(
        new,
        selected_dataset_name="scan",
        selected_binning_name="Fine",
    )

    assert new_calls == [0]
    assert viewer._dataset_states[1] is not None
    assert viewer._dataset_states[1].model.data is old_payloads[1]

    viewer.binning_combo.setCurrentIndex(1)

    assert new_calls == [0, 1]
    assert viewer.data is new_payloads[1]
    assert viewer.model.data is new_payloads[1]
    assert viewer.model.channel == "errors"
    assert viewer.x_combo.currentText() == expected_x


@pytest.mark.parametrize(
    ("failure", "reported"),
    [(ViewerLoadCancelled(), False), (RuntimeError("broken"), True)],
)
def test_qt_viewer_multi_dataset_mode_restores_previous_mode_on_load_failure(
    monkeypatch, failure, reported
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    descriptors = [
        ViewerDatasetDescriptor("first", binning_name="Default", group_key="cuts"),
        ViewerDatasetDescriptor("second", binning_name="Default", group_key="cuts"),
    ]

    def load(index):
        if index == 1:
            raise failure
        return tiny_1d_mdhisto_data()

    viewer = QtMDHistoSliceViewer(
        DeferredViewerDatasets(descriptors, load), x_dim=3, y_dim=2
    )
    viewer.view_mode_combo.setCurrentIndex(1)

    assert viewer.view_mode_combo.currentIndex() == 0
    assert viewer._active_plot_view_mode == 0
    assert hasattr(viewer, "_last_dataset_load_error") is reported


@pytest.mark.parametrize("manual", [False, True])
def test_deferred_refresh_reconciles_metadata_tile_range_on_revisit(
    monkeypatch, manual
):
    from dataclasses import replace

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer
    from tests.test_metadata_dimensions import temperature_volume

    expanded = temperature_volume()
    old_axis = replace(
        expanded.axes[-1],
        values=expanded.axes[-1].values[:-1],
        metadata={
            **expanded.axes[-1].metadata,
            "discrete_centers": expanded.axes[-1].centers[:-1].tolist(),
        },
    )
    old = expanded.with_updates(
        axes=(*expanded.axes[:-1], old_axis),
        signal=expanded.signal[..., :-1],
        errors=expanded.errors[..., :-1],
        mask=expanded.mask[..., :-1],
        num_events=expanded.num_events[..., :-1],
    )
    descriptors = [
        ViewerDatasetDescriptor("other", source_dataset_name="other"),
        ViewerDatasetDescriptor("temperature", source_dataset_name="temperature"),
    ]
    old_sequence = DeferredViewerDatasets(
        descriptors,
        lambda index: tiny_mdhisto_data() if index == 0 else old,
        initial_index=1,
    )
    new_calls = []
    new_sequence = DeferredViewerDatasets(
        descriptors,
        lambda index: new_calls.append(index)
        or (tiny_mdhisto_data() if index == 0 else expanded),
    )
    viewer = QtMDHistoSliceViewer(old_sequence, x_dim=0, y_dim=1)
    viewer.tile_dim = 4
    viewer.tile_range = (5.24, 20.12) if manual else (1.54, 30.06)
    viewer.dataset_combo.setCurrentIndex(0)

    viewer.replace_datasets(new_sequence, selected_dataset_name="other")
    assert new_calls == [0]
    viewer.dataset_combo.setCurrentIndex(1)

    assert new_calls == [0, 1]
    assert viewer.tile_range == (
        (5.24, 20.12) if manual else (1.54, 39.28)
    )
