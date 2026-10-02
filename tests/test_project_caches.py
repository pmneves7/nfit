from __future__ import annotations

import numpy as np
import pytest

import nfit.project_caches as project_caches
import nfit.project_data as project_data
from nfit import clear_project_caches, electronic_structure
from nfit.analysis import fingerprint
from nfit.dataset import PointListData
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DataGroup, DatasetEntry, ModelComponentSpec, PlotEntry
from nfit.project_gui import NfitProject
from nfit.raw_dgs_cache import cache_event_chunks
from nfit.rebin_cache import CompressedBinning, _DiskBinning


def test_project_gui_compatibility_names_share_cache_service_owners():
    import nfit.project_gui as project_gui

    assert project_gui.clear_project_caches is clear_project_caches
    assert project_gui._MODEL_OVERLAY_CACHE is project_caches.MODEL_OVERLAY_CACHE
    assert project_gui._MODEL_OVERLAY_ERRORS is project_caches.MODEL_OVERLAY_ERRORS
    assert project_gui._VIEWER_VIEW_CACHE is project_data._VIEWER_VIEW_CACHE
    assert project_gui._COMPOSITE_DATA_CACHE is project_data._COMPOSITE_DATA_CACHE


def _data(value: float = 1.0) -> MDHistoData:
    axis = MDHistoAxis("H", np.array([0.0, 1.0]), "rlu", "momentum")
    signal = np.array([value])
    return MDHistoData(
        axes=(axis,),
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={},
    )


def test_clear_project_caches_is_scoped_and_evicts_rebin_tiers(tmp_path):
    viewer_cache = project_data._VIEWER_VIEW_CACHE
    composite_cache = project_data._COMPOSITE_DATA_CACHE
    viewer_cache.clear()
    composite_cache.clear()

    owned_data = _data(1.0)
    owned_dataset = DatasetEntry("owned", owned_data, kind="mdhisto")
    outside_dataset = DatasetEntry("outside", _data(2.0), kind="mdhisto")
    owned_group = DataGroup("Owned", datasets=[owned_dataset])
    outside_group = DataGroup("Outside", datasets=[outside_dataset])
    project = NfitProject([owned_group])
    owned_id = owned_dataset.id
    outside_id = outside_dataset.id
    resident_key = owned_id
    compressed_key = f"{owned_id}:overview"
    disk_key = f"{owned_id}:saved"
    viewer_cache[resident_key] = ("resident", owned_data)
    viewer_cache[compressed_key] = ("compressed", owned_data)
    artifact = CompressedBinning.from_data(owned_data, max_bytes=1_000_000)
    assert artifact is not None
    viewer_cache._compressed[compressed_key] = ("compressed", artifact)
    viewer_cache._compressed_bytes += artifact.nbytes
    disk_path = tmp_path / "owned-cache.npz"
    disk_path.write_bytes(b"owned cache artifact")
    viewer_cache._disk[disk_key] = _DiskBinning("disk", disk_path)
    backing_path = tmp_path / "owned-project.nfit"
    backing_path.write_bytes(b"project backing")
    viewer_cache._project_backings[compressed_key] = _DiskBinning(
        "backing", backing_path, owned=False
    )
    viewer_cache[outside_id] = ("outside", outside_dataset.data)

    owned_composite_key = (id(owned_group), "overview")
    outside_composite_key = id(outside_group)
    composite_cache[owned_composite_key] = ("owned", owned_data)
    composite_cache[outside_composite_key] = ("outside", outside_dataset.data)

    try:
        clear_project_caches(project)

        for key in (resident_key, compressed_key, disk_key):
            assert key not in viewer_cache
            assert key not in viewer_cache._compressed
            assert key not in viewer_cache._disk
            assert key not in viewer_cache._project_backings
        assert not disk_path.exists()
        assert outside_id in viewer_cache
        assert outside_composite_key in composite_cache
        assert owned_composite_key not in composite_cache
        assert owned_data is owned_dataset.data
    finally:
        viewer_cache.clear()
        composite_cache.clear()


def test_clear_project_caches_drops_reduced_model_and_prepared_data_caches(
    tmp_path,
):
    data = _data()
    dataset = DatasetEntry("run", data, kind="mdhisto")
    disabled = project_data.dataset_rebin_config(dataset)
    disabled.update(enabled=False, stale=False)
    group = DataGroup("Workspace", datasets=[dataset])
    named_id = project_data.add_dataset_rebin_binning(dataset, name="Overview")
    named = project_data.dataset_rebin_config_by_id(dataset, named_id)
    named.update(enabled=False, stale=False)
    composite = project_data.data_group_composite_config(group)
    composite.update(enabled=False, stale=False)
    group.metadata[project_data.GROUP_COMPOSITE_KEY] = composite

    component = ModelComponentSpec("tight binding", type="tight_binding")
    group.models[component.name] = component
    model_payload = object()
    component._nfit_electronic_model_cache = ("digest", model_payload)
    group.plots.append(PlotEntry("Stored plot"))
    plot = group.plots[0]
    fit = object()
    group.fits.append(fit)
    project = NfitProject([group], settings={"cache_binnings": True, "binning_cache_entries": [1]})

    reduced_events = np.arange(12.0).reshape(2, 6)
    tuple(
        cache_event_chunks(
            dataset,
            "reduction-signature",
            {"source": "test"},
            {},
            [(reduced_events, 2)],
        )
    )
    reduced_path = dataset._raw_dgs_reduction_cache.content
    assert reduced_path.exists()
    point_list_cache = project_data._PREPARED_POINT_LIST_CACHE
    point_list_cache.clear()
    point_list_cache[dataset.id] = ("prepared-signature", PointListData(
        columns={"x": [1.0]}, units={}, coordinate_names=("x",), channels=()
    ))
    project_caches.MODEL_OVERLAY_CACHE[id(group)] = {"compiled": model_payload}
    project_caches.MODEL_OVERLAY_ERRORS[id(group)] = {"run": "error"}
    electronic_structure._FOURIER_COEFFICIENT_CACHE["project-test"] = np.ones(1)
    electronic_structure._HAMILTONIAN_COMPONENT_CACHE["project-test"] = np.ones(1)
    fingerprint._DATASET_FINGERPRINT_CACHE[dataset.id] = ("signature", "fingerprint")
    fingerprint._DATA_ARRAY_HASH_CACHE[dataset.id] = ("hash",)

    try:
        clear_project_caches(project)

        assert dataset._raw_dgs_reduction_cache is None
        assert "raw_dgs_reduction_cache" not in dataset.metadata
        assert not reduced_path.exists()
        assert dataset.id not in point_list_cache
        assert id(group) not in project_caches.MODEL_OVERLAY_CACHE
        assert id(group) not in project_caches.MODEL_OVERLAY_ERRORS
        assert "_nfit_electronic_model_cache" not in component.__dict__
        assert not electronic_structure._FOURIER_COEFFICIENT_CACHE
        assert not electronic_structure._HAMILTONIAN_COMPONENT_CACHE
        assert not fingerprint._DATASET_FINGERPRINT_CACHE
        assert not fingerprint._DATA_ARRAY_HASH_CACHE
        assert disabled["stale"] is True and named["stale"] is True
        assert composite["stale"] is True
        assert project.settings["cache_binnings"] is False
        assert "binning_cache_entries" not in project.settings
        assert dataset.data is data
        assert group.plots == [plot]
        assert group.fits == [fit]
    finally:
        point_list_cache.clear()
        project_caches.MODEL_OVERLAY_CACHE.clear()
        project_caches.MODEL_OVERLAY_ERRORS.clear()
        with electronic_structure._FOURIER_CACHE_LOCK:
            electronic_structure._FOURIER_COEFFICIENT_CACHE.clear()
        with electronic_structure._HAMILTONIAN_COMPONENT_CACHE_LOCK:
            electronic_structure._HAMILTONIAN_COMPONENT_CACHE.clear()
        fingerprint._DATASET_FINGERPRINT_CACHE.clear()
        fingerprint._DATA_ARRAY_HASH_CACHE.clear()


def test_clear_all_project_caches_requires_yes_and_closes_viewers(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    import nfit.project_gui as project_gui

    group = DataGroup("Workspace", datasets=[DatasetEntry("run", _data())])
    project = NfitProject([group], settings={"cache_binnings": True})
    explorer = project_gui.NfitProjectExplorer(project)
    explorer._sync_cache_binnings_action()
    project_gui._VIEWER_VIEW_CACHE.clear()
    project_gui._VIEWER_VIEW_CACHE[group.datasets[0].id] = ("sig", group.datasets[0].data)

    def question_spy(*args):
        calls.append(args)
        return QtWidgets.QMessageBox.StandardButton.No

    calls = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", question_spy)
    monkeypatch.setattr(
        explorer,
        "_close_all_slice_viewers",
        lambda: pytest.fail("declining must leave viewers open"),
    )
    monkeypatch.setattr(
        project_gui,
        "clear_project_caches",
        lambda _project: pytest.fail("declining must leave caches untouched"),
    )

    assert explorer.clear_all_project_caches() is False
    assert len(calls) == 1
    assert calls[0][1] == "Clear all caches in project"
    assert "may take a while" in calls[0][2]
    assert calls[0][3] == (
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.No
    )
    assert calls[0][4] == QtWidgets.QMessageBox.StandardButton.No
    assert project_gui._VIEWER_VIEW_CACHE
    assert explorer.cache_binnings_action.isChecked()
    monkeypatch.setattr(explorer, "_handle_window_close", lambda event: event.accept())
    explorer.window.close()
    project_gui._VIEWER_VIEW_CACHE.clear()


def test_clear_all_project_caches_acceptance_updates_project_ui(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    import nfit.project_gui as project_gui

    dataset = DatasetEntry("run", _data())
    project = NfitProject(
        [DataGroup("Workspace", datasets=[dataset])],
        settings={"cache_binnings": True, "binning_cache_entries": ["old"]},
    )
    explorer = project_gui.NfitProjectExplorer(project)
    explorer._sync_cache_binnings_action()
    project_gui._VIEWER_VIEW_CACHE.clear()
    project_gui._VIEWER_VIEW_CACHE[dataset.id] = ("sig", dataset.data)
    events = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(explorer, "_close_all_slice_viewers", lambda: events.append("close"))
    original_clear = project_gui.clear_project_caches

    def clear_and_record(current_project):
        events.append("clear")
        original_clear(current_project)

    monkeypatch.setattr(project_gui, "clear_project_caches", clear_and_record)
    monkeypatch.setattr(explorer, "_refresh_cache_badges", lambda: events.append("badges"))
    monkeypatch.setattr(explorer, "_sync_details", lambda: events.append("details"))

    assert explorer.clear_all_project_caches() is True
    assert events == ["close", "clear", "badges", "details"]
    assert not project_gui._VIEWER_VIEW_CACHE
    assert project.settings["cache_binnings"] is False
    assert "binning_cache_entries" not in project.settings
    assert not explorer.cache_binnings_action.isChecked()
    project_gui._VIEWER_VIEW_CACHE.clear()
    explorer._allow_window_close = True
    explorer.window.close()
