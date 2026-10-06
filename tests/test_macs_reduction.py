"""MACS acquisition edits and calibrated count pooling across GUI and scripts."""

import copy

import h5py
import numpy as np
import pytest

from nfit import DataGroup, NfitProject, NfitProjectExplorer
from nfit.cache_utils import dataset_content_signature
from nfit.histogram_statistics import selected_event_statistics
from nfit.macs import import_macs_nexus
from nfit.macs_reduction import MACS_DEFAULTS
from nfit.measurement_rebinning import coarsen_measurement_histogram
from nfit.project_imports import _ensure_dataset_data_loaded, import_dataset_paths
from nfit.project_rebinning import _rebin_point_data
from nfit.reduction_recipes import (
    effective_reduction_config,
    reduction_family,
    reduction_recipe_script,
    reduction_settings_schema,
    set_reduction_settings,
)
from nfit.source_selection import SourceSelection
from nfit.source_selection_imports import import_source_selection, update_source_selection
from tests.test_macs import _write_macs_nexus


def _single_cell_config():
    return {"mean_weighting": "inverse_variance", "fractional": False,
            "axes": [{"lower": -20., "upper": 20., "num_bins": 1,
                      "mode": "bins", "vector": np.eye(4)[dim].tolist()}
                     for dim in range(4)]}


def test_default_response_equal_sensitivity_and_zero_variance(tmp_path):
    source = _write_macs_nexus(tmp_path / "scan.ng0")
    with h5py.File(source, "r+") as handle:
        handle["entry/DAS_logs/specDetector/detectorEfficiency"][0] = 17.
    data = import_macs_nexus(source)
    factor = np.sqrt(2.0721246 / 3.7)
    assert data.metadata["detector_efficiency_applied"] is False
    assert data.intensity[0] == pytest.approx(45. * factor)
    raw = data.measurement_payload["num_events"]
    np.testing.assert_allclose(data.intensity * data.normalization_denominator, raw)
    np.testing.assert_allclose((data.sigma * data.normalization_denominator)**2, raw)
    assert np.all(data.sigma[raw == 0] == 0)
    assert data.metadata["measurement_contract"]["estimator"] == "exposure_pool"


def test_monitor_response_and_kinematics_are_independent_choices(tmp_path):
    source = _write_macs_nexus(tmp_path / "scan.ng0")
    corrected = import_macs_nexus(source)
    uncorrected = import_macs_nexus(source, {"monitor_response": "constant", "ki_kf_normalization": False})
    kinematic_only = import_macs_nexus(source, {"monitor_response": "constant"})
    np.testing.assert_allclose(corrected.intensity, uncorrected.intensity * np.sqrt(2.0721246/3.7))
    np.testing.assert_allclose(kinematic_only.intensity, uncorrected.intensity * np.sqrt(4.9/3.7))


def test_counts_exposure_and_errors_pool_including_zero_observations(tmp_path):
    source = _write_macs_nexus(tmp_path / "scan.ng0")
    with h5py.File(source, "r+") as handle:
        handle["entry/DAS_logs/specDetector/counts"][0, :10] = 0.
        handle["entry/DAS_logs/counter/liveMonitor"][::2] = 1.e5
    data = import_macs_nexus(source)
    result = _rebin_point_data(data, _single_cell_config())
    count = data.measurement_payload["num_events"][data.mask].sum()
    exposure = data.normalization_denominator[data.mask].sum()
    assert result.signal.item() == pytest.approx(count/exposure)
    assert result.errors.item() == pytest.approx(np.sqrt(count)/exposure)
    assert result.num_events.item() == pytest.approx(count)
    c, v, n = selected_event_statistics(result)
    assert c.item() == pytest.approx(count)
    assert v.item() == pytest.approx(count)
    assert n.item() == pytest.approx(exposure)
    assert result.metadata["rebin"]["mean_weighting"] == "uniform"


def test_fractional_count_variance_and_final_coarsening(tmp_path):
    source = _write_macs_nexus(tmp_path / "scan.ng0")
    data = import_macs_nexus(source).subset(np.arange(480) == 0)
    data = data.with_updates(H=np.array([.25]), K=np.array([0.]), L=np.array([0.]), E=np.array([0.]))
    config = _single_cell_config()
    config["axes"][0].update(lower=0., upper=1., mode="step", step_size=.5, fractional=True)
    config["fractional_axes"] = [True, False, False, False]
    result = _rebin_point_data(data, config)
    c, v, n = selected_event_statistics(result)
    np.testing.assert_allclose(c.ravel()[:2], [4.5, 4.5])
    np.testing.assert_allclose(v.ravel()[:2], [2.25, 2.25])
    np.testing.assert_allclose(n.ravel()[:2], data.normalization_denominator[0] * .5)
    assert "poisson_count_model" not in result.metadata
    # Whole-cell profiles retain the pooled estimator and diagonal variance.
    merged = coarsen_measurement_histogram(result, [axis.values[[0, -1]] for axis in result.axes])
    assert merged.signal.item() == pytest.approx(data.intensity[0])
    assert merged.errors.item() == pytest.approx(data.sigma[0]/np.sqrt(2))


def test_higher_order_model_respects_filter_logs_and_unknown_states(tmp_path):
    source = _write_macs_nexus(tmp_path / "scan.ng0")
    with pytest.raises(ValueError, match="filter state is unknown"):
        import_macs_nexus(source, {"higher_order_monitor_correction": "dave"})
    for name in ("CFXBE", "CFXHOPG"):
        with h5py.File(source, "r+") as handle:
            handle.create_dataset(f"entry/DAS_logs/{name}/primaryNode", data=np.bytes_("IN"))
    normal = import_macs_nexus(source)
    logged = import_macs_nexus(source, {"higher_order_monitor_correction": "dave"})
    forced = import_macs_nexus(source, {"higher_order_monitor_correction": "dave", "incident_filter": "out"})
    np.testing.assert_array_equal(normal.intensity, logged.intensity)
    assert np.all(forced.intensity[normal.intensity > 0] > normal.intensity[normal.intensity > 0])


def test_source_edit_and_reduction_recipe_replay_are_complete(tmp_path):
    for run in (1, 2):
        _write_macs_nexus(tmp_path / f"Ef3p7_et_1_{run}.nxs.ng0")
    root = DataGroup("MACS")
    selection = SourceSelection(tmp_path, "Ef*_et*_", ".nxs.ng0", "1")
    collection = import_source_selection(root, selection)
    group = collection.subgroups[0]
    dataset = group.datasets[0]
    old_id = dataset.id
    old_signature = dataset_content_signature(dataset)
    assert reduction_family(group) == "macs-step-nexus"
    assert {field.key for field in reduction_settings_schema(group)} == set(MACS_DEFAULTS)
    set_reduction_settings(group, {"monitor_target": 2.e6})
    assert dataset.data is None
    assert dataset_content_signature(dataset) != old_signature
    assert _ensure_dataset_data_loaded(dataset).metadata["monitor_target"] == 2.e6
    update_source_selection(group, SourceSelection(tmp_path, "Ef*_et*_", ".nxs.ng0", "1:2"), parent=root)
    assert len(group.datasets) == 2 and not group.subgroups
    assert group.metadata["reduction_recipe"]["source_selection"]["expression"] == "1:2"
    assert group.datasets[0].id == old_id
    assert group.datasets[1].metadata["import_options"]["monitor_target"] == 2.e6
    set_reduction_settings(group, {"a3_offset_deg": 66.5}, dataset_ids=[group.datasets[1].id])
    namespace = {}
    exec(reduction_recipe_script(group), namespace)
    replayed = namespace["group"]
    assert [entry.id for entry in replayed.datasets] == [entry.id for entry in group.datasets]
    for original, replay in zip(group.datasets, replayed.datasets, strict=True):
        for field in reduction_settings_schema(group):
            assert effective_reduction_config(group, original).get(field.key, field.default) == effective_reduction_config(replayed, replay).get(field.key, field.default)
        np.testing.assert_allclose(_ensure_dataset_data_loaded(original).H, replay.data.H)


def test_legacy_settings_migrate_without_losing_per_file_choices(tmp_path):
    paths = [_write_macs_nexus(tmp_path / f"scan_{run}.ng0") for run in (1, 2)]
    root = DataGroup("MACS")
    import_dataset_paths(root, paths)
    group = root.subgroups[0]
    group.metadata.pop("macs")
    group.datasets[1].metadata["import_options"]["a3_offset_deg"] = 71.
    set_reduction_settings(group, {"monitor_target": 3.e6})
    assert effective_reduction_config(group, group.datasets[1])["a3_offset_deg"] == 71.
    before = copy.deepcopy(group.metadata)
    with pytest.raises(ValueError):
        set_reduction_settings(group, {"monitor_target": 0})
    assert group.metadata == before


def test_composite_pool_and_reload_preserve_user_channel_configuration(tmp_path):
    from nfit.project_composites import _composite_point_data

    first = _write_macs_nexus(tmp_path / "run1.ng0")
    second = _write_macs_nexus(tmp_path / "run2.ng0")
    with h5py.File(second, "r+") as handle:
        handle["entry/DAS_logs/counter/liveMonitor"][:] *= 3.
    root = DataGroup("MACS")
    import_dataset_paths(root, [first, second])
    group = root.subgroups[0]
    data = _composite_point_data(root, _single_cell_config(), datasets=group.datasets)
    sources = [entry.data.valid(require_positive_sigma=False) for entry in group.datasets]
    count = sum(source.measurement_payload["num_events"].sum() for source in sources)
    exposure = sum(source.normalization_denominator.sum() for source in sources)
    assert data.signal.item() == pytest.approx(count/exposure)
    assert data.errors.item() == pytest.approx(np.sqrt(count)/exposure)
    assert data.num_events.item() == pytest.approx(count)
    entry = group.datasets[0]
    entry.parameters["spectral_channels"]["normalization_basis"] = "per_formula_unit"
    set_reduction_settings(group, {"ki_kf_normalization": False})
    _ensure_dataset_data_loaded(entry)
    assert entry.parameters["spectral_channels"]["normalization_basis"] == "per_formula_unit"
    assert entry.parameters["spectral_channels"]["kf_ki_state"] == "included"
    entry.scale_factor = -1.
    with pytest.raises(ValueError, match="background link or derived dataset"):
        _composite_point_data(root, _single_cell_config(), datasets=group.datasets)


def test_macs_reduction_gui_exposes_schema_without_loading(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    source = _write_macs_nexus(tmp_path / "scan.ng0")
    root = DataGroup("MACS")
    import_dataset_paths(root, [source])
    group = root.subgroups[0]
    group.datasets[0].unload_data()
    explorer = NfitProjectExplorer(NfitProject([root]))
    group.datasets[0].unload_data()
    group.metadata["composite"]["auto_rebin"] = False
    explorer._refresh_tree(select_dataset_group=group)
    for field in reduction_settings_schema(group):
        editor = explorer.details_widget.findChild(QtWidgets.QWidget, f"macs_{field.key}")
        assert editor is not None and editor.toolTip()
    from nfit.reduction_recipe_gui import _field_editor
    from nfit.reduction_recipes import validated_reduction_settings

    field = next(field for field in reduction_settings_schema(group) if field.key == "masked_analyzer_channels")
    edits = []
    editor = _field_editor(field, [19], object_name="clear_masks", on_changed=edits.append)
    editor.setText("")
    editor.editingFinished.emit()
    assert edits == [[]]
    assert validated_reduction_settings(group, {field.key: edits[-1]}) == {field.key: []}
    target = explorer.details_widget.findChild(QtWidgets.QComboBox, "group_composite_mean_weighting")
    assert not target.isEnabled() and target.currentData() == "uniform"
    assert group.datasets[0].data is None
    explorer.has_unsaved_changes = False
    explorer.window.close()
