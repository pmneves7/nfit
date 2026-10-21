"""MACS measured backgrounds share GUI, saved recipes and scripting dispatch."""

import numpy as np
import pytest

from nfit import (
    BackgroundSpec,
    DataGroup,
    DatasetGroup,
    NfitProject,
    composite_dataset_data,
    dataset_entry_from_path,
    load_project,
    save_project,
)
from nfit.macs_reduction import adopt_macs_reduction
from nfit.project_composites import (
    _composite_background_signature,
    _composite_scope,
    data_group_composite_config,
)
from nfit.workflow import composite_workflow_script
from tests.test_macs import _write_macs_nexus


@pytest.fixture
def macs_groups(tmp_path):
    groups = []
    for name in ("sample", "dummy"):
        path = _write_macs_nexus(tmp_path / f"{name}.nxs", points=3, energy_transfer=-0.2)
        entry = dataset_entry_from_path(path, importer_name="macs_nexus",
            data_type="single_crystal_inelastic", importer_options={
                "stream": "spec", "incident_energy_policy": "recorded",
                "final_energy_policy": "recorded", "monitor_response": "constant",
                "ki_kf_normalization": False,
            })
        group = DatasetGroup(name, datasets=[entry])
        adopt_macs_reduction(group)
        groups.append(group)
    sample, dummy = groups
    link = BackgroundSpec("Dummy", source_group_id=dummy.id, source_group=dummy,
                          projection="measured_events")
    sample.backgrounds.append(link)
    root = DataGroup("MACS", subgroups=groups)
    scope = _composite_scope(root, sample)
    config = data_group_composite_config(scope)
    config.update(enabled=True, auto_rebin=False, minimum_coverage=0.,
                  resolution_mode="bins", fractional=False)
    for axis in config["axes"]:
        axis.update(lower=-10., upper=10., auto_lower=False, auto_upper=False,
                    num_bins=1, mode="bins", auto_step_size=False, fractional=False)
    return root, sample, dummy, link, config


def test_macs_replay_gui_option_and_tooltip(macs_groups, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import NfitProjectExplorer

    root, sample, _, link, _ = macs_groups
    explorer = NfitProjectExplorer(NfitProject([root]))
    monkeypatch.setattr(explorer, "_background_changed", lambda *_: None)
    explorer._set_background_details(root, sample, link)
    projection = explorer.details_widget.findChild(widgets.QComboBox, "background_projection")
    assert projection.findData("measured_events") >= 0
    assert projection.findData("sample_trajectories") < 0
    assert "monitor-normalized" in projection.toolTip()
    assert "MACS" in projection.toolTip()
    assert "Fractional off" in projection.toolTip()
    explorer.window.close()


def test_macs_replay_archive_and_exported_workflow_equivalence(macs_groups, tmp_path):
    root, sample, _, _, config = macs_groups
    project = NfitProject([root])
    expected = composite_dataset_data(root, node=sample, config_override=config)
    assert np.any(~expected.mask)
    sample_only = composite_dataset_data(root, node=sample, config_override=config,
                                         apply_backgrounds=False)
    # Identical synthetic runs are independent measurements, while replayed
    # angle copies of the dummy must not create additional counting precision.
    np.testing.assert_allclose(expected.signal[~expected.mask], 0., atol=1e-12)
    np.testing.assert_allclose(expected.errors[~expected.mask],
                               np.sqrt(2.) * sample_only.errors[~expected.mask])
    assert expected.source_dependencies is not None
    assert expected.metadata["background_subtractions"][-1]["projection"]["mode"] == "measured_events"
    script = composite_workflow_script(project, root.name, node_id=sample.id)
    namespace = {"__name__": "standalone"}
    exec(compile(script, "macs_replay.py", "exec"), namespace)
    replayed = namespace["run"]()
    path = tmp_path / "replay.nfit"
    save_project(project, path)
    restored_root = load_project(path).data_groups[0]
    restored = composite_dataset_data(restored_root, node=restored_root.subgroups[0], config_override=config)
    for actual in (replayed, restored):
        np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
        np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
        np.testing.assert_array_equal(actual.mask, expected.mask)
        assert actual.source_dependencies is not None


def test_macs_replay_cache_includes_effective_source_settings(macs_groups):
    root, sample, dummy, link, config = macs_groups
    scope = _composite_scope(root, sample)
    baseline = _composite_background_signature(scope, link, config)
    dummy.datasets[0].metadata["reduction_overrides"] = {"monitor_target": 500_000.}
    changed = _composite_background_signature(scope, link, config)
    assert changed != baseline
    dummy.datasets[0].metadata.pop("reduction_overrides")
    sample.metadata["macs"]["a3_offset_deg"] = 2.
    assert _composite_background_signature(scope, link, config) != baseline


@pytest.mark.parametrize("projection", ["measured_events", "macs_isotropic"])
def test_macs_projection_keeps_cache_version_out_of_projector_arguments(macs_groups, monkeypatch, projection):
    from nfit import macs_background
    from nfit.project_composites import _background_projection_settings, _composite_cache_signature

    root, sample, _dummy, link, config = macs_groups
    link.projection = projection
    link.metadata.update(energy_tolerance_meV=0.05, q_bin_width=0.3)
    arguments = _background_projection_settings(link)
    assert arguments == ({"energy_tolerance_meV": 0.05, "q_bin_width": 0.3}
                         if projection == "macs_isotropic" else {"energy_tolerance_meV": 0.05})
    scope = _composite_scope(root, sample)
    background_before = _composite_background_signature(scope, link, config)
    result_before = _composite_cache_signature(scope, config_override=config)
    base_before = _composite_cache_signature(scope, config_override=config, include_backgrounds=False)
    monkeypatch.setattr(macs_background, "MACS_BACKGROUND_NUMERICAL_VERSION",
                        macs_background.MACS_BACKGROUND_NUMERICAL_VERSION + 1)
    assert _background_projection_settings(link) == arguments
    assert _composite_background_signature(scope, link, config) != background_before
    assert _composite_cache_signature(scope, config_override=config, include_backgrounds=False) == base_before
    if projection == "macs_isotropic":
        assert _composite_cache_signature(scope, config_override=config) != result_before


def test_macs_replay_does_not_extrapolate_to_unmatched_energy(macs_groups):
    root, sample, dummy, _, config = macs_groups
    dummy.datasets[0].metadata["reduction_overrides"] = {"incident_energy_override": 4.5}
    with pytest.raises(ValueError, match="energ|matching"):
        composite_dataset_data(root, node=sample, config_override=config)


def test_macs_subtraction_rejects_untracked_sample_symmetry_copies(macs_groups):
    from nfit.measurement_dependencies import SourceReplayRequired

    root, sample, _, _, config = macs_groups
    config["symmetry"] = {"mode": "space_group", "expression": "P -1"}
    with pytest.raises(SourceReplayRequired, match="symmetry"):
        composite_dataset_data(root, node=sample, config_override=config)


def test_macs_replay_provenance_names_background_acquisitions(macs_groups):
    from nfit import project_measured_background_macs
    from nfit.source_lineage import source_identity, tracked_source_ids

    root, sample, dummy, _, config = macs_groups
    sample_path = sample.datasets[0].metadata["source_file"]
    dummy_path = dummy.datasets[0].metadata["source_file"]
    target = composite_dataset_data(root, node=sample, config_override=config,
                                    apply_backgrounds=False)
    target = target.with_updates(metadata={**target.metadata,
        "source_selection_lineage": {"version": 1, "source_identity": source_identity(
            sample_path, importer="macs_nexus", stream="spec")}})
    replayed = project_measured_background_macs(sample, dummy, target)
    assert replayed.metadata["source_file"] == dummy_path
    assert replayed.metadata["source_files"] == [dummy_path]
    assert replayed.metadata["source_datasets"] == [dummy.datasets[0].name]
    assert tracked_source_ids(replayed) == (source_identity(
        dummy_path, importer="macs_nexus", stream="spec"),)
    assert replayed.metadata["combined_scientific_identity"] == target.metadata["combined_scientific_identity"]
    corrected = composite_dataset_data(root, node=sample, config_override=config)
    assert corrected.metadata["background_subtractions"][-1]["source"] == dummy_path


def test_macs_isotropic_gui_width_updates_saved_projection(macs_groups, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import NfitProjectExplorer

    root, sample, _, link, _ = macs_groups
    explorer = NfitProjectExplorer(NfitProject([root]))
    monkeypatch.setattr(explorer, "_background_changed", lambda *_: None)
    explorer._set_background_details(root, sample, link)
    projection = explorer.details_widget.findChild(widgets.QComboBox, "background_projection")
    width = explorer.details_widget.findChild(widgets.QDoubleSpinBox, "background_q_bin_width")
    interpolation = explorer.details_widget.findChild(widgets.QComboBox, "background_interpolation")
    assert projection.findData("macs_isotropic") >= 0
    assert width.value() == pytest.approx(.05)
    assert "monitor" in width.toolTip()
    projection.setCurrentIndex(projection.findData("macs_isotropic"))
    width.setValue(.025)
    width.editingFinished.emit()
    assert link.projection == "macs_isotropic"
    assert link.metadata["q_bin_width"] == pytest.approx(.025)
    assert interpolation.isEnabled()
    explorer.window.close()


def test_macs_isotropic_width_invalidates_result_and_background_caches(macs_groups):
    from nfit.project_composites import _composite_cache_signature

    root, sample, _, link, config = macs_groups
    link.projection = "macs_isotropic"
    scope = _composite_scope(root, sample)
    background_before = _composite_background_signature(scope, link, config)
    result_before = _composite_cache_signature(scope, config_override=config)
    link.metadata["q_bin_width"] = .025
    assert _composite_background_signature(scope, link, config) != background_before
    assert _composite_cache_signature(scope, config_override=config) != result_before


def test_macs_isotropic_script_and_archive_round_trip(macs_groups, tmp_path):
    root, sample, _, link, config = macs_groups
    link.projection = "macs_isotropic"
    link.interpolation = "nearest"
    link.metadata["q_bin_width"] = .3
    for axis in config["axes"][:3]:
        axis.update(lower=-4., upper=4., num_bins=31)
    config["axes"][3].update(lower=-.3, upper=-.1)
    project = NfitProject([root])
    expected = composite_dataset_data(root, node=sample, config_override=config)
    assert np.any(~expected.mask)
    assert expected.metadata["background_subtractions"][-1]["projection"]["mode"] == "macs_isotropic"
    namespace = {"__name__": "standalone"}
    exec(compile(composite_workflow_script(project, root.name, node_id=sample.id),
                 "macs_isotropic.py", "exec"), namespace)
    replayed = namespace["run"]()
    path = tmp_path / "isotropic.nfit"
    save_project(project, path)
    restored_root = load_project(path).data_groups[0]
    restored_sample = restored_root.subgroups[0]
    assert restored_sample.backgrounds[0].metadata["q_bin_width"] == .3
    restored = composite_dataset_data(restored_root, node=restored_sample, config_override=config)
    for actual in (replayed, restored):
        np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
        np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
        np.testing.assert_array_equal(actual.mask, expected.mask)
        assert actual.source_dependencies is not None
