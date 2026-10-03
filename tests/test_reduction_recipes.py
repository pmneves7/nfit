"""Persistent native source recipes and atomic, source-specific edits."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from nfit.pipeline import BackgroundSpec, DatasetEntry, DatasetGroup, MaskSpec
from nfit.reduction_recipes import (
    apply_reduction_recipe,
    effective_reduction_config,
    ensure_reduction_recipe,
    export_reduction_recipe,
    get_reduction_overrides,
    reduction_family,
    reduction_recipe_script,
    reduction_settings_schema,
    replay_reduction_recipe,
    resolved_reduction_values,
    set_reduction_settings,
    validated_reduction_settings,
)


def _group(tmp_path, format_name="raw-direct-geometry-nexus"):
    sources = [tmp_path / "run1.nxs", tmp_path / "run2.nxs"]
    for source in sources:
        source.write_bytes(b"source")
    datasets = [DatasetEntry(f"run {index}", None, kind="raw_dgs_nexus", metadata={
        "source_file": str(path), "run_number": str(index), "incident_energy": 60.0 + index,
        "t0": -3.0 - index, "instrument_name": f"instrument {index}",
        "geometry_signature": f"geometry {index}", "l1": 20.0 + index})
        for index, path in enumerate(sources, 1)]
    config = {"format": format_name, "source_files": [str(path) for path in sources], "ub_matrix": np.eye(3).tolist()}
    key = "mdevent" if format_name == "mantid-mdevent" else "raw_dgs"
    return DatasetGroup("runs", datasets=datasets, metadata={key: config, "composite": {"enabled": True},
        "composite_binnings": {"items": [{"config": {"enabled": True}}]}})


@pytest.mark.parametrize("format_name", ["raw-direct-geometry-nexus", "corelli-correlation-nexus", "mantid-mdevent"])
def test_schema_is_complete_readable_and_has_no_inert_mde_t0(tmp_path, format_name):
    group = _group(tmp_path, format_name)
    schema = reduction_settings_schema(group)
    assert all(setting.label and setting.tooltip and setting.scope for setting in schema)
    assert len({item.key for item in schema}) == len(schema)
    assert reduction_family(group) == format_name
    assert reduction_family(DatasetGroup("ordinary")) is None
    settings = {item.key: item for item in schema}
    assert settings["ub_matrix"].scope == "coordinates"
    assert not settings["ub_matrix"].per_run
    if format_name == "mantid-mdevent":
        assert "t0_override" not in settings
        assert settings["incident_energy_override"].scope == "normalization"
    if format_name != "corelli-correlation-nexus":
        for key in ("event_precision_policy", "trajectory_energy_policy", "symmetry_variance_policy"):
            assert not settings[key].per_run


def test_effective_read_is_pure_preserves_legacy_absent_keys_and_stable_id_clones(tmp_path):
    group = _group(tmp_path)
    first = group.datasets[0]
    first.metadata["reduction_overrides"] = {"t0_override": -11.0}
    before = copy.deepcopy(group.metadata)
    config = effective_reduction_config(group, copy.copy(first))
    assert config["t0_override"] == -11.0
    assert "bad_pulse_threshold" not in config
    assert "reduction_recipe" not in group.metadata
    assert group.metadata == before
    assert get_reduction_overrides(group, first.id) == {"t0_override": -11.0}
    with pytest.raises(ValueError, match="belong"):
        effective_reduction_config(group, "not a member")


def test_transient_duplicate_id_views_read_their_own_masks_and_overrides(tmp_path):
    group = _group(tmp_path)
    first, second = group.datasets
    second.id = first.id
    first.metadata["reduction_overrides"] = {"t0_override": -5.0, "mask_file": "first-mask.nxs"}
    second.metadata["reduction_overrides"] = {"t0_override": -12.0, "mask_file": "second-mask.nxs"}
    first_config = effective_reduction_config(group, first)
    second_config = effective_reduction_config(group, second)
    assert first_config["t0_override"] == -5 and first_config["mask_file"] == "first-mask.nxs"
    assert second_config["t0_override"] == -12 and second_config["mask_file"] == "second-mask.nxs"
    assert get_reduction_overrides(group, second) == second.metadata["reduction_overrides"]
    assert resolved_reduction_values(group, second)["t0_override"]["value"] == -12
    before = copy.deepcopy(group.metadata)
    with pytest.raises(ValueError, match="belong"):
        set_reduction_settings(group, {"t0_override": -20}, dataset_ids=[first.id])
    assert group.metadata == before
    with pytest.raises(ValueError, match="belong"):
        effective_reduction_config(group, copy.copy(first))


def test_recipe_retains_selector_per_run_geometry_resolved_values_and_current_membership(tmp_path, monkeypatch):
    group = _group(tmp_path)
    first, second = group.datasets
    first.metadata["resolved_reduction"] = {"version": 1, "automatic_values": {
        "incident_energy_meV": 61.012, "t0_microseconds": -4.25}, "provenance": {"monitor": "fit"}}
    first.metadata["goniometer_matrix"] = np.eye(3)
    selector = {"mode": "expression", "directory": str(tmp_path), "prefix": "run", "suffix": ".nxs",
                "expression": "1:2", "padding": 0}
    before = copy.deepcopy(group.metadata["raw_dgs"])
    recipe = ensure_reduction_recipe(group, source_selection=selector)
    assert recipe["source_selection"]["expression"] == "1:2"
    assert [run["acquisition"]["geometry_signature"] for run in recipe["runs"]] == ["geometry 1", "geometry 2"]
    assert recipe["runs"][0]["resolved_automatic_values"]["t0_override"]["value"] == -4.25
    assert recipe["runs"][0]["resolved_reduction"]["provenance"] == {"monitor": "fit"}
    assert recipe["runs"][0]["acquisition"]["goniometer_matrix"] == np.eye(3).tolist()
    assert group.metadata["raw_dgs"] == before  # no manufactured default cache keys
    group.datasets.remove(second)
    retained = ensure_reduction_recipe(group)
    assert retained["source_selection"]["resolved_files"] == [first.metadata["source_file"]]
    assert retained["source_selection"]["expression"] == "1:2"
    assert retained["source_selection"]["resolved_runs"][0]["dataset_id"] == first.id
    assert len(retained["provenance"]["source_files"]) == 1
    json.dumps(retained, allow_nan=False)


def test_recipe_calibration_fingerprints_each_unique_path_once(tmp_path, monkeypatch):
    import nfit.reduction_recipes as module

    group = _group(tmp_path)
    calibration = tmp_path / "vanadium.nxs"
    calibration.write_bytes(b"calibration")
    group.metadata["raw_dgs"].update(normalization_file=str(calibration), mask_file=str(calibration))
    original = module._file_provenance
    calls = []

    def capture(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(module, "_file_provenance", capture)
    recipe = ensure_reduction_recipe(group)
    assert calls.count(str(calibration)) == 1
    assert recipe["provenance"]["calibrations_by_run"][group.datasets[0].id]["normalization_file"]["size_bytes"] == 11


def test_negative_t0_explicit_automatic_and_inherit_are_distinct(tmp_path):
    group = _group(tmp_path)
    first, second = group.datasets
    set_reduction_settings(group, {"t0_override": -10})
    set_reduction_settings(group, {"t0_override": None}, dataset_ids=[first.id])
    assert effective_reduction_config(group, first)["t0_override"] is None
    assert effective_reduction_config(group, second)["t0_override"] == -10
    assert resolved_reduction_values(group, first)["t0_override"]["automatic"]
    assert resolved_reduction_values(group, first)["t0_override"]["value"] == -4
    set_reduction_settings(group, {"t0_override": None}, dataset_ids=[first.id], inherit=True)
    assert get_reduction_overrides(group, first) == {}
    assert effective_reduction_config(group, first)["t0_override"] == -10


def test_hyspec_controls_export_automatic_resolution_and_targeted_replay(tmp_path):
    group = _group(tmp_path)
    first, second = group.datasets
    first.metadata["resolved_reduction"] = {"automatic_values": {"hyspec_tank_offset_degrees": 1.25}}
    settings = {item.key: item for item in reduction_settings_schema(group)}
    assert settings["hyspec_tof_crop"].default is True
    assert settings["hyspec_tank_offset_override"].automatic
    set_reduction_settings(group, {"hyspec_tof_crop": False})
    set_reduction_settings(group, {"hyspec_tank_offset_override": 0.}, dataset_ids=[second.id])
    resolved = resolved_reduction_values(group, first)["hyspec_tank_offset_override"]
    assert resolved["automatic"] and resolved["value"] == 1.25
    assert resolved["units"] == "deg"
    assert resolved_reduction_values(group, second)["hyspec_tank_offset_override"]["value"] == 0.
    recipe = export_reduction_recipe(group)
    replay = _group(tmp_path)
    for current, original in zip(replay.datasets, group.datasets, strict=True):
        current.id = original.id
    apply_reduction_recipe(replay, recipe)
    assert effective_reduction_config(replay, replay.datasets[0])["hyspec_tof_crop"] is False
    assert effective_reduction_config(replay, replay.datasets[1])["hyspec_tank_offset_override"] == 0.


def test_only_effectively_affected_run_caches_are_removed(tmp_path):
    group = _group(tmp_path)
    first, second = group.datasets
    first._raw_dgs_reduction_cache = object()
    second._raw_dgs_reduction_cache = object()
    first.metadata["resolved_reduction"] = {"values": {"t0_microseconds": -4}}
    first.parameters["rebin"] = {"enabled": True}
    second.parameters["rebin"] = {"enabled": True}
    retained = second._raw_dgs_reduction_cache
    changed = set_reduction_settings(group, {"t0_override": -8}, dataset_ids=[first.id])
    assert changed.changed_dataset_ids == (first.id,)
    assert changed.changed_keys == ("t0_override",)
    assert first._raw_dgs_reduction_cache is None
    assert second._raw_dgs_reduction_cache is retained
    assert first.parameters["rebin"]["stale"]
    assert "stale" not in second.parameters["rebin"]
    assert first.metadata["resolved_reduction"]["stale"]
    assert group.metadata["composite"]["stale"]
    assert group.metadata["composite_binnings"]["items"][0]["config"]["stale"]


def test_explicit_default_equal_inheritance_keeps_cache_and_histogram(tmp_path):
    group = _group(tmp_path)
    first = group.datasets[0]
    first._raw_dgs_reduction_cache = object()
    cache = first._raw_dgs_reduction_cache
    edit = set_reduction_settings(group, {"he3_detector_efficiency_correction": True}, dataset_ids=[first.id])
    assert edit.changed_dataset_ids == ()
    assert edit.changed_keys == ()
    assert edit.settings_changed
    assert first._raw_dgs_reduction_cache is cache
    assert "stale" not in group.metadata["composite"]
    repeated = set_reduction_settings(group, {"he3_detector_efficiency_correction": True}, dataset_ids=[first.id])
    assert not repeated.settings_changed


@pytest.mark.parametrize("updates", [
    {"t0_override": float("nan")}, {"incident_energy_override": 0}, {"incident_energy_override": -60},
    {"bad_pulse_threshold": -1}, {"energy_min_fraction": 0.95}, {"energy_max_fraction": 1.0},
    {"ki_kf_normalization": "false"}, {"t0_override": True}, {"unknown": 1},
    {"ub_matrix": np.zeros((3, 3)).tolist()},
])
def test_invalid_edits_are_atomic(tmp_path, updates):
    group = _group(tmp_path)
    before = copy.deepcopy(group.metadata)
    first = group.datasets[0]
    first._raw_dgs_reduction_cache = object()
    cache = first._raw_dgs_reduction_cache
    with pytest.raises(ValueError):
        set_reduction_settings(group, updates)
    assert group.metadata == before
    assert first._raw_dgs_reduction_cache is cache


def test_shared_edit_validates_all_run_bound_overrides_before_mutation(tmp_path):
    group = _group(tmp_path)
    first = group.datasets[0]
    set_reduction_settings(group, {"energy_min_fraction": 0.7}, dataset_ids=[first.id])
    before = copy.deepcopy(group.metadata)
    with pytest.raises(ValueError, match="energy fractions"):
        set_reduction_settings(group, {"energy_max_fraction": 0.6})
    assert group.metadata == before


def test_shared_only_settings_and_corelli_bounds(tmp_path):
    group = _group(tmp_path)
    for key, value in (("ub_matrix", np.eye(3)), ("trajectory_energy_policy", "per_run"),
                       ("symmetry_variance_policy", "within_bin_covariance"), ("event_precision_policy", "high_precision")):
        with pytest.raises(ValueError, match="shared group"):
            set_reduction_settings(group, {key: value}, dataset_ids=[group.datasets[0].id])
    corelli = _group(tmp_path, "corelli-correlation-nexus")
    assert validated_reduction_settings(corelli, {"timing_offset_ns": -15}) == {"timing_offset_ns": -15}
    with pytest.raises(ValueError, match="wavelength"):
        set_reduction_settings(corelli, {"wavelength_min_angstrom": 3})
    mde = _group(tmp_path, "mantid-mdevent")
    with pytest.raises(ValueError, match="unsupported"):
        set_reduction_settings(mde, {"t0_override": -5})


def test_coordinate_histogram_and_storage_edits_keep_raw_reductions(tmp_path):
    group = _group(tmp_path)
    cache = object()
    group.datasets[0]._raw_dgs_reduction_cache = cache
    edit = set_reduction_settings(group, {"ub_matrix": (2*np.eye(3)).tolist()})
    assert edit.scopes == ("coordinates",)
    assert group.datasets[0]._raw_dgs_reduction_cache is cache
    edit = set_reduction_settings(group, {"symmetry_variance_policy": "within_bin_covariance"})
    assert edit.scopes == ("histogram",)
    assert group.datasets[0]._raw_dgs_reduction_cache is cache
    edit = set_reduction_settings(group, {"trajectory_energy_policy": "per_run"})
    assert edit.scopes == ("normalization",)
    assert group.datasets[0]._raw_dgs_reduction_cache is cache
    group.metadata["composite"].pop("stale")
    edit = set_reduction_settings(group, {"cache_reduced_events": False})
    assert edit.scopes == ("storage",)
    assert edit.changed_dataset_ids == ()
    assert "stale" not in group.metadata["composite"]


def test_export_is_pure_and_retains_scientific_per_run_settings_without_caches(tmp_path):
    group = _group(tmp_path)
    first = group.datasets[0]
    first.metadata["raw_dgs_reduction_cache"] = {"asset": "project event cache"}
    first.metadata["_project_path"] = "private project cache context"
    first.parameters["fit_likelihood"] = "gaussian"
    first.masks = [MaskSpec("mask", parameters={"axis": 0})]
    first.backgrounds = [BackgroundSpec("background", source_dataset_id=group.datasets[1].id, source_entry=group.datasets[1])]
    before = copy.deepcopy(group.metadata)
    recipe = export_reduction_recipe(group)
    assert group.metadata == before
    assert "raw_dgs_reduction_cache" not in recipe["datasets"][0]["metadata"]
    assert "_project_path" not in recipe["datasets"][0]["metadata"]
    assert recipe["datasets"][0]["parameters"] == {"fit_likelihood": "gaussian"}
    assert "source_entry" not in recipe["datasets"][0]["backgrounds"][0]
    assert recipe["datasets"][0]["masks"][0]["parameters"] == {"axis": 0}


def test_apply_authoritative_sections_is_lazy_preserves_object_graph_and_default_keys(tmp_path, monkeypatch):
    group = _group(tmp_path)
    first, second = group.datasets
    first.masks = [MaskSpec("existing")]
    first.backgrounds = [BackgroundSpec("link", source_entry=second, source_dataset_id=second.id)]
    first._raw_dgs_reduction_cache = object()
    second._raw_dgs_reduction_cache = object()
    first_cache, second_cache = first._raw_dgs_reduction_cache, second._raw_dgs_reduction_cache
    recipe = export_reduction_recipe(group)
    recipe["shared_defaults"]["t0_override"] = -2.5
    recipe["per_run_overrides"][second.id] = {"t0_override": None}
    recipe["datasets"].reverse()

    def unexpected(*args, **kwargs):
        raise AssertionError("existing source metadata must remain lazy")

    monkeypatch.setattr("nfit.raw_dgs.inspect_raw_dgs_run", unexpected)
    edit = apply_reduction_recipe(group, recipe)
    assert group.datasets == [second, first]
    assert group.datasets[0] is second and group.datasets[1] is first
    assert first.backgrounds[0].source_entry is second
    assert first.masks[0].name == "existing"
    assert first._raw_dgs_reduction_cache is None
    assert second._raw_dgs_reduction_cache is second_cache
    assert first_cache is not second_cache
    assert effective_reduction_config(group, first)["t0_override"] == -2.5
    assert effective_reduction_config(group, second)["t0_override"] is None
    assert "bad_pulse_threshold" not in group.metadata["raw_dgs"]
    assert edit.scopes == ("reduction", "sources")


def test_apply_remove_run_does_not_reduce_retained_runs_or_keep_stale_source_paths(tmp_path):
    group = _group(tmp_path)
    retained = group.datasets[1]
    retained._raw_dgs_reduction_cache = object()
    cache = retained._raw_dgs_reduction_cache
    recipe = export_reduction_recipe(group)
    recipe["datasets"] = recipe["datasets"][1:]
    recipe["source_selection"]["resolved_files"] = [retained.metadata["source_file"]]
    edit = apply_reduction_recipe(group, recipe)
    assert group.datasets == [retained]
    assert retained._raw_dgs_reduction_cache is cache
    assert edit.scopes == ("sources",)
    assert group.metadata["raw_dgs"]["source_files"] == [retained.metadata["source_file"]]
    assert group.metadata["reduction_recipe"]["source_selection"]["resolved_files"] == [retained.metadata["source_file"]]


def test_native_replay_matches_ids_overrides_and_histogram_without_qt(tmp_path):
    from nfit import bin_raw_dgs_group, raw_dgs_dataset_group
    from tests.test_raw_dgs import _write_raw_dgs

    path = tmp_path / "raw.nxs"
    _write_raw_dgs(path, with_he3=True)
    group = raw_dgs_dataset_group([path])
    set_reduction_settings(group, {"t0_override": -1.5}, dataset_ids=[group.datasets[0].id])
    recipe = export_reduction_recipe(group)
    replay = replay_reduction_recipe(recipe)
    assert replay.id == group.id and replay.datasets[0].id == group.datasets[0].id
    assert effective_reduction_config(replay, replay.datasets[0])["t0_override"] == -1.5
    options = {"lower": [-10, -10, -10, -100], "upper": [10, 10, 10, 20], "num_bins": [2]*4}
    expected = bin_raw_dgs_group(group, **options)
    actual = bin_raw_dgs_group(replay, **options)
    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)


def test_mde_recipe_selects_subset_of_container_experiments_and_no_inert_t0(tmp_path):
    from nfit import mdevent_dataset_group
    from tests.test_mdevent import _write_mdevent

    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    group = mdevent_dataset_group(path)
    group.datasets = group.datasets[:1]
    recipe = export_reduction_recipe(group)
    assert "t0_override" not in recipe["shared_defaults"]
    assert "t0_override" in recipe["provenance"]["stored_coordinate_parameters"]
    replay = replay_reduction_recipe(recipe)
    assert len(replay.datasets) == 1
    assert replay.datasets[0].id == group.datasets[0].id


def test_recipe_script_is_safe_python_and_replays(tmp_path, monkeypatch):
    from nfit import raw_dgs_dataset_group
    from tests.test_raw_dgs import _write_raw_dgs

    path = tmp_path / "raw.nxs"
    _write_raw_dgs(path)
    group = raw_dgs_dataset_group([path])
    script = reduction_recipe_script(group, group_variable="replayed")
    # Public export is installed by the coordinator of this change. Bind the
    # same API here so this focused test does not depend on integration order.
    monkeypatch.setattr("nfit.replay_reduction_recipe", replay_reduction_recipe, raising=False)
    namespace = {}
    exec(script, namespace)
    assert namespace["replayed"].datasets[0].id == group.datasets[0].id
    for name in ("class", "group.attr", "group);bad()", ""):
        with pytest.raises(ValueError, match="Python variable"):
            reduction_recipe_script(group, group_variable=name)


def test_recipe_validation_rejects_ambiguous_ids_and_invalid_json(tmp_path):
    group = _group(tmp_path)
    recipe = export_reduction_recipe(group)
    recipe["datasets"].append(copy.deepcopy(recipe["datasets"][0]))
    before = group.datasets[:]
    with pytest.raises(ValueError, match="duplicate source"):
        apply_reduction_recipe(group, recipe)
    assert group.datasets == before
    group.datasets[0].metadata["bad scientific value"] = float("nan")
    with pytest.raises(TypeError, match="finite JSON"):
        export_reduction_recipe(group)
    assert Path(before[0].metadata["source_file"]).exists()
