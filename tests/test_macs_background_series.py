"""One MACS collection pools matched corrections and untouched sample energies."""

import ast
from pathlib import Path

import h5py
import numpy as np
import pytest

from nfit import (
    DataGroup,
    DatasetGroup,
    NfitProject,
    composite_dataset_data,
    dataset_entry_from_path,
    load_project,
    save_project,
)
from nfit.macs_reduction import adopt_macs_reduction
from nfit.project_composites import _composite_scope, data_group_composite_config
from nfit.project_viewer_loading import plan_viewer_items
from nfit.reduction_recipes import set_reduction_settings
from nfit.workflow import composite_workflow_script
from tests.test_macs import _write_macs_nexus
from tests.test_macs_vanadium import _scan


def _source(path, *, energy=-.2, counts=9., monitor=200_000.):
    _write_macs_nexus(path, points=2, energy_transfer=energy)
    with h5py.File(path, "r+") as handle:
        logs = handle["entry/DAS_logs"]
        for stream in ("spec", "diff"):
            logs[f"{stream}Detector/counts"][...] = counts
        logs["counter/liveMonitor"][...] = monitor
        logs["kidneyMotor/softPosition"][...] = 0.
    return path


def _collection(paths, background_paths, *, stream="spec", energies=(-.3, .5), bins=2):
    entries = [dataset_entry_from_path(
        path, importer_name="macs_nexus", data_type="single_crystal_inelastic",
        importer_options={"stream": stream, "incident_energy_policy": "recorded",
                          "final_energy_policy": "recorded", "monitor_response": "constant",
                          "ki_kf_normalization": True, "mask_misaligned_analyzers": False},
    ) for path in paths]
    sample = DatasetGroup(f"MACS {stream.upper()}", datasets=entries)
    adopt_macs_reduction(sample)
    set_reduction_settings(sample, {
        "background_files": [str(path) for path in background_paths],
        "background_delta_e_tolerance_meV": .05,
        "background_method": "replay",
    })
    root = DataGroup("Ef5_2K", subgroups=[sample])
    config = data_group_composite_config(_composite_scope(root, sample))
    config.update(enabled=True, auto_rebin=False, minimum_coverage=0.,
                  resolution_mode="bins", fractional=False)
    for axis in config["axes"]:
        axis.update(lower=-100., upper=100., auto_lower=False, auto_upper=False,
                    num_bins=1, mode="bins", auto_step_size=False, fractional=False)
    config["axes"][-1].update(lower=energies[0], upper=energies[1], num_bins=bins)
    return root, sample, config


def _prepare(root, sample, config, *, backgrounds=True):
    return composite_dataset_data(root, node=sample, config_override=config,
                                  apply_backgrounds=backgrounds)


@pytest.mark.parametrize("stream", ["spec", "diff"])
def test_matched_background_and_unmatched_energy_preserve_exposure_pool(tmp_path, stream):
    paths = [_source(tmp_path / "matched.nxs", energy=-.2, counts=9.),
             _source(tmp_path / "unmatched.nxs", energy=.2, counts=16.)]
    background = _source(tmp_path / "dummy.nxs", energy=-.2, counts=4.)
    root, sample, config = _collection(paths, [background], stream=stream,
                                      bins=2 if stream == "spec" else 1)
    original = _prepare(root, sample, config, backgrounds=False)
    corrected = _prepare(root, sample, config)
    assert not np.any(corrected.mask)
    if stream == "spec":
        np.testing.assert_allclose(corrected.signal.ravel(),
                                   [25. * np.sqrt(3.5 / 3.7), 80. * np.sqrt(3.9 / 3.7)])
        assert corrected.signal.ravel()[1] == original.signal.ravel()[1]
        assert corrected.errors.ravel()[1] == original.errors.ravel()[1]
    else:
        # DIFF maps both acquisitions to E=0. Only the acquisition whose SPEC
        # energy transfer matches the dummy contributes to the correction.
        assert original.signal.item() == pytest.approx(62.5)
        assert corrected.signal.item() == pytest.approx(52.5)
    assert corrected.source_dependencies is not None
    np.testing.assert_allclose(corrected.source_dependencies.variance(), corrected.errors**2)
    assert "poisson_count_model" not in corrected.metadata
    assert "measurement_contract" not in corrected.metadata


def test_reused_background_retains_full_uncertainty_across_sample_runs(tmp_path):
    paths = [_source(tmp_path / f"sample-{index}.nxs", counts=9.) for index in range(2)]
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    root, sample, config = _collection(paths, [background], bins=1)
    original = _prepare(root, sample, config, backgrounds=False)
    corrected = _prepare(root, sample, config)
    one_root, one_sample, one_config = _collection(paths[:1], [background], bins=1)
    one_original = _prepare(one_root, one_sample, one_config, backgrounds=False)
    one_corrected = _prepare(one_root, one_sample, one_config)
    np.testing.assert_allclose(corrected.signal, one_corrected.signal)
    background_variance = corrected.errors**2 - original.errors**2
    one_background_variance = one_corrected.errors**2 - one_original.errors**2
    np.testing.assert_allclose(background_variance, one_background_variance)
    assert background_variance.item() > 0
    np.testing.assert_allclose(original.errors**2, one_original.errors**2 / 2)


def test_background_file_references_do_not_create_viewer_datasets(tmp_path):
    path = _source(tmp_path / "sample.nxs")
    background = _source(tmp_path / "dummy.nxs")
    root, spec, _ = _collection([path], [background])
    _, diff, _ = _collection([path], [background], stream="diff", bins=1)
    root.subgroups.append(diff)
    items = plan_viewer_items(root, use_composite=True, dataset_binnings=lambda _dataset: [])
    assert [item.source_name for item in items] == ["MACS SPEC Composite", "MACS DIFF Composite"]
    assert all(item.dataset is None for item in items)
    assert len(list(root.iter_datasets())) == 2
    assert all(entry.metadata["source_file"] == str(path) for entry in root.iter_datasets())
    assert spec.backgrounds == diff.backgrounds == []


def test_series_archive_and_exported_workflow_round_trip(tmp_path):
    paths = [_source(tmp_path / "sample.nxs"),
             _source(tmp_path / "unmatched.nxs", energy=.2, counts=16.)]
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    root, sample, config = _collection(paths, [background])
    project = NfitProject([root])
    expected = _prepare(root, sample, config)
    script = composite_workflow_script(project, root.name, node_id=sample.id)
    namespace = {"__name__": "standalone"}
    exec(compile(script, "macs_background_series.py", "exec"), namespace)
    replayed = namespace["run"]()
    destination = tmp_path / "series.nfit"
    save_project(project, destination)
    restored_root = load_project(destination).data_groups[0]
    restored_sample = restored_root.subgroups[0]
    assert restored_sample.metadata["macs"]["background_files"] == [str(background)]
    restored = _prepare(restored_root, restored_sample, config)
    for actual in (replayed, restored):
        np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
        np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
        np.testing.assert_array_equal(actual.mask, expected.mask)
        assert actual.source_dependencies is not None


def test_background_file_content_change_invalidates_composite_cache(tmp_path):
    path = _source(tmp_path / "sample.nxs", counts=9.)
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    root, sample, config = _collection([path], [background], bins=1)
    before = _prepare(root, sample, config)
    with h5py.File(background, "r+") as handle:
        handle["entry/DAS_logs/specDetector/counts"][...] = 8.
    after = _prepare(root, sample, config)
    assert before.signal.item() == pytest.approx(25. * np.sqrt(3.5 / 3.7))
    assert after.signal.item() == pytest.approx(5. * np.sqrt(3.5 / 3.7))
    assert after.errors.item() > before.errors.item()


def test_vanadium_settings_are_inherited_by_background_source(tmp_path):
    path = _source(tmp_path / "sample.nxs", counts=9.)
    background = _source(tmp_path / "dummy.nxs", counts=9.)
    calibrations = [_scan(tmp_path / "vanadium-a.nxs", kidney=0),
                    _scan(tmp_path / "vanadium-b.nxs", kidney=40)]
    root, sample, config = _collection([path], [background], bins=1)
    set_reduction_settings(sample, {"vanadium_files": [str(path) for path in calibrations],
                                    "vanadium_a4_mask_deg": 15.})
    original = _prepare(root, sample, config, backgrounds=False)
    corrected = _prepare(root, sample, config)
    assert not np.any(corrected.mask)
    np.testing.assert_allclose(corrected.signal, 0., atol=1e-12)
    np.testing.assert_allclose(corrected.errors, np.sqrt(2.) * original.errors)


def test_series_service_is_gui_independent_and_public():
    import nfit
    from nfit import macs_background_series

    assert nfit.apply_macs_background_series is macs_background_series.apply_macs_background_series
    tree = ast.parse(Path(macs_background_series.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith(("PySide", "project_gui", "project_data", "project_composites"))
                   for name in imports)


def test_per_run_correction_keeps_observations_below_collection_sample_threshold(tmp_path):
    paths = [_source(tmp_path / f"sample-{index}.nxs", counts=9.) for index in range(2)]
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    root, sample, config = _collection(paths, [background], bins=1)
    # Each acquisition has 40 points. Only the combined 80-point bin passes.
    config["minimum_samples"] = 60
    corrected = _prepare(root, sample, config)
    assert not corrected.mask.item()
    assert corrected.signal.item() == pytest.approx(25. * np.sqrt(3.5 / 3.7))


@pytest.mark.parametrize("unsupported", ["fractional", "symmetry"])
def test_series_rejects_untracked_sample_copy_dependence(tmp_path, unsupported):
    from nfit.measurement_dependencies import SourceReplayRequired

    path = _source(tmp_path / "sample.nxs")
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    root, sample, config = _collection([path], [background], bins=1)
    if unsupported == "fractional":
        config["fractional"] = True
        for axis in config["axes"]:
            axis["fractional"] = True
    else:
        config["symmetry"] = {"mode": "space_group", "expression": "P -1"}
    with pytest.raises(SourceReplayRequired, match=unsupported):
        _prepare(root, sample, config)


def test_series_service_rejects_unnormalized_sample(tmp_path):
    from nfit.macs_background_series import apply_macs_background_series

    path = _source(tmp_path / "sample.nxs")
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    root, sample, config = _collection([path], [background], bins=1)
    target = _prepare(root, sample, config, backgrounds=False)
    target = target.with_updates(metadata={**target.metadata,
        "rebin": {**target.metadata["rebin"], "normalize": False}})

    def unexpected_prepare(_runs):
        pytest.fail("The invalid statistical target must be rejected before source replay")

    with pytest.raises(ValueError, match="normalized exposure pooling"):
        apply_macs_background_series(sample, target, prepare_sample=unexpected_prepare)


def test_individual_scale_and_disable_change_only_background_statistics(tmp_path):
    from nfit import set_macs_background_source_settings

    path = _source(tmp_path / "sample.nxs", counts=9.)
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    root, sample, config = _collection([path], [background], bins=1)
    original = _prepare(root, sample, config, backgrounds=False)
    unit = _prepare(root, sample, config)
    set_macs_background_source_settings(sample, background, scale_factor=2.)
    doubled = _prepare(root, sample, config)
    np.testing.assert_allclose(original.signal - doubled.signal, 2 * (original.signal - unit.signal))
    np.testing.assert_allclose(doubled.errors**2 - original.errors**2,
                               4 * (unit.errors**2 - original.errors**2))
    set_macs_background_source_settings(sample, background, enabled=False)
    # Disabled files need not remain accessible and do not enter the projector.
    background.unlink()
    disabled = _prepare(root, sample, config)
    np.testing.assert_array_equal(disabled.signal, original.signal)
    np.testing.assert_array_equal(disabled.errors, original.errors)


def test_background_matching_uses_individual_energy_overrides(tmp_path):
    from nfit import set_macs_background_source_settings

    path = _source(tmp_path / "sample.nxs", counts=9.)
    background = _source(tmp_path / "dummy.nxs", energy=.2, counts=4.)
    root, sample, config = _collection([path], [background], bins=1)
    original = _prepare(root, sample, config)
    set_macs_background_source_settings(sample, background,
                                       reduction_overrides={"incident_energy_override": 3.5})
    corrected = _prepare(root, sample, config)
    assert corrected.signal.item() < original.signal.item()
    assert corrected.metadata["macs_background_series"]["matches"][0]["background_files"] == [str(background)]


def test_background_overrides_survive_complete_workflow_and_project_round_trip(tmp_path):
    from nfit import list_macs_background_sources, set_macs_background_source_settings

    path = _source(tmp_path / "sample.nxs", counts=9.)
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    root, sample, config = _collection([path], [background], bins=1)
    set_macs_background_source_settings(sample, background, scale_factor=.4,
                                       reduction_overrides={"a3_offset_deg": 12.})
    expected = _prepare(root, sample, config)
    project = NfitProject([root])
    script = composite_workflow_script(project, root.name, node_id=sample.id)
    namespace = {"__name__": "standalone"}
    exec(compile(script, "per_file_macs.py", "exec"), namespace)
    replayed = namespace["run"]()
    destination = tmp_path / "overrides.nfit"
    save_project(project, destination)
    restored_root = load_project(destination).data_groups[0]
    restored_sample = restored_root.subgroups[0]
    assert list_macs_background_sources(restored_sample) == list_macs_background_sources(sample)
    restored = _prepare(restored_root, restored_sample, config)
    for actual in (replayed, restored):
        np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
        np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
        np.testing.assert_array_equal(actual.mask, expected.mask)


def test_individual_calibration_contents_participate_in_composite_signature(tmp_path):
    from nfit import set_macs_background_source_settings
    from nfit.project_composites import _composite_cache_signature

    path = _source(tmp_path / "sample.nxs")
    background = _source(tmp_path / "dummy.nxs")
    calibration = _scan(tmp_path / "vanadium.nxs", kidney=0)
    root, sample, config = _collection([path], [background], bins=1)
    set_macs_background_source_settings(sample, background,
                                       reduction_overrides={"vanadium_files": [str(calibration)]})
    scope = _composite_scope(root, sample)
    before = _composite_cache_signature(scope, config_override=config)
    _scan(calibration, kidney=0, weak=.25)
    after = _composite_cache_signature(scope, config_override=config)
    assert before != after


def test_individual_detector_mask_changes_background_without_masking_sample(tmp_path):
    from nfit import set_macs_background_source_settings

    path = _source(tmp_path / "sample.nxs", counts=9.)
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    with h5py.File(background, "r+") as handle:
        handle["entry/DAS_logs/specDetector/counts"][:, 0] = 100.
    root, sample, config = _collection([path], [background], bins=1)
    original = _prepare(root, sample, config, backgrounds=False)
    before = _prepare(root, sample, config)
    set_macs_background_source_settings(sample, background,
                                       reduction_overrides={"masked_analyzer_channels": [1]})
    after = _prepare(root, sample, config)
    np.testing.assert_array_equal(after.mask, original.mask)
    assert after.signal.item() > before.signal.item()
    np.testing.assert_allclose(after.signal, 25. * np.sqrt(3.5 / 3.7))
    np.testing.assert_array_equal(_prepare(root, sample, config, backgrounds=False).signal, original.signal)


def test_background_reporting_units_are_converted_to_sample_units(tmp_path):
    from nfit import set_macs_background_source_settings

    path = _source(tmp_path / "sample.nxs", counts=9.)
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    root, sample, config = _collection([path], [background], bins=1)
    set_reduction_settings(sample, {"monitor_response": "inverse_velocity"})
    before = _prepare(root, sample, config)
    set_macs_background_source_settings(sample, background, reduction_overrides={
        "monitor_target": 2_000_000., "monitor_reference_k_inv_angstrom": 2.})
    after = _prepare(root, sample, config)
    np.testing.assert_allclose(after.signal, before.signal)
    np.testing.assert_allclose(after.errors, before.errors)
    record = after.metadata["macs_background_series"]["source_settings"][0]
    assert record["reduction_overrides"]["monitor_target"] == 2_000_000.
    assert record["reduction_overrides"]["monitor_reference_k_inv_angstrom"] == 2.


def test_background_response_policy_is_independent_and_applied(tmp_path):
    from nfit import set_macs_background_source_settings
    from nfit.macs_geometry import MACS_E_KSQ_MEV_ANGSTROM2

    path = _source(tmp_path / "sample.nxs", counts=9.)
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    root, sample, config = _collection([path], [background], bins=1)
    original = _prepare(root, sample, config, backgrounds=False)
    constant = _prepare(root, sample, config)
    set_macs_background_source_settings(sample, background, reduction_overrides={
        "monitor_response": "inverse_velocity", "incident_energy_policy": "bragg"})
    inverse_velocity = _prepare(root, sample, config)
    ki = np.sqrt(3.5 / MACS_E_KSQ_MEV_ANGSTROM2)
    np.testing.assert_allclose(original.signal - inverse_velocity.signal,
                               (original.signal - constant.signal) / ki)
    np.testing.assert_allclose(inverse_velocity.errors**2 - original.errors**2,
                               (constant.errors**2 - original.errors**2) / ki**2)


def test_background_can_use_its_own_vanadium_calibration(tmp_path):
    from nfit import set_macs_background_source_settings

    path = _source(tmp_path / "sample.nxs", counts=9.)
    background = _source(tmp_path / "dummy.nxs", counts=4.)
    calibrations = [_scan(tmp_path / "vanadium-a.nxs", kidney=0),
                    _scan(tmp_path / "vanadium-b.nxs", kidney=40)]
    root, sample, config = _collection([path], [background], bins=1)
    before = _prepare(root, sample, config)
    set_macs_background_source_settings(sample, background, reduction_overrides={
        "vanadium_files": [str(p) for p in calibrations], "vanadium_a4_mask_deg": 15.})
    after = _prepare(root, sample, config)
    assert not np.any(after.mask)
    assert not np.allclose(after.signal, before.signal)
    assert after.source_dependencies is not None
