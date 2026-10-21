"""Vanadium factors participate in the existing MACS counting estimator."""

import numpy as np
import pytest

from nfit import DataGroup
from nfit.macs import import_macs_nexus
from nfit.project_imports import _ensure_dataset_data_loaded, import_dataset_paths
from nfit.project_rebinning import _rebin_point_data
from nfit.reduction_recipes import (
    reduction_recipe_script,
    set_reduction_settings,
    validated_reduction_settings,
)
from tests.test_macs import _write_macs_nexus
from tests.test_macs_reduction import _single_cell_config
from tests.test_macs_vanadium import _scan


def test_factors_scale_counts_and_errors_and_retain_poisson_primitives(tmp_path):
    sample = _write_macs_nexus(tmp_path / "sample.ng0")
    files = [_scan(tmp_path / "van1.nxs", kidney=0),
             _scan(tmp_path / "van2.nxs", kidney=40)]
    original = import_macs_nexus(sample)
    calibrated = import_macs_nexus(sample, {"vanadium_files": files})
    factors = np.asarray(calibrated.metadata["macs_vanadium"]["factors"])
    np.testing.assert_allclose(calibrated.intensity.reshape(-1, 20),
                               original.intensity.reshape(-1, 20) * factors)
    np.testing.assert_allclose(calibrated.sigma.reshape(-1, 20),
                               original.sigma.reshape(-1, 20) * factors)
    assert factors[0] / factors[1] == pytest.approx(2.)
    np.testing.assert_array_equal(calibrated.measurement_payload["num_events"],
                                  original.measurement_payload["num_events"])
    np.testing.assert_allclose(calibrated.normalization_denominator.reshape(-1, 20),
                               original.normalization_denominator.reshape(-1, 20) / factors)
    pooled = _rebin_point_data(calibrated, _single_cell_config())
    count = calibrated.measurement_payload["num_events"][calibrated.mask].sum()
    exposure = calibrated.normalization_denominator[calibrated.mask].sum()
    assert pooled.signal.item() == pytest.approx(count / exposure)
    assert pooled.errors.item() == pytest.approx(np.sqrt(count) / exposure)


def test_missing_calibration_channels_mask_spec_only(tmp_path):
    sample = _write_macs_nexus(tmp_path / "sample.ng0")
    files = [_scan(tmp_path / "van.nxs")]
    data = import_macs_nexus(sample, {"vanadium_files": files})
    missing = np.array(data.metadata["macs_vanadium"]["missing_channels"]) - 1
    assert missing.size > 0
    assert not data.mask.reshape(-1, 20)[:, missing].any()
    assert set(missing + 1).issubset(data.metadata["masked_analyzer_channels"])
    diff = import_macs_nexus(sample, {"stream": "diff", "vanadium_files": files})
    original = import_macs_nexus(sample, {"stream": "diff"})
    np.testing.assert_array_equal(diff.intensity, original.intensity)
    assert not diff.metadata["vanadium_sensitivity_applied"]


def test_recipe_script_reproduces_calibrated_data_and_diagnostics(tmp_path):
    sample = _write_macs_nexus(tmp_path / "sample.ng0")
    files = [_scan(tmp_path / "van1.nxs", kidney=0),
             _scan(tmp_path / "van2.nxs", kidney=40)]
    root = DataGroup("MACS")
    import_dataset_paths(root, [sample])
    group = root.subgroups[0]
    set_reduction_settings(group, {"vanadium_files": files, "vanadium_a4_mask_deg": 10.})
    original = _ensure_dataset_data_loaded(group.datasets[0])
    namespace = {}
    exec(reduction_recipe_script(group), namespace)
    replay = _ensure_dataset_data_loaded(namespace["group"].datasets[0])
    np.testing.assert_allclose(replay.intensity, original.intensity)
    assert replay.metadata["macs_vanadium"] == original.metadata["macs_vanadium"]
    recipe = group.metadata["reduction_recipe"]
    provenance = recipe["provenance"]["calibrations_by_run"][group.datasets[0].id]
    assert [f["path"] for f in provenance["vanadium_files"]] == list(map(str, files))


def test_configuration_rejects_ambiguous_factors_and_bad_paths(tmp_path):
    fmt = "macs-step-nexus"
    with pytest.raises(ValueError, match="not both"):
        validated_reduction_settings(fmt, {"vanadium_files": [tmp_path / "van.nxs"],
                                           "apply_detector_efficiency": True})
    for value in ("file.nxs", [""], [2], None):
        with pytest.raises(ValueError):
            validated_reduction_settings(fmt, {"vanadium_files": value})
    for angle in (-1, 181, np.nan):
        with pytest.raises(ValueError):
            validated_reduction_settings(fmt, {"vanadium_a4_mask_deg": angle})


def test_background_compatibility_resolves_legacy_calibration_defaults(tmp_path):
    from nfit.macs_background import _MacsSourcePreparation

    sample = _write_macs_nexus(tmp_path / "sample.ng0")
    root = DataGroup("MACS")
    import_dataset_paths(root, [sample])
    group = root.subgroups[0]
    run = group.datasets[0]
    prepare = _MacsSourcePreparation()
    prepare(group, run, None, None)
    group.metadata["macs"].pop("vanadium_files")
    group.metadata["macs"].pop("vanadium_a4_mask_deg")
    prepare(group, run, None, None)
