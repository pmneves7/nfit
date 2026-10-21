"""MACS monitor pooling and correlated angle replay."""

import ast
from pathlib import Path

import h5py
import numpy as np
import pytest

from nfit.macs_background import (
    prepare_macs_background_sample,
    project_measured_background_macs,
)
from nfit.macs_reduction import MACS_DEFAULTS, MACS_REDUCTION_FORMAT
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_dependencies import SourceReplayRequired
from nfit.pipeline import DatasetEntry, DatasetGroup
from tests.test_macs import _write_macs_nexus


def _group(path, *, points=2, monitor=(10., 30.), counts=(4., 12.), angles=None):
    _write_macs_nexus(path, points=points, energy_transfer=-.2)
    with h5py.File(path, "r+") as handle:
        logs = handle["entry/DAS_logs"]
        logs["counter/liveMonitor"][...] = np.resize(monitor, points)
        logs["specDetector/counts"][...] = np.repeat(np.resize(counts, points)[:, None], 20, axis=1)
        logs["kidneyMotor/softPosition"][...] = 0.
        logs["sampleTheta/primaryNode"][...] = np.resize([0., 90.] if angles is None else angles, points)
    config = {**MACS_DEFAULTS, "format": MACS_REDUCTION_FORMAT, "stream": "spec",
              "monitor_target": 1., "monitor_response": "constant", "ki_kf_normalization": False,
              "mask_misaligned_analyzers": False, "a3_offset_deg": 0.}
    run = DatasetEntry(path.stem, None, metadata={"source_file": str(path), "importer": "macs_nexus"})
    return DatasetGroup(path.stem, datasets=[run], metadata={"macs": config})


def _target(split=False):
    values = [np.array([-100., 0., 100.]) if split else np.array([-100., 100.]),
              np.array([-100., 100.]), np.array([-100., 100.]), np.array([-.25, -.15])]
    axes = [MDHistoAxis(name, edge, "meV" if name == "E" else "RLU", "energy" if name == "E" else "momentum")
            for name, edge in zip("HKLE", values, strict=True)]
    shape = tuple(len(edge) - 1 for edge in values)
    return MDHistoData(axes=axes, signal=np.zeros(shape), errors=np.zeros(shape), mask=np.zeros(shape, bool),
                       num_events=np.zeros(shape), metadata={"measurement_reduction": {"assignment": "nearest_physical_bin"}})


def test_monitor_pooling_and_angle_reuse_do_not_create_counts(tmp_path):
    sample = _group(tmp_path / "sample.nxs", points=3)
    background = _group(tmp_path / "background.nxs")
    result = project_measured_background_macs(sample, background, _target())
    assert result.signal.item() == pytest.approx(16 / 40)
    assert result.errors.item() == pytest.approx(np.sqrt(20 * 16) / (20 * 40))
    repeat = _group(tmp_path / "repeat.nxs", points=6)
    repeated = project_measured_background_macs(repeat, background, _target())
    np.testing.assert_allclose(repeated.signal, result.signal)
    np.testing.assert_allclose(repeated.errors, result.errors)
    assert result.source_dependencies is not None


def test_background_a3_is_undone_and_cross_bin_covariance_retained(tmp_path):
    sample = _group(tmp_path / "sample.nxs", angles=[0, 180])
    background = _group(tmp_path / "background.nxs", angles=[-53, 21])
    first = project_measured_background_macs(sample, background, _target(split=True))
    with h5py.File(tmp_path / "background.nxs", "r+") as handle:
        handle["entry/DAS_logs/sampleTheta/primaryNode"][...] = [41, 149]
    second = project_measured_background_macs(sample, background, _target(split=True))
    np.testing.assert_allclose(first.signal, second.signal)
    np.testing.assert_allclose(first.errors, second.errors)
    dependencies = first.source_dependencies
    ids = [set(dependencies.source_indices[dependencies.observation_indices == i]) for i in range(2)]
    assert ids[0] & ids[1]


def test_full_rotation_removes_directional_asymmetry_but_half_rotation_need_not(tmp_path):
    half = _group(tmp_path / "half.nxs", monitor=(10.,), angles=[0, 90])
    full = _group(tmp_path / "full.nxs", points=4, monitor=(10.,), angles=[0, 90, 180, 270])
    background = _group(tmp_path / "background.nxs", monitor=(10.,))
    with h5py.File(tmp_path / "background.nxs", "r+") as handle:
        handle["entry/DAS_logs/specDetector/counts"][...] = np.arange(1., 21.)[None, :]
    half_result = project_measured_background_macs(half, background, _target(split=True))
    full_result = project_measured_background_macs(full, background, _target(split=True))
    assert not np.isclose(*half_result.signal.ravel())
    np.testing.assert_allclose(full_result.signal.ravel()[0], full_result.signal.ravel()[1])


def test_replay_checks_energy_stream_and_unsupported_transforms(tmp_path):
    sample = _group(tmp_path / "sample.nxs")
    background = _group(tmp_path / "background.nxs")
    background.metadata["macs"]["stream"] = "diff"
    with pytest.raises(ValueError, match="stream"):
        project_measured_background_macs(sample, background, _target())
    background.metadata["macs"]["stream"] = "spec"
    background.metadata["macs"]["incident_energy_override"] = 4.
    with pytest.raises(ValueError, match="energies"):
        project_measured_background_macs(sample, background, _target())
    background.metadata["macs"]["incident_energy_override"] = None
    background.datasets[0].transforms = [lambda data: data]
    with pytest.raises(ValueError, match="untransformed"):
        project_measured_background_macs(sample, background, _target())


def test_sample_tracking_requires_independent_bin_assignment():
    target = _target()
    tracked = prepare_macs_background_sample(target, source_signature="sample")
    assert tracked.source_dependencies is not None
    assert prepare_macs_background_sample(tracked, source_signature="sample") is tracked
    target.metadata["measurement_reduction"]["assignment"] = "fractional"
    with pytest.raises(SourceReplayRequired, match="fractional"):
        prepare_macs_background_sample(target, source_signature="sample")
    target = _target()
    target.metadata["rebin"] = {"symmetry": {"operations_hkl": [np.eye(3).tolist(), (-np.eye(3)).tolist()]}}
    with pytest.raises(SourceReplayRequired, match="symmetry"):
        prepare_macs_background_sample(target, source_signature="sample")


def test_batching_and_measured_zero_counts(tmp_path):
    sample = _group(tmp_path / "sample.nxs", points=100)
    background = _group(tmp_path / "background.nxs", counts=(0., 0.))
    expected = project_measured_background_macs(sample, background, _target(split=True))
    actual = project_measured_background_macs(sample, background, _target(split=True), max_batch_bytes=65536)
    np.testing.assert_allclose(expected.signal, actual.signal)
    np.testing.assert_allclose(expected.errors, actual.errors)
    assert not np.any(actual.mask)
    assert np.all(actual.signal == 0)
    assert np.all(actual.errors == 0)
    with pytest.raises(SourceReplayRequired, match="budget"):
        project_measured_background_macs(sample, background, _target(), max_batch_bytes=64)


def test_invalid_masked_detector_counts_do_not_poison_other_primitives(tmp_path):
    sample = _group(tmp_path / "sample.nxs")
    background = _group(tmp_path / "background.nxs")
    with h5py.File(tmp_path / "background.nxs", "r+") as handle:
        handle["entry/DAS_logs/specDetector/counts"][0, 0] = np.nan
        handle["entry/DAS_logs/specDetector/counts"][1, 0] = -1.
    result = project_measured_background_macs(sample, background, _target())
    assert result.signal.item() == pytest.approx(.4)
    assert np.all(np.isfinite(result.errors))
    assert np.all(np.isfinite(result.source_dependencies.source_variances))


def test_macs_replay_service_is_gui_independent_and_public():
    import nfit
    import nfit.macs_background as module

    assert nfit.project_measured_background_macs is module.project_measured_background_macs
    tree = ast.parse(Path(module.__file__).read_text())
    names = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith(("PySide", "project_gui", "project_data", "project_composites")) for name in names)
