"""Independent exposure and covariance references for directional event replay."""

from dataclasses import replace

import numpy as np
import pytest

import nfit.mdevent_background as replay
from nfit import (
    bin_mdevent_group,
    mdevent_dataset_group,
    set_dgs_reduction_policies,
    set_dgs_trajectory_energy_policy,
)
from nfit.dgs_reduction_policy import ENERGY_TO_K
from tests.test_mdevent import _write_mdevent


def _directional_fixture(tmp_path, *, repeated_angle=False):
    h5py = pytest.importorskip("h5py")
    source_path, sample_path = (tmp_path / name for name in ("background.nxs", "sample.nxs"))
    # One lab-fixed detector, one weighted source observation, and two sample
    # orientations. The entire one-meV trajectory stays inside each spatial bin,
    # so its analytic exposure is simply charge times energy width.
    k = np.sqrt(10 * ENERGY_TO_K)
    lab = np.array([-k / 2, 0.0, k * (1 - np.sqrt(3) / 2)])
    rotation = np.eye(3) if repeated_angle else np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for path in (source_path, sample_path):
        _write_mdevent(path)
        with h5py.File(path, "r+") as handle:
            workspace = handle["MDEventWorkspace"]
            for index in range(2):
                experiment = workspace[f"experiment{index}"]
                experiment["instrument/physical_detectors/polar_angle"][...] = [30.]
                matrix = rotation if path == sample_path and index == 1 else np.eye(3)
                experiment["logs/goniometer/rotation_matrix"][...] = matrix.ravel()
                charge = 2. if path == source_path else (1., 3.)[index]
                experiment["logs/gd_prtn_chrg/value"][...] = [charge]
            del workspace["event_data/event_data"]
            workspace["event_data"].create_dataset(
                "event_data", data=np.array([[8., 8., 0, 0, 10, *lab, 0.]])
            )
    source = mdevent_dataset_group(source_path)
    source.datasets = source.datasets[:1]
    sample = mdevent_dataset_group(sample_path)
    return sample, source


@pytest.fixture(params=["numpy", "numba"])
def backend(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setattr(replay, "_REPLAY_NUMBA", None)
    elif replay._REPLAY_NUMBA is None:
        pytest.skip("Numba replay is unavailable")
    return request.param


def _target(sample, bins):
    return bin_mdevent_group(
        sample, lower=[-2., -2., -2., -.5], upper=[.5, 2., 2., .5],
        num_bins=[bins, 1, 1, 1],
    )


def test_unequal_angle_exposure_and_background_calibration_match_analytic_reference(
    tmp_path, backend,
):
    sample, source = _directional_fixture(tmp_path)
    # Different charge and fit weight must both enter the relative angle
    # exposure; source fit weight enters C and N, calibration enters C and V.
    sample.datasets[0].fit_weight = 2.
    sample.datasets[1].fit_weight = .5
    source.datasets[0].fit_weight = .4
    source.datasets[0].scale_factor = 2.5
    result = replay.project_measured_background_mdevent(sample, source, _target(sample, 2))
    fractions = np.array([4 / 7, 3 / 7])
    numerator = 8 * (2.5 * .4) * fractions
    variance = 8 * ((2.5 * .4) * fractions)**2
    exposure = 2 * .4 * fractions  # charge times fit weight times one-meV width
    np.testing.assert_allclose(result.metadata["normalization_denominator"].ravel(), exposure)
    np.testing.assert_allclose(result.signal.ravel(), numerator / exposure)
    np.testing.assert_allclose(result.errors.ravel(), np.sqrt(variance) / exposure)
    np.testing.assert_array_equal(result.num_events.ravel(), [1., 1.])
    assert not result.mask.any()


def test_repeated_angles_preserve_one_source_observation_uncertainty(tmp_path, backend):
    sample, source = _directional_fixture(tmp_path, repeated_angle=True)
    result = replay.project_measured_background_mdevent(sample, source, _target(sample, 1))
    np.testing.assert_allclose(result.signal.item(), 8 / 2)
    np.testing.assert_allclose(result.errors.item(), np.sqrt(8) / 2)
    np.testing.assert_allclose(result.metadata["normalization_denominator"].item(), 2)
    assert result.num_events.item() == 1
    # Splitting one acquisition into more angle entries changes neither its
    # exposure nor the information content of the background source.
    split = [replace(sample.datasets[0], fit_weight=1 / 5) for _ in range(5)]
    split.extend([replace(sample.datasets[1], fit_weight=1 / 3) for _ in range(3)])
    repeated = replay.project_measured_background_mdevent(
        sample, source, _target(sample, 1), datasets=split,
    )
    np.testing.assert_allclose(repeated.signal, result.signal)
    np.testing.assert_allclose(repeated.errors, result.errors)
    np.testing.assert_array_equal(repeated.num_events, result.num_events)


def test_final_grid_replay_retains_covariance_between_angle_copies(tmp_path, backend):
    sample, source = _directional_fixture(tmp_path)
    fine = replay.project_measured_background_mdevent(sample, source, _target(sample, 2))
    final = replay.project_measured_background_mdevent(sample, source, _target(sample, 1))
    fractions = np.array([.25, .75])
    # The primitive is one Poisson count, not two independent observations.
    # Cov(C_i,C_j)=8*f_i*f_j; summing all terms gives V=8*(sum f)^2.
    covariance = 8 * np.outer(fractions, fractions)
    exposure = 2 * fractions
    np.testing.assert_allclose(fine.errors.ravel()**2 * exposure**2, covariance.diagonal())
    np.testing.assert_allclose(final.errors.item()**2, covariance.sum() / exposure.sum()**2)
    np.testing.assert_allclose(final.signal.item(), 4.)
    assert final.num_events.item() == 1
    # Merely summing stored marginal variances would omit positive covariance.
    # Until replay sources retain sensitivities, re-run on the final cut grid.
    diagonal_variance = covariance.trace() / exposure.sum()**2
    assert diagonal_variance < final.errors.item()**2


@pytest.mark.parametrize("policy,expected_exposure", [("first_run", 2.), ("per_run", 2.01)])
def test_background_incident_energy_reference_is_resolved_before_run_partition(
    tmp_path, backend, policy, expected_exposure,
):
    h5py = pytest.importorskip("h5py")
    sample, source = _directional_fixture(tmp_path, repeated_angle=True)
    path = source.datasets[0].metadata["source_file"]
    with h5py.File(path, "r+") as handle:
        workspace = handle["MDEventWorkspace"]
        row = workspace["event_data/event_data"][()]
        second = row.copy()
        second[:, 2] = 1
        del workspace["event_data/event_data"]
        workspace["event_data"].create_dataset("event_data", data=np.concatenate((row, second)))
        workspace["experiment1/logs/Ei/value"][...] = [10.005]
    source = mdevent_dataset_group(path)
    set_dgs_trajectory_energy_policy(source, policy)
    # Isolate Ei selection from the separate six-significant-digit Mantid
    # extent convention, so the requested clipping point is exact.
    set_dgs_reduction_policies(sample, event_precision_policy="high_precision")
    # The spatial box clips the detector trajectory at kf=sqrt(10*ENERGY_TO_K).
    # With Ei=10 this selects E=[-.5,0], width .5 meV. Ei=10.005 instead
    # selects E=[-.5,.005], width .505. Each independent source has charge2.
    boundary = -np.sqrt(10 * ENERGY_TO_K) / 2
    target = bin_mdevent_group(
        sample, lower=[-2., -2., -2., -.5], upper=[boundary, 2., 2., .5],
        num_bins=[1, 1, 1, 1],
    )
    result = replay.project_measured_background_mdevent(sample, source, target)
    np.testing.assert_allclose(
        result.metadata["normalization_denominator"].item(), expected_exposure,
        rtol=1e-12, atol=0,
    )
    np.testing.assert_allclose(result.signal.item(), 16 / expected_exposure)
    np.testing.assert_allclose(result.errors.item(), np.sqrt(16) / expected_exposure)
    assert result.num_events.item() == 2
