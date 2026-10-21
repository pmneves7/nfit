"""Physical radial pooling retains uncertainty of the measured background."""

import ast
from pathlib import Path

import numpy as np
import pytest

from nfit.macs import import_macs_nexus
from nfit.macs_radial_background import (
    _radial_assignments,
    project_isotropic_background_macs,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_dependencies import SourceReplayRequired, project_source_dependencies
from tests.test_macs_background import _group


def _radial_target(sample, radius=1.):
    data = import_macs_nexus(sample.datasets[0].metadata["source_file"], {"stream": "spec"})
    rlu = np.asarray(data.metadata["rlu_to_inv_angstrom_matrix"])
    h = radius / np.linalg.norm(rlu[:, 0])
    edges = [np.array([-2 * h, 0., 2 * h]), np.array([-1e-5, 1e-5]),
             np.array([-1e-5, 1e-5]), np.array([-.25, -.15])]
    axes = [MDHistoAxis(name, values, "meV" if name == "E" else "RLU",
                       "energy" if name == "E" else "momentum")
            for name, values in zip("HKLE", edges, strict=True)]
    shape = (2, 1, 1, 1)
    return MDHistoData(axes=axes, signal=np.zeros(shape), errors=np.zeros(shape),
        mask=np.zeros(shape, bool), num_events=np.zeros(shape),
        metadata={"measurement_reduction": {"assignment": "nearest_physical_bin"}})


@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
def test_isotropic_radial_pooling_and_shared_uncertainty(tmp_path, interpolation):
    sample = _group(tmp_path / "sample.nxs")
    background = _group(tmp_path / "background.nxs")
    result = project_isotropic_background_macs(sample, background, _radial_target(sample),
                                               q_bin_width=.3, interpolation=interpolation)
    assert not np.any(result.mask)
    np.testing.assert_allclose(result.signal, .4)
    np.testing.assert_allclose(result.errors.ravel()[0], result.errors.ravel()[1])
    deps = result.source_dependencies
    mean = project_source_dependencies(deps, np.array([0, 1]), np.array([0, 0]), np.array([.5, .5]), (1,))
    assert mean.variances()[0] == pytest.approx(result.errors.ravel()[0] ** 2)
    assert result.metadata["background_projection"]["mode"] == "macs_isotropic"
    assert result.metadata["source_file"].endswith("background.nxs")


def test_radial_gaps_are_not_interpolated():
    centers = np.array([.5, 1.5, 2.5])
    measured = np.array([True, False, True])
    q = np.array([.5, 1., 1.5, 2., 2.5])
    output, bins, weights = _radial_assignments(q, centers, measured, 1., "linear")
    np.testing.assert_array_equal(output, [0, 4])
    np.testing.assert_array_equal(bins, [0, 2])
    np.testing.assert_array_equal(weights, [1., 1.])


def test_radial_projection_masks_extrapolation_and_checks_width(tmp_path):
    sample = _group(tmp_path / "sample.nxs")
    background = _group(tmp_path / "background.nxs")
    result = project_isotropic_background_macs(sample, background, _radial_target(sample, radius=100.))
    assert np.all(result.mask)
    with pytest.raises(ValueError, match="positive"):
        project_isotropic_background_macs(sample, background, _radial_target(sample), q_bin_width=0)
    with pytest.raises(SourceReplayRequired, match="radial grid"):
        project_isotropic_background_macs(sample, background, _radial_target(sample), q_bin_width=1e-20)


def test_radial_service_is_gui_independent_and_public():
    import nfit
    import nfit.macs_radial_background as module

    assert nfit.project_isotropic_background_macs is module.project_isotropic_background_macs
    tree = ast.parse(Path(module.__file__).read_text())
    names = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith(("PySide", "project_gui", "project_data", "project_composites")) for name in names)


def test_radial_source_reporting_units_and_policy_overrides_are_explicit(tmp_path):
    sample = _group(tmp_path / "sample.nxs")
    background = _group(tmp_path / "background.nxs")
    target = _radial_target(sample)
    expected = project_isotropic_background_macs(sample, background, target, q_bin_width=.3)
    background.metadata["macs"].update(monitor_target=4., final_energy_policy="recorded")
    with pytest.raises(ValueError, match="settings must match"):
        project_isotropic_background_macs(sample, background, target, q_bin_width=.3)
    actual = project_isotropic_background_macs(sample, background, target, q_bin_width=.3,
                                              allow_background_reduction_overrides=True)
    np.testing.assert_allclose(actual.signal, expected.signal)
    np.testing.assert_allclose(actual.errors, expected.errors)
    assert actual.metadata["background_projection"]["monitor_target"] == 1.
    assert actual.metadata["background_projection"]["sources"][0]["reduction_settings"]["monitor_target"] == 4.
