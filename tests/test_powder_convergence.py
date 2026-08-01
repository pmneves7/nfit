import numpy as np
import pytest

from nfit import (
    PowderConvergenceResult,
    is_powder_dataset,
    powder_convergence_scan,
)
from tests.fit_config_test_support import (
    _heisenberg_chain_component,
    _spin_fluctuation_points,
)


def _powder_component_and_data(*, exchange=0.05):
    q = np.array([0.25, 0.45, 0.7, 0.95, 1.2])
    energy = np.array([0.6, 1.4, 2.2, 3.1, 4.0])
    points = _spin_fluctuation_points(
        np.ones(q.size), q, energy, temperature=10.0
    )
    points.metadata.update(
        {
            "data_type": "powder_inelastic",
            "coordinate_units": "1/angstrom",
            "powder_q_modulus_axis": True,
            "spectral_observable": {
                "fit_representation": "chi_double_prime",
                "unit": "mu_B^2/meV/f.u.",
                "moment_unit": "mu_B_squared",
                "g_factor": 2.0,
                "kf_ki_state": "removed",
            },
        }
    )
    component = _heisenberg_chain_component()
    component.parameters.update({"chi0": 0.4, "gamma0": 2.0, "J1": exchange})
    component.fit_parameters = {}
    component.config["crystal"] = {
        "lattice": {
            "a": 8.0, "b": 8.0, "c": 8.0,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        }
    }
    component.config["powder_orientations"] = 26
    return component, points


def test_powder_dataset_detection_accepts_declared_and_axis_metadata():
    _component, points = _powder_component_and_data()
    assert is_powder_dataset(points) is True
    assert is_powder_dataset(points, "powder_elastic") is True

    plain = _spin_fluctuation_points(
        np.ones(3), np.array([0.2, 0.4, 0.6]), np.array([1.0, 2.0, 3.0]),
        temperature=10.0,
    )
    assert is_powder_dataset(plain) is False
    assert is_powder_dataset(plain, "single_crystal_inelastic") is False


def test_powder_scan_converges_towards_the_densest_quadrature():
    component, points = _powder_component_and_data()
    result = powder_convergence_scan(
        component,
        points,
        orientation_counts=(8, 20, 60, 180),
        relative_tolerance=5.0e-2,
    )

    assert isinstance(result, PowderConvergenceResult)
    assert result.orientation_counts == (8, 20, 60, 180)
    assert result.reference_count == 180
    assert result.values.shape == (4, points.size)
    # The reference row is compared against itself.
    assert result.max_absolute_deviation[-1] == 0.0
    assert result.max_relative_deviation[-1] == 0.0
    # Denser quadratures sit closer to the reference than the coarsest one.
    assert result.max_relative_deviation[0] > result.max_relative_deviation[2]
    assert result.provenance["reference_orientation_count"] == 180
    assert result.provenance["monotonic"] is False


def test_powder_scan_does_not_modify_the_supplied_component():
    component, points = _powder_component_and_data()
    powder_convergence_scan(
        component, points, orientation_counts=(8, 16, 40)
    )
    assert component.config["powder_orientations"] == 26


def test_powder_scan_reports_the_smallest_count_meeting_the_tolerance():
    component, points = _powder_component_and_data()
    loose = powder_convergence_scan(
        component,
        points,
        orientation_counts=(8, 20, 60, 180),
        relative_tolerance=1.0,
    )
    strict = powder_convergence_scan(
        component,
        points,
        orientation_counts=(8, 20, 60, 180),
        relative_tolerance=1.0e-14,
    )

    # A tolerance nothing but the reference can meet certifies only the
    # reference; an easy tolerance certifies the coarsest count.
    assert loose.converged_count == 8
    assert strict.converged_count == 180
    # Whatever is certified must actually satisfy the tolerance, and so must
    # every denser count -- the quadrature is not monotone in the count.
    for result, tolerance in ((loose, 1.0), (strict, 1.0e-14)):
        index = result.orientation_counts.index(result.converged_count)
        assert np.all(result.max_relative_deviation[index:] <= tolerance)


def test_powder_scan_rejects_invalid_input():
    component, points = _powder_component_and_data()
    with pytest.raises(ValueError, match="at least two orientation counts"):
        powder_convergence_scan(component, points, orientation_counts=(50,))
    with pytest.raises(ValueError, match="distinct"):
        powder_convergence_scan(component, points, orientation_counts=(20, 20))
    with pytest.raises(ValueError, match="at least six"):
        powder_convergence_scan(component, points, orientation_counts=(3, 20))
    with pytest.raises(ValueError, match="relative_tolerance"):
        powder_convergence_scan(
            component, points, orientation_counts=(8, 20), relative_tolerance=0.0
        )

    plain = _spin_fluctuation_points(
        np.ones(3), np.array([0.2, 0.4, 0.6]), np.array([1.0, 2.0, 3.0]),
        temperature=10.0,
    )
    with pytest.raises(ValueError, match="requires a powder dataset"):
        powder_convergence_scan(component, plain, orientation_counts=(8, 20))


def test_powder_convergence_result_round_trips_through_a_dict():
    component, points = _powder_component_and_data()
    result = powder_convergence_scan(
        component, points, orientation_counts=(8, 20, 60)
    )
    restored = PowderConvergenceResult.from_dict(result.to_dict())

    assert restored.orientation_counts == result.orientation_counts
    assert restored.reference_index == result.reference_index
    assert restored.converged_count == result.converged_count
    np.testing.assert_allclose(restored.values, result.values)
    np.testing.assert_allclose(
        restored.max_relative_deviation, result.max_relative_deviation
    )
    assert restored.provenance["component_type"] == "heisenberg_rpa"


def test_isotropic_powder_response_is_converged_at_every_count():
    """With no exchange the response is Q-isotropic, so orientation is moot.

    This separates a genuine angular-convergence signal from noise in the
    machinery: an isotropic model must agree across counts to machine
    precision, so any deviation the scan reports for a real model is physical.
    """

    component, points = _powder_component_and_data(exchange=0.0)
    result = powder_convergence_scan(
        component,
        points,
        orientation_counts=(8, 20, 60),
        relative_tolerance=1.0e-10,
    )
    assert np.max(result.max_relative_deviation) < 1.0e-12
    assert result.converged_count == 8


def test_converged_count_is_not_the_first_count_under_the_tolerance():
    """A coarse quadrature can pass by accident while a denser one fails.

    The sphere directions are a quasi-uniform spiral, not a nested sequence, so
    the deviation is not monotone in the count. At a 2e-3 tolerance the 8- and
    26-direction quadratures both pass while the 14-direction one does not, so
    reporting the first passing count would certify 8 and hide the failure just
    above it.
    """

    component, points = _powder_component_and_data(exchange=0.05)
    result = powder_convergence_scan(
        component,
        points,
        orientation_counts=(8, 14, 26, 50, 100),
        relative_tolerance=2.0e-3,
    )
    deviations = result.max_relative_deviation
    assert deviations[0] <= 2.0e-3   # 8 passes
    assert deviations[1] > 2.0e-3    # 14 fails
    assert deviations[2] <= 2.0e-3   # 26 passes
    assert result.converged_count == 26


def test_powder_geometry_deduplicates_q_modulus_before_orienting():
    """Collapsing repeated |Q| must not change the sampled orientations.

    A powder map repeats each |Q| across every energy bin, so the evaluator
    orients only the distinct moduli and re-expands the point map. That has to
    reproduce, exactly, what orienting every point separately produces --
    including the orientation-major ordering the powder average relies on.
    """

    from nfit.fit_config import (
        _powder_reciprocal_matrix,
        _powder_sphere_directions,
        _RpaComponentEvaluator,
    )
    from nfit.fitting import q_modulus_inv_angstrom
    from nfit.spin_fluctuations import build_rpa_geometry

    component, _unused = _powder_component_and_data()
    # Repeat each |Q| across several energies, as a real powder map does.
    q = np.repeat(np.array([0.25, 0.45, 0.7]), 4)
    energy = np.tile(np.array([0.6, 1.4, 2.2, 3.1]), 3)
    points = _spin_fluctuation_points(np.ones(q.size), q, energy, temperature=10.0)
    points.metadata.update(
        {"data_type": "powder_inelastic", "coordinate_units": "1/angstrom",
         "powder_q_modulus_axis": True}
    )

    evaluator = _RpaComponentEvaluator(component)
    geometry, _form_factor, _tensor, count = evaluator._geometry(points)

    # Reference: orient every fitted point, with no |Q| deduplication.
    directions = _powder_sphere_directions(count)
    reciprocal = _powder_reciprocal_matrix(points, evaluator._lattice)
    modulus = np.asarray(q_modulus_inv_angstrom(points), dtype=float)
    hkl = (modulus[:, None, None] * directions[None, :, :]).reshape(-1, 3) @ np.linalg.inv(
        reciprocal
    ).T
    reference = build_rpa_geometry(
        hkl[:, 0], hkl[:, 1], hkl[:, 2], evaluator.site_positions, evaluator.orbits
    )

    assert geometry.n_q == reference.n_q
    assert np.array_equal(geometry.unique_hkl, reference.unique_hkl)
    assert np.array_equal(geometry.point_index, reference.point_index)
    for label, phases in reference.bond_phases.items():
        assert np.array_equal(geometry.bond_phases[label], phases)
