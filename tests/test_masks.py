import numpy as np
import pytest

from nfit import (
    PointData4D,
    attach_lattice_parameters,
    mask_out_box,
    mask_out_ellipsoid,
    mask_out_energy_q_range,
    mask_out_phonon_cone,
    q_modulus_inv_angstrom,
)


def _point_data(H, K, L, E):
    H = np.asarray(H, dtype=float)
    return PointData4D(
        H=H,
        K=K,
        L=L,
        E=E,
        intensity=np.ones_like(H),
        sigma=np.ones_like(H),
    )


def test_lattice_parameters_enable_q_modulus_masks():
    data = _point_data([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 5.0, 10.0])
    data = attach_lattice_parameters(data, a=2.0, b=2.0, c=2.0)

    np.testing.assert_allclose(q_modulus_inv_angstrom(data), [0.0, np.pi, 2.0 * np.pi])

    masked = mask_out_energy_q_range(data, energy=(4.0, 6.0), q_modulus=(2.0, 4.0))

    np.testing.assert_array_equal(masked.mask, [True, False, True])


def test_q_modulus_masks_require_conversion_metadata():
    data = _point_data([1.0], [0.0], [0.0], [5.0])

    with pytest.raises(ValueError, match="attached lattice/UB matrix"):
        mask_out_energy_q_range(data, q_modulus=(0.0, 2.0))


def test_q_modulus_can_use_direct_inverse_angstrom_coordinates():
    data = _point_data([1.0], [2.0], [2.0], [5.0])
    data.metadata["coordinate_units"] = "inverse angstroms"

    np.testing.assert_allclose(q_modulus_inv_angstrom(data), [3.0])


def test_projected_box_masks_any_other_dimensions():
    data = _point_data(
        H=[0.5, 0.5, 2.0],
        K=[0.5, 0.5, 0.0],
        L=[0.0, 10.0, 0.0],
        E=[5.0, 5.0, 5.0],
    )

    masked = mask_out_box(
        data,
        dimensions=[(1.0, 1.0, 0.0), "E"],
        center=[1.0, 5.0],
        half_widths=[0.1, 0.5],
    )

    np.testing.assert_array_equal(masked.mask, [False, False, True])


def test_projected_ellipsoid_masks_inside_region():
    data = _point_data(
        H=[0.0, 0.5, 2.0],
        K=[0.0, 0.0, 0.0],
        L=[0.0, 0.0, 0.0],
        E=[0.0, 2.0, 0.0],
    )

    masked = mask_out_ellipsoid(
        data,
        dimensions=["H", "E"],
        center=[0.0, 0.0],
        radii=[1.0, 1.0],
    )

    np.testing.assert_array_equal(masked.mask, [False, True, True])


def test_phonon_cone_masks_q_sphere_growing_with_energy():
    data = _point_data(
        H=[0.4, 0.6, 1.0, 0.4],
        K=[0.0, 0.0, 0.0, 0.0],
        L=[0.0, 0.0, 0.0, 0.0],
        E=[5.0, 5.0, 5.0, 12.0],
    )
    data = attach_lattice_parameters(data, a=2.0 * np.pi, b=2.0 * np.pi, c=2.0 * np.pi)

    masked = mask_out_phonon_cone(
        data,
        center=[0.0, 0.0, 0.0],
        slope=10.0,
        energy_range=(0.0, 6.0),
    )

    np.testing.assert_array_equal(masked.mask, [False, True, True, True])


def test_phonon_cone_accepts_multiple_bragg_centers():
    data = _point_data(
        H=[0.0, 1.0, 2.0],
        K=[0.0, 0.0, 0.0],
        L=[0.0, 0.0, 0.0],
        E=[0.0, 0.0, 0.0],
    )
    data = attach_lattice_parameters(data, a=2.0 * np.pi, b=2.0 * np.pi, c=2.0 * np.pi)

    masked = mask_out_phonon_cone(
        data,
        center=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        slope=10.0,
        radius_offset=0.1,
    )

    np.testing.assert_array_equal(masked.mask, [False, True, False])


def test_phonon_cone_rejects_malformed_center_lists():
    data = _point_data([0.0], [0.0], [0.0], [0.0])

    with pytest.raises(ValueError, match="one 3-vector or a nonempty list"):
        mask_out_phonon_cone(data, center=[[0.0, 0.0]], slope=10.0)
