import numpy as np
import pytest

from nfit import project_gui
from nfit.dataset import PointData4D, PointListData
from nfit.rebin import rebin_nd_symmetry
from nfit.symmetry import SymmetrySpec, resolve_symmetry, transform_hkl

pytest.importorskip("gemmi")


def test_space_group_uses_only_its_point_group_operations():
    operations = resolve_symmetry(SymmetrySpec("space_group", "P -1"))
    assert len(operations) == 2
    np.testing.assert_allclose(operations[0].matrix_hkl, np.eye(3))
    np.testing.assert_allclose(operations[1].matrix_hkl, -np.eye(3))
    assert len(resolve_symmetry(SymmetrySpec("space_group", "P 1"))) == 1


def test_explicit_jones_faithful_operations_transform_hkl_on_the_dual_space():
    operations = resolve_symmetry(SymmetrySpec("operations", "x-y,x,z+1/2"))
    transformed = list(transform_hkl(np.array([[1.0, 2.0, 3.0]]), operations))[0]
    # x' = x-y, y' = x acts on Miller indices as (h, k, l) -> (-k, h+k, l).
    np.testing.assert_allclose(transformed, [[-2.0, 3.0, 3.0]])


def test_geometric_rotation_and_mirror_generators_close_the_group():
    lattice = {"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    rotation = resolve_symmetry(
        SymmetrySpec("generators", "rotate(order=3, axis=[1,1,1])"), lattice_parameters=lattice
    )
    mirror = resolve_symmetry(
        SymmetrySpec("generators", "mirror(plane=(0,0,1))"), lattice_parameters=lattice
    )
    assert len(rotation) == 3
    assert len(mirror) == 2
    np.testing.assert_allclose(mirror[1].matrix_hkl, np.diag([1.0, 1.0, -1.0]), atol=1e-12)


def test_symmetry_rebin_streams_inversion_images():
    operations = resolve_symmetry(SymmetrySpec("space_group", "P -1"))
    result = rebin_nd_symmetry(
        [5.0],
        [[0.5, 0.0, 0.0, 1.0]],
        [operation.matrix_hkl for operation in operations],
        data_errs=[1.0],
        lower=[-1.0, -0.5, -0.5, 0.0],
        upper=[1.0, 0.5, 0.5, 2.0],
        num_bins=[2, 1, 1, 1],
        fractional=False,
    )
    np.testing.assert_allclose(result.binned_data[:, 0, 0, 0], [5.0, 5.0])
    np.testing.assert_allclose(result.binned_data_errs[:, 0, 0, 0], [1.0, 1.0])


def test_symmetry_rebin_does_not_duplicate_a_fixed_point():
    operations = resolve_symmetry(SymmetrySpec("space_group", "P -1"))
    result = rebin_nd_symmetry(
        [5.0],
        [[0.0, 0.0, 0.0]],
        [operation.matrix_hkl for operation in operations],
        data_errs=[2.0],
        lower=[-0.5, -0.5, -0.5],
        upper=[0.5, 0.5, 0.5],
        num_bins=[1, 1, 1],
        fractional=False,
    )
    np.testing.assert_allclose(result.binned_data, [[[5.0]]])
    np.testing.assert_allclose(result.binned_data_errs, [[[2.0]]])


def test_project_rebin_configuration_applies_symmetry_before_binning():
    data = PointData4D([0.5], [0.0], [0.0], [1.0], [7.0], [1.0])
    config = {
        "fractional": False,
        "resolution_mode": "bins",
        "axes": project_gui._default_rebin_axes(data),
        "symmetry": {"mode": "space_group", "expression": "P -1"},
    }
    config["axes"][0].update({"lower": -1.0, "upper": 1.0, "num_bins": 2})
    config["axes"][1].update({"lower": -0.5, "upper": 0.5, "num_bins": 1})
    config["axes"][2].update({"lower": -0.5, "upper": 0.5, "num_bins": 1})
    config["axes"][3].update({"lower": 0.0, "upper": 2.0, "num_bins": 1})
    for axis in config["axes"]:
        axis.update({"auto_lower": False, "auto_upper": False})
    output = project_gui._rebin_point_data(data, config)
    assert isinstance(output, project_gui.MDHistoData)
    np.testing.assert_allclose(output.axes[0].centers, [-0.5, 0.5])
    np.testing.assert_allclose(output.signal[:, 0, 0, 0], [7.0, 7.0])
    assert output.metadata["rebin"]["symmetry"]["operation_count"] == 2


def test_point_list_rebin_supports_hkl_symmetry():
    data = PointListData(
        {"H": [0.5], "K": [0.0], "L": [0.0], "I": [4.0], "dI": [1.0]},
        coordinate_names=["H", "K", "L"],
        channels=[{"label": "Intensity", "value": "I", "error": "dI"}],
    )
    operations = resolve_symmetry(SymmetrySpec("space_group", "P -1"))
    output = data.rebin_to_histogram(
        ["H", "K", "L"],
        lower=[-1.0, -0.5, -0.5],
        upper=[1.0, 0.5, 0.5],
        num_bins=[2, 1, 1],
        fractional=False,
        symmetry_operations=[operation.matrix_hkl for operation in operations],
    )
    np.testing.assert_allclose(np.sort(output.column("H")), [-0.5, 0.5])


def _operation_matrices(point_group: str) -> set[bytes]:
    return {
        np.round(operation.matrix_hkl, 6).tobytes()
        for operation in resolve_symmetry(SymmetrySpec("point_group", point_group))
    }


def _with_inversion(point_group: str) -> set[bytes]:
    matrices: set[bytes] = set()
    for operation in resolve_symmetry(SymmetrySpec("point_group", point_group)):
        matrices.add(np.round(operation.matrix_hkl, 6).tobytes())
        matrices.add(np.round(-operation.matrix_hkl, 6).tobytes())
    return matrices


@pytest.mark.parametrize(
    ("noncentrosymmetric", "laue"),
    [
        ("3m1", "-3m1"),
        ("31m", "-31m"),
        ("321", "-3m1"),
        ("312", "-31m"),
    ],
)
def test_trigonal_laue_groups_extend_their_own_setting(noncentrosymmetric, laue):
    """The secondary/tertiary setting must survive adding inversion.

    ``-3m1`` and ``-31m`` share the abbreviated Hermann-Mauguin symbol ``-3m``,
    so they are resolved through an explicit space-group alias. Getting that
    alias backwards silently folds trigonal data with mirrors along the wrong
    in-plane directions, which no other test would catch.
    """

    assert _with_inversion(noncentrosymmetric) == _operation_matrices(laue)


def test_trigonal_laue_settings_are_distinct():
    assert _operation_matrices("-3m1") != _operation_matrices("-31m")
    assert _operation_matrices("-6m2") != _operation_matrices("-62m")
    assert _operation_matrices("3m1") != _operation_matrices("31m")
