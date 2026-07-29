from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib import pyplot as plt

from nfit import (
    BasisState,
    DataGroup,
    ElectronicModel,
    ModelComponentSpec,
    NfitProject,
    WavevectorSampling,
    band_path,
    build_electronic_model,
    calculate_bands,
    create_model_component,
    density_of_states,
    fermi_surface,
    import_wannier90,
    k_mesh,
    load_electronic_model,
    load_project,
    model_definition,
    save_electronic_model,
    save_project,
)


def _chain_model(*, hopping=-100.0, onsite=0.0):
    return build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=[BasisState("s", site="A", orbital="s")],
        hoppings={
            (0, 0, 0): np.array([[onsite]]),
            (1, 0, 0): np.array([[hopping]]),
        },
        orbital_centers=[[0.25, 0.0, 0.0]],
        periodic_axes=(0,),
    )


def _square_two_orbital_model():
    onsite = np.diag([-25.0, 25.0])
    hopping = np.diag([-80.0, -40.0])
    return build_electronic_model(
        direct_lattice=np.diag([4.0, 4.0, 12.0]),
        basis=[
            BasisState("d", site="M", orbital="d"),
            BasisState("p", site="X", orbital="p"),
        ],
        hoppings={
            (0, 0, 0): onsite,
            (1, 0, 0): hopping,
            (0, 1, 0): hopping,
        },
        orbital_centers=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        periodic_axes=(0, 1),
    )


def test_arbitrary_chain_uses_wannier_phase_and_physical_path_distance():
    model = _chain_model(hopping=-100.0, onsite=12.0)
    wavevectors = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]
    )
    expected = 12.0 - 200.0 * np.cos(2.0 * np.pi * wavevectors[:, 0])

    np.testing.assert_allclose(
        np.linalg.eigvalsh(model.hamiltonian(wavevectors))[:, 0], expected
    )
    np.testing.assert_allclose(
        model.hamiltonian(wavevectors + [1.0, 0.0, 0.0]),
        model.hamiltonian(wavevectors),
    )

    path = band_path(
        model,
        [[0.0], [0.5]],
        labels=[r"$\Gamma$", "X"],
        points_per_segment=4,
    )
    result = calculate_bands(model, path)
    assert path.labels == ((0, r"$\Gamma$"), (4, "X"))
    assert path.path_distance_inv_angstrom[-1] == pytest.approx(np.pi / 2.0)
    np.testing.assert_allclose(
        result.energies_meV[:, 0],
        12.0 - 200.0 * np.cos(2.0 * np.pi * path.reduced_coordinates[:, 0]),
    )
    restored_path = WavevectorSampling.from_dict(path.to_dict())
    np.testing.assert_allclose(
        restored_path.path_distance_inv_angstrom,
        path.path_distance_inv_angstrom,
    )


def test_projected_bands_and_dos_preserve_state_count():
    model = _square_two_orbital_model()
    mesh = k_mesh(model, (50, 50))
    bands = calculate_bands(
        model,
        mesh,
        projections={"d": [0], "p": [1], "all": [0, 1]},
    )
    np.testing.assert_allclose(
        bands.projected_weights["d"] + bands.projected_weights["p"], 1.0
    )
    np.testing.assert_allclose(bands.projected_weights["all"], 1.0)

    energy = np.linspace(-500.0, 500.0, 2001)
    dos = density_of_states(
        model,
        mesh,
        energy,
        broadening_meV=5.0,
        projections={"d": [0], "p": [1]},
        max_chunk_bytes=1_000_000,
    )
    assert np.trapezoid(dos.total_per_meV_cell, energy) == pytest.approx(
        2.0, rel=2.0e-3
    )
    np.testing.assert_allclose(
        dos.projected_per_meV_cell["d"] + dos.projected_per_meV_cell["p"],
        dos.total_per_meV_cell,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_basis_permutation_and_orbital_origin_do_not_change_bands():
    model = _square_two_orbital_model()
    permutation = np.array([1, 0])
    blocks = {
        tuple(vector): block[np.ix_(permutation, permutation)]
        for vector, block in zip(
            model.translations, model.hamiltonian_blocks, strict=True
        )
        if tuple(vector) >= (0, 0, 0)
    }
    permuted = build_electronic_model(
        direct_lattice=model.direct_lattice,
        basis=[model.basis[index] for index in permutation],
        hoppings=blocks,
        interpolation_weights={
            tuple(vector): weight
            for vector, weight in zip(
                model.translations, model.interpolation_weights, strict=True
            )
        },
        orbital_centers=model.orbital_centers[permutation] + [0.2, -0.1, 0.0],
        periodic_axes=model.periodic_axes,
    )
    wavevectors = np.array([[0.13, 0.27, 0.0], [0.4, 0.1, 0.0]])

    np.testing.assert_allclose(
        np.linalg.eigvalsh(permuted.hamiltonian(wavevectors)),
        np.linalg.eigvalsh(model.hamiltonian(wavevectors)),
    )


def test_named_hopping_parameters_return_new_models():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["s"],
        hoppings={(0, 0, 0): [[5.0]]},
        periodic_axes=(0,),
        parameter_values={"t": -10.0},
        parameter_hoppings={"t": {(1, 0, 0): [[1.0]]}},
    )
    changed = model.with_parameters(t=-20.0)

    assert model.parameter_values["t"] == -10.0
    assert changed.parameter_values["t"] == -20.0
    assert np.linalg.eigvalsh(model.hamiltonian([0.0, 0.0, 0.0]))[0] == -15.0
    assert np.linalg.eigvalsh(changed.hamiltonian([0.0, 0.0, 0.0]))[0] == -35.0
    assert changed.content_digest != model.content_digest


def test_fermi_surfaces_cover_one_and_two_dimensions():
    chain = fermi_surface(_chain_model(), (200,), target_energy_meV=0.0)
    points = np.sort(chain.sheets[0].vertices_reduced[:, 0])
    np.testing.assert_allclose(points, [0.25, 0.75], atol=1.0e-12)
    assert chain.sheets[0].vertices_inv_angstrom.shape == (2, 3)

    square = fermi_surface(
        _square_two_orbital_model(),
        (50, 50),
        target_energy_meV=0.0,
        projections={"d": [0], "p": [1]},
    )
    assert square.dimension == 2
    assert square.sheets
    assert all(sheet.connectivity.shape[1] == 2 for sheet in square.sheets)
    assert all(
        sheet.vertices_reduced.shape == sheet.vertices_inv_angstrom.shape
        for sheet in square.sheets
    )


def test_three_dimensional_fermi_surface_is_triangulated():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["s"],
        hoppings={
            (0, 0, 0): [[0.0]],
            (1, 0, 0): [[-1.0]],
            (0, 1, 0): [[-1.0]],
            (0, 0, 1): [[-1.0]],
        },
    )
    result = fermi_surface(model, (10, 10, 10), target_energy_meV=0.0)

    assert len(result.sheets) == 1
    assert result.sheets[0].vertices_reduced.shape[1] == 3
    assert result.sheets[0].connectivity.shape[1] == 3


def test_canonical_model_round_trip_is_digest_protected_and_immutable(tmp_path):
    model = _square_two_orbital_model()
    path = tmp_path / "model.json"
    save_electronic_model(model, path)
    restored = load_electronic_model(path)

    assert restored.content_digest == model.content_digest
    np.testing.assert_allclose(
        restored.hamiltonian([[0.13, 0.27, 0.0]]),
        model.hamiltonian([[0.13, 0.27, 0.0]]),
    )
    with pytest.raises(ValueError):
        restored.hamiltonian_blocks[0, 0, 0] = 1.0

    payload = restored.to_dict()
    payload["energy_zero_meV"] = 1.0
    with pytest.raises(ValueError, match="digest"):
        ElectronicModel.from_dict(payload)


def _write_hr_fixture(tmp_path):
    hr = tmp_path / "chain_hr.dat"
    hr.write_text(
        "\n".join(
            [
                "created for nfit test",
                "1",
                "3",
                "1 1 1",
                "-1 0 0 1 1 -0.100000 0.0",
                "0 0 0 1 1 0.012000 0.0",
                "1 0 0 1 1 -0.100000 0.0",
                "",
            ]
        )
    )
    (tmp_path / "chain.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "2.0 0.0 0.0",
                "0.0 8.0 0.0",
                "0.0 0.0 9.0",
                "end unit_cell_cart",
                "",
            ]
        )
    )
    (tmp_path / "chain_centres.xyz").write_text(
        "1\nWannier centres\nX 0.5 0.0 0.0\n"
    )
    return hr


def test_wannier90_hr_import_converts_units_centres_and_provenance(tmp_path):
    hr = _write_hr_fixture(tmp_path)
    model = import_wannier90(hr, periodic_axes=(0,))

    assert model.basis[0].label == "w1"
    np.testing.assert_allclose(model.orbital_centers[0], [0.25, 0.0, 0.0])
    energies = np.linalg.eigvalsh(
        model.hamiltonian([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    )[:, 0]
    np.testing.assert_allclose(energies, [-188.0, 212.0])
    assert model.provenance["energy_conversion"] == "eV to meV (x1000)"
    assert {item["role"] for item in model.provenance["files"]} == {
        "wannier90_hr",
        "wannier90_input",
        "wannier_centres",
    }


def test_wannier90_tb_and_wsvec_import(tmp_path):
    tb = tmp_path / "one_tb.dat"
    tb.write_text(
        "\n".join(
            [
                "created for nfit test",
                "2.0 0.0 0.0",
                "0.0 8.0 0.0",
                "0.0 0.0 9.0",
                "1",
                "3",
                "1 1 1",
                "-1 0 0 1 1 -0.100000 0.0",
                "0 0 0 1 1 0.012000 0.0",
                "1 0 0 1 1 -0.100000 0.0",
                "-1 0 0",
                "1 1 0.5 0.0 0.0 0.0 0.0 0.0",
                "0 0 0",
                "1 1 0.5 0.0 0.0 0.0 0.0 0.0",
                "1 0 0",
                "1 1 0.5 0.0 0.0 0.0 0.0 0.0",
                "",
            ]
        )
    )
    direct = import_wannier90(tb, periodic_axes=(0,))
    np.testing.assert_allclose(direct.orbital_centers[0], [0.25, 0.0, 0.0])

    hr = _write_hr_fixture(tmp_path)
    wsvec = tmp_path / "chain_wsvec.dat"
    wsvec.write_text(
        "\n".join(
            [
                "written with use_ws_distance=.true.",
                "-1 0 0 1 1",
                "1",
                "1 0 0",
                "1 0 0 1 1",
                "1",
                "-1 0 0",
                "",
            ]
        )
    )
    corrected = import_wannier90(hr, periodic_axes=(0,))
    np.testing.assert_allclose(
        np.linalg.eigvalsh(
            corrected.hamiltonian([[0.0, 0.0, 0.0], [0.37, 0.0, 0.0]])
        )[:, 0],
        [-188.0, -188.0],
    )
    assert corrected.provenance["wsvec_applied"] is True


def test_tight_binding_registry_plots_and_scripts_are_component_driven():
    model = _chain_model()
    component = ModelComponentSpec(
        name="bands",
        type="tight_binding",
        config={
            **{
                field.name: field.default
                for field in model_definition("tight_binding").config_fields
            },
            "model_data": model.to_dict(),
            "periodic_axes": [0],
            "band_path": [
                {"label": "G", "k": [0.0, 0.0, 0.0]},
                {"label": "X", "k": [0.5, 0.0, 0.0]},
            ],
            "dos_mesh": [60],
            "fermi_mesh": [100],
        },
    )
    definition = model_definition("tight_binding")
    assert definition.data_types == ("electronic_structure",)
    assert [plot.key for plot in definition.plots] == [
        "bands",
        "dos",
        "fermi_surface",
    ]
    for plot in definition.plots:
        result = plot.calculate(component)
        assert result.model_digest == model.content_digest
        figure, _axes = plot.render(result)
        assert figure.axes
        plt.close(figure)
        script = plot.script(component)
        namespace = {}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="FigureCanvasAgg is non-interactive",
                category=UserWarning,
            )
            exec(compile(script, f"<tight-binding-{plot.key}>", "exec"), namespace)
        assert namespace["result"].model_digest == result.model_digest
        plt.close(namespace["figure"])
        assert "model = ElectronicModel.from_dict" in script


def test_manual_electronic_model_reopens_from_a_project(tmp_path):
    model = _chain_model()
    group = DataGroup("Electronic")
    component = create_model_component(group, "chain", type="tight_binding")
    component.config["model_data"] = model.to_dict()
    component.config["periodic_axes"] = [0]
    path = tmp_path / "electronic.nfit"

    save_project(NfitProject([group]), path)
    restored = load_project(path)
    restored_component = restored.data_groups[0].models["chain"]
    result = model_definition("tight_binding").plots[0].calculate(
        restored_component
    )

    assert result.model_digest == model.content_digest
