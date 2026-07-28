# ruff: noqa: F401
from __future__ import annotations

import numpy as np
import pytest

from nfit.dataset import PointData4D, PointListData
from nfit.fit_config import (
    FitDatasetInput,
    compile_fit_problem,
    component_parameter_names,
    dataset_scale_parameter_name,
    instanced_parameter_name,
    model_supports_data_type,
    qualified_parameter_name,
)
from nfit.fitting import fit_problem_least_squares
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup, ModelComponentSpec
from nfit.plotting import MDHistoSliceViewer
from nfit.project_gui import (
    NfitProject,
    attach_fit_channels_to_view,
    create_mask,
    create_model_component,
    current_model_channels,
    dataset_for_slice_viewer,
    ensure_fit_history,
    fit_data_bundle,
    latest_fit_channels,
    load_project,
    run_group_fit,
    save_project,
    slice_viewer_datasets,
)

RNG = np.random.default_rng(7)


def _points(level: float, n: int = 64) -> PointData4D:
    return PointData4D(
        H=np.zeros(n),
        K=np.zeros(n),
        L=np.zeros(n),
        E=np.linspace(0.0, 10.0, n),
        intensity=level + RNG.normal(0.0, 0.01, n),
        sigma=np.full(n, 0.01),
    )


def _constant_component(**overrides) -> ModelComponentSpec:
    spec = ModelComponentSpec(
        name="bg",
        type="constant_background",
        parameters={"constant": 0.5},
        fit_parameters={"constant": True},
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def _grid_mdhisto(signal: np.ndarray) -> MDHistoData:
    ny, nx = signal.shape
    axis_h = MDHistoAxis("H", np.linspace(0.0, 1.0, ny + 1), "rlu", "momentum")
    axis_e = MDHistoAxis("E", np.linspace(0.0, 10.0, nx + 1), "meV", "energy")
    return MDHistoData(
        axes=(axis_h, axis_e),
        signal=signal,
        errors=np.full_like(signal, 0.1),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={},
    )


def _fit_ready_group(levels: dict[str, float]) -> DataGroup:
    group = DataGroup("g")
    for name, level in levels.items():
        signal = np.full((4, 5), level)
        group.datasets.append(DatasetEntry(name, _grid_mdhisto(signal)))
    return group


def _rlu_matrix(a: float) -> list[list[float]]:
    return (2.0 * np.pi / a * np.eye(3)).tolist()


def _spin_fluctuation_points(
    intensity: np.ndarray,
    H: np.ndarray,
    E: np.ndarray,
    *,
    temperature: float | None,
    lattice_a: float | None = None,
) -> PointData4D:
    metadata = {}
    if lattice_a is not None:
        metadata["rlu_to_inv_angstrom_matrix"] = _rlu_matrix(lattice_a)
    return PointData4D(
        H=H,
        K=np.zeros_like(H),
        L=np.zeros_like(H),
        E=E,
        intensity=intensity,
        sigma=np.full(H.shape, 0.01),
        temperature=temperature,
        metadata=metadata,
    )


def _heisenberg_chain_component(**overrides) -> ModelComponentSpec:
    spec = ModelComponentSpec(
        name="rpa",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.5, "gamma0": 2.5, "J1": 0.05},
        fit_parameters={"chi0": True, "gamma0": True, "J1": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}],
        },
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def _rpa_component(**overrides) -> ModelComponentSpec:
    spec = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.2, "chi0": 0.3, "gamma0": 2.0, "J1": 0.1, "J2": -0.05},
        fit_parameters={"scale": True, "chi0": True, "gamma0": True, "J1": True, "J2": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11]],
            "orbits": [
                {"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]},
                {"label": "J2", "bonds": [{"site_i": 0, "site_j": 0, "offset": [0, 0, 1]}]},
            ],
        },
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def _rpa_points(temperature: float, seed: int, n: int = 60) -> PointData4D:
    rng = np.random.default_rng(seed)
    return PointData4D(
        H=rng.uniform(-1.0, 1.0, n),
        K=rng.uniform(-1.0, 1.0, n),
        L=rng.uniform(-1.0, 1.0, n),
        E=rng.uniform(0.5, 5.0, n),
        intensity=rng.uniform(0.0, 1.0, n),
        sigma=np.full(n, 0.1),
        temperature=float(temperature),
        metadata={"coordinate_units": "r.l.u.", "energy_units": "meV"},
    )


def _pyrochlore_tensor_component(fit_aniso=True):
    from nfit.crystal import (
        generate_bond_orbits,
        orbits_to_config,
        sites_to_config,
        symmetry_allowed_exchange_basis,
    )

    crystal = {
        "lattice": {"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "spacegroup": "F d -3 m:2",
        "sites": [{"label": "M1", "position": [0.0, 0.0, 0.0], "ion": "V2"}],
    }
    nn = 10.0 * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(crystal, ["M1"], cutoff_angstrom=nn + 0.01)
    basis = symmetry_allowed_exchange_basis(crystal, sites, orbits[0])
    return crystal, ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={
            "scale": 1.0,
            "chi0": 0.02,
            "gamma0": 3.0,
            "J1": 0.05,
            "J1_S1": 0.0,
            "J1_S2": 0.0,
            "J1_D1": 0.0,
        },
        fit_parameters={"scale": True, "chi0": True, "J1": True, "J1_S1": fit_aniso},
        config={
            "site_positions": sites_to_config(sites),
            "orbits": orbits_to_config(orbits),
            "crystal": crystal,
            "ion": "V2",
            "anisotropy": {"J1": {"enabled": True, "basis": basis}},
        },
    )


def _tensor_points(seed, n=200):
    from nfit.fitting import reciprocal_basis_from_lattice_parameters

    rng = np.random.default_rng(seed)
    matrix = reciprocal_basis_from_lattice_parameters(10, 10, 10, 90, 90, 90)
    return PointData4D(
        H=rng.uniform(-1.0, 1.0, n),
        K=rng.uniform(-1.0, 1.0, n),
        L=rng.uniform(-1.0, 1.0, n),
        E=rng.uniform(0.5, 5.0, n),
        intensity=np.zeros(n),
        sigma=np.full(n, 0.05),
        temperature=5.0,
        metadata={
            "coordinate_units": "r.l.u.",
            "energy_units": "meV",
            "rlu_to_inv_angstrom_matrix": matrix.tolist(),
        },
    )


def _closure_points(seed, n=120):
    rng = np.random.default_rng(seed)
    return PointData4D(
        H=rng.uniform(-1.0, 1.0, n),
        K=rng.uniform(-1.0, 1.0, n),
        L=rng.uniform(-1.0, 1.0, n),
        E=rng.uniform(0.5, 5.0, n),
        intensity=np.zeros(n),
        sigma=np.full(n, 0.05),
        temperature=10.0,
    )


def _closure_scalar_component(closure=None, **params):
    values = {"scale": 1.0, "chi0": 0.5, "gamma0": 2.0, "J1": 0.1}
    values.update(params)
    config = {
        "site_positions": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        "orbits": [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]}],
    }
    if closure is not None:
        config["closure"] = closure
    return ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters=values,
        fit_parameters={},
        config=config,
    )


def _evaluate(component, points):
    from nfit.fitting import evaluate_problem_model

    compiled = compile_fit_problem(
        [component], [FitDatasetInput("d", points, data_type="single_crystal_inelastic")]
    )
    return evaluate_problem_model(
        compiled.problem,
        "d",
        {spec.name: spec.value for spec in compiled.problem.parameter_specs},
    )


def _magnetization_points(temperatures, fields_tesla, moment, sigma=0.01):
    """Build a magnetization PointData4D (H=K=L=E=0, per-point T and B_z)."""
    temps = np.asarray(temperatures, dtype=float)
    fields = np.asarray(fields_tesla, dtype=float)
    n = temps.size
    field_vecs = np.zeros((n, 3))
    field_vecs[:, 2] = fields
    return PointData4D(
        H=np.zeros(n),
        K=np.zeros(n),
        L=np.zeros(n),
        E=np.zeros(n),
        intensity=np.asarray(moment, dtype=float),
        sigma=np.full(n, sigma),
        temperature=temps,
        magnetic_field=field_vecs,
        metadata={"data_type": "magnetization"},
    )


def _magnetization_component(closure=None, **params):
    values = {"scale": 1.0, "chi0": 0.5, "gamma0": 2.0, "J1": 0.0}
    values.update(params)
    config = {
        "site_positions": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        "orbits": [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]}],
    }
    if closure is not None:
        config["closure"] = closure
    return ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters=values,
        fit_parameters={},
        config=config,
    )
