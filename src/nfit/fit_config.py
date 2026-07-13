"""Compile declarative model-component configuration into a FitProblem.

This module is the bridge between the GUI-facing, serializable layer
(:class:`~nfit.pipeline.ModelComponentSpec` with sharing policies, bounds,
dataset selectors, and inequality constraints) and the deliberately dumb
optimizer layer in :mod:`nfit.fitting` (a flat list of named scalars with
box bounds).

Concepts
--------
Qualified parameter names
    Every model-component parameter has a qualified name
    ``"<component>.<parameter>"`` (for example ``"bg.constant"``). Model
    evaluators read their parameters from the trial dictionary under these
    qualified names.

Sharing policies
    A parameter is *instantiated* according to its sharing mode. ``"global"``
    emits one optimizer parameter named after the qualified name.
    ``"per_dataset"`` emits one instance per applicable dataset, named
    ``"<component>.<parameter>[<dataset>]"``. ``"grouped"`` emits one instance
    per distinct tie key, named ``"<component>.<parameter>[<key>]"``; datasets
    not listed in the group map get their own private instance. Per-dataset
    parameter bindings map the qualified name each evaluator reads back to the
    right instance.

Constraints
    ``{"parameter": p, "op": ">=", "reference": r}`` reparameterizes the
    target as ``p = r + delta`` with ``delta >= 0`` (``"<="`` uses ``-``), so
    the constraint is hard: the optimizer cannot violate it. ``r`` is a number
    or the qualified name of another global parameter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cross_section import intensity_from_chipp
from .dataset import PointData4D
from .fitting import (
    DerivedParameter,
    FitDataset,
    FitProblem,
    ModelFunction,
    ParameterBinding,
    ParameterSpec,
    _metadata_coordinate_units_are_inv_angstrom,
    _resolve_q_transform,
)
from .form_factors import form_factor_sq
from .models import paramagnon_chipp
from .spin_fluctuations import (
    build_rpa_geometry,
    heisenberg_rpa_chipp,
    heisenberg_rpa_chipp_and_gradients,
    local_relaxational_chipp,
    mmp_chipp,
    reduce_site_network,
    reduce_site_network_with_tensors,
    rpa_exchange_matrix,
)

FALLBACK_DATA_TYPE = "single_crystal_inelastic"

SHARING_MODES = ("global", "per_dataset", "grouped")

CONSTRAINT_OFFSET_SUFFIX = "__offset"

# A model Jacobian returns d(model intensity)/d(parameter) for every parameter
# the model reads, keyed by the qualified name it reads that parameter under
# (the same names the model's ``params`` lookups use). Optimizer-variable and
# constraint bookkeeping is handled by the fit assembler, not here.
ModelJacobian = Callable[[PointData4D, dict[str, float]], dict[str, np.ndarray]]


def qualified_parameter_name(component_name: str, parameter: str) -> str:
    """Return the qualified name a model evaluator reads a parameter under."""

    return f"{component_name}.{parameter}"


def instanced_parameter_name(component_name: str, parameter: str, key: str) -> str:
    """Return the optimizer name of one instance of a shared parameter."""

    return f"{component_name}.{parameter}[{key}]"


def _constant_background_factory(component: Any) -> ModelFunction:
    key = qualified_parameter_name(component.name, "constant")

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.full(data.size, float(params[key]), dtype=float)

    return model


def _constant_background_jacobian_factory(component: Any) -> ModelJacobian:
    key = qualified_parameter_name(component.name, "constant")

    def jacobian(data: PointData4D, params: dict[str, float]) -> dict[str, np.ndarray]:
        return {key: np.ones(data.size, dtype=float)}

    return jacobian


def _linear_background_factory(component: Any) -> ModelFunction:
    c0_key = qualified_parameter_name(component.name, "c0")
    c1_key = qualified_parameter_name(component.name, "c1")

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return float(params[c0_key]) + float(params[c1_key]) * np.asarray(data.E, dtype=float)

    return model


def _linear_background_jacobian_factory(component: Any) -> ModelJacobian:
    c0_key = qualified_parameter_name(component.name, "c0")
    c1_key = qualified_parameter_name(component.name, "c1")

    def jacobian(data: PointData4D, params: dict[str, float]) -> dict[str, np.ndarray]:
        return {
            c0_key: np.ones(data.size, dtype=float),
            c1_key: np.asarray(data.E, dtype=float).copy(),
        }

    return jacobian


def _single_q_paramagnon_factory(component: Any) -> ModelFunction:
    name = component.name

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return paramagnon_chipp(
            data.H,
            data.K,
            data.L,
            data.E,
            amplitude=float(params[qualified_parameter_name(name, "amplitude")]),
            q0=(
                float(params[qualified_parameter_name(name, "q0_h")]),
                float(params[qualified_parameter_name(name, "q0_k")]),
                float(params[qualified_parameter_name(name, "q0_l")]),
            ),
            kappa=float(params[qualified_parameter_name(name, "kappa")]),
            omega_sf=float(params[qualified_parameter_name(name, "omega_sf")]),
        )

    return model


ISOTROPIC_POLARIZATION = 2.0 / 3.0
"""Polarization factor for isotropic (Heisenberg) spins.

Unpolarized neutrons couple only to spin components perpendicular to Q; for an
isotropic spin system the orientation factor averages to ``2/3``.
"""


def _dataset_temperature(data: PointData4D) -> float | np.ndarray:
    """Return the sample temperature of a fitted dataset, validating it."""

    temperature = data.temperature
    if temperature is None or np.any(~np.isfinite(np.asarray(temperature, dtype=float))):
        raise ValueError(
            "this model requires a valid sample temperature for every fitted "
            "dataset: set it in the dataset details (Temperature) or import "
            "data that carries temperature metadata"
        )
    return temperature


def _dataset_magnetic_field(data: PointData4D) -> np.ndarray:
    """Return the applied field of a fitted dataset, validating it.

    Zeeman-enabled models require a Cartesian Tesla 3-vector on every fitted
    dataset (stamped from the dataset's Sample environment settings).
    """

    field_vector = data.magnetic_field
    if field_vector is None or not np.all(np.isfinite(np.asarray(field_vector, dtype=float))):
        raise ValueError(
            "this model has the Zeeman term enabled and requires a valid "
            "applied magnetic field for every fitted dataset: set the field "
            "magnitude and direction in the dataset details (Sample "
            "environment), and make sure the data group has lattice "
            "parameters to orient the direction"
        )
    return np.asarray(field_vector, dtype=float)


def _form_factor_sq_from_config(component: Any, data: PointData4D) -> float | np.ndarray:
    """Return ``|f(Q)|^2`` from a component's ``ion`` / coefficient config.

    Returns 1.0 when the component configures no form factor. Computing
    ``|Q|`` in inverse angstrom requires lattice or UB metadata on the data.
    """

    config = component.config if isinstance(component.config, dict) else {}
    ion = str(config.get("ion", "") or "").strip()
    if ion == "__custom__":
        ion = ""
    coefficients = config.get("form_factor_coefficients")
    if not ion and not coefficients:
        return 1.0
    from .fitting import q_modulus_inv_angstrom

    q = q_modulus_inv_angstrom(data)
    return form_factor_sq(q, ion=ion or None, coefficients=coefficients)


def _q_offset_sq_inv_angstrom(
    data: PointData4D, q0: tuple[float, float, float]
) -> np.ndarray:
    """Return ``|q - Q0|^2`` in inverse square angstrom with ``Q0`` in RLU."""

    delta = np.column_stack(
        [
            np.asarray(data.H, dtype=float) - q0[0],
            np.asarray(data.K, dtype=float) - q0[1],
            np.asarray(data.L, dtype=float) - q0[2],
        ]
    )
    if not _metadata_coordinate_units_are_inv_angstrom(data.metadata):
        delta = delta @ _resolve_q_transform(data).T
    return np.einsum("ij,ij->i", delta, delta)


def _local_relaxational_factory(component: Any) -> ModelFunction:
    name = component.name
    scale_key = qualified_parameter_name(name, "scale")
    chi_key = qualified_parameter_name(name, "chi_loc")
    gamma_key = qualified_parameter_name(name, "gamma")

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        chipp = local_relaxational_chipp(
            data.E,
            chi_loc=float(params[chi_key]),
            gamma=float(params[gamma_key]),
        )
        return intensity_from_chipp(
            chipp,
            data.E,
            _dataset_temperature(data),
            scale=float(params[scale_key]),
            form_factor_sq=_form_factor_sq_from_config(component, data),
            polarization=ISOTROPIC_POLARIZATION,
        )

    return model


def _mmp_relaxational_factory(component: Any) -> ModelFunction:
    name = component.name
    keys = {
        parameter: qualified_parameter_name(name, parameter)
        for parameter in ("scale", "chi_pk", "xi", "omega_sf", "q0_h", "q0_k", "q0_l")
    }

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        q0 = (
            float(params[keys["q0_h"]]),
            float(params[keys["q0_k"]]),
            float(params[keys["q0_l"]]),
        )
        chipp = mmp_chipp(
            _q_offset_sq_inv_angstrom(data, q0),
            data.E,
            chi_pk=float(params[keys["chi_pk"]]),
            xi=float(params[keys["xi"]]),
            omega_sf=float(params[keys["omega_sf"]]),
        )
        return intensity_from_chipp(
            chipp,
            data.E,
            _dataset_temperature(data),
            scale=float(params[keys["scale"]]),
            form_factor_sq=_form_factor_sq_from_config(component, data),
            polarization=ISOTROPIC_POLARIZATION,
        )

    return model


def heisenberg_rpa_orbit_labels(component: Any) -> tuple[str, ...]:
    """Return the Heisenberg exchange-parameter names of a component.

    One fit parameter per bond orbit in ``config["orbits"]``, in stored order.
    """

    config = component.config if isinstance(component.config, dict) else {}
    orbits = config.get("orbits") or []
    labels: list[str] = []
    for orbit in orbits:
        label = str(orbit.get("label", "")).strip()
        if not label:
            raise ValueError(
                f"component {component.name!r} has a bond orbit without a label"
            )
        if label in labels:
            raise ValueError(
                f"component {component.name!r} has duplicate bond orbit label {label!r}"
            )
        labels.append(label)
    return tuple(labels)


def _component_has_tensor_terms(config: Mapping[str, Any]) -> bool:
    """Return whether any anisotropic/tensor interaction section is enabled."""

    for section_key in ("anisotropy", "sia"):
        section = config.get(section_key)
        if isinstance(section, dict) and any(
            isinstance(spec, dict) and spec.get("enabled") for spec in section.values()
        ):
            return True
    for section_key in ("dipole", "zeeman"):
        section = config.get(section_key)
        if isinstance(section, dict) and section.get("enabled"):
            return True
    return False


def _component_has_zeeman(config: Mapping[str, Any]) -> bool:
    zeeman = config.get("zeeman")
    return isinstance(zeeman, dict) and bool(zeeman.get("enabled"))


def heisenberg_rpa_parameter_labels(component: Any) -> tuple[str, ...]:
    """All dynamic fit-parameter names of a ``heisenberg_rpa`` component.

    Heisenberg orbit labels (``J1``, ...) plus one coefficient per enabled
    anisotropic-exchange basis element (``J1_S1``, ``J1_D1``, ...) and per
    enabled single-ion-anisotropy basis element (``K1_<site>``, ...), matching
    the parameter names the tensor evaluator assembles.
    """

    labels = list(heisenberg_rpa_orbit_labels(component))
    config = component.config if isinstance(component.config, dict) else {}
    anisotropy = config.get("anisotropy")
    if isinstance(anisotropy, dict):
        for orbit_label, spec in anisotropy.items():
            if not isinstance(spec, dict) or not spec.get("enabled"):
                continue
            for element in spec.get("basis", []):
                labels.append(f"{orbit_label}_{element['name']}")
    sia = config.get("sia")
    if isinstance(sia, dict):
        for class_label, spec in sia.items():
            if not isinstance(spec, dict) or not spec.get("enabled"):
                continue
            for element in spec.get("basis", []):
                labels.append(f"{element['name']}_{class_label}")
    dipole = config.get("dipole")
    if isinstance(dipole, dict) and dipole.get("enabled"):
        labels.append("D_dip")
    if _component_has_zeeman(config):
        labels.extend(("g_factor", "chi_perp_ratio", "gamma_perp_ratio"))
    from .closures import ClosureSpec

    closure_spec = ClosureSpec.from_config(config)
    if closure_spec is not None:
        labels.extend(closure_spec.parameter_names())
    return tuple(labels)


class _RpaComponentEvaluator:
    """Shared evaluation state for one ``heisenberg_rpa`` component.

    Parses the crystal/orbit config once, folds the network onto its primitive
    cell, and caches the Q-dependent phase geometry per fitted dataset. Exposes
    :meth:`value` (measured intensity) and :meth:`gradients` (its exact
    parameter derivatives), which share the same cached geometry so the model
    and its analytic Jacobian never rebuild it.
    """

    def __init__(self, component: Any) -> None:
        name = component.name
        config = component.config if isinstance(component.config, dict) else {}
        site_positions = config.get("site_positions")
        if not site_positions:
            raise ValueError(
                f"heisenberg_rpa component {name!r} defines no magnetic site "
                "positions; configure the crystal and generate bond orbits first"
            )
        orbits = config.get("orbits") or []
        if not orbits:
            raise ValueError(
                f"heisenberg_rpa component {name!r} defines no bond orbits; "
                "generate symmetry orbits or enter bonds manually first"
            )
        self.component = component
        self.config = config
        self.labels = heisenberg_rpa_orbit_labels(component)
        self.j_keys = {label: qualified_parameter_name(name, label) for label in self.labels}
        self.scale_key = qualified_parameter_name(name, "scale")
        self.chi0_key = qualified_parameter_name(name, "chi0")
        self.gamma0_key = qualified_parameter_name(name, "gamma0")
        self.tensor_mode = _component_has_tensor_terms(config)
        self.zeeman_mode = _component_has_zeeman(config)
        self._zeeman_keys = {
            key: qualified_parameter_name(name, key)
            for key in ("g_factor", "chi_perp_ratio", "gamma_perp_ratio")
        }
        from .closures import ClosureSpec

        self.closure_spec = ClosureSpec.from_config(config)
        closure_labels = (
            self.closure_spec.parameter_names() if self.closure_spec else ()
        )
        self._closure_keys = {
            label: qualified_parameter_name(name, label) for label in closure_labels
        }
        # The tensor J(Q) coefficients exclude the Zeeman propagator and
        # closure parameters (neither enters the interaction matrix).
        self.tensor_keys = {
            label: qualified_parameter_name(name, label)
            for label in heisenberg_rpa_parameter_labels(component)
            if label not in self._zeeman_keys and label not in self._closure_keys
        }
        # Closure solve/eigenvalue caches (see _solve_closure). Keyed by exact
        # parameter tuples: scipy evaluates value and FD columns back to back
        # at identical floats, so exact keys hit without rounding tolerance.
        self._closure_moment_cache: dict[tuple, Any] = {}
        self._closure_result_cache: dict[tuple, Any] = {}
        self._closure_bz_context: Any = None
        if self.tensor_mode:
            # Fold onto the primitive cell carrying the tensor payloads: pure
            # lattice translations do not rotate spins, so anisotropic exchange,
            # single-ion anisotropy, and dipole coupling are invariant under the
            # fold. The bond rotations/reversals and per-site generators travel
            # with the reduced network; the dipole Ewald tensor is rebuilt from
            # the reduced sites (unchanged lattice). Declines to the full cell if
            # any payload cannot fold cleanly.
            (
                self.site_positions,
                self.orbits,
                self._site_rotations,
                self._sia,
            ) = reduce_site_network_with_tensors(
                site_positions,
                orbits,
                site_rotations=config.get("site_rotations"),
                sia=config.get("sia"),
                dipole_enabled=bool((config.get("dipole") or {}).get("enabled")),
            )
            self._anisotropy = config.get("anisotropy")
            self._dipole = config.get("dipole")
            self._lattice = (config.get("crystal") or {}).get("lattice")
        else:
            # Fold the network onto its primitive translational cell (exact:
            # identical chi'' at far lower eigendecomposition cost). Purely an
            # evaluation-time detail -- the component config, GUI, and fit
            # outputs all stay in the user's specified cell.
            self.site_positions, self.orbits = reduce_site_network(site_positions, orbits)
        # Q-dependent, exchange-independent phase arrays (and the Q-fixed form
        # factor) are expensive to build but constant for a given dataset. The
        # cache holds a reference to the data object and verifies identity on
        # lookup, so the id() key can never alias a freed-then-reused object.
        self._geometry_cache: dict[int, tuple[PointData4D, Any, Any, Any]] = {}

    def _geometry(self, data: PointData4D) -> tuple[Any, Any, Any]:
        cached = self._geometry_cache.get(id(data))
        if cached is not None and cached[0] is data:
            return cached[1], cached[2], cached[3]
        if len(self._geometry_cache) > 32:
            self._geometry_cache.clear()
        geometry = build_rpa_geometry(data.H, data.K, data.L, self.site_positions, self.orbits)
        form_factor_sq = _form_factor_sq_from_config(self.component, data)
        tensor_context = None
        if self.tensor_mode:
            tensor_context = self._build_tensor_context(geometry, data)
        self._geometry_cache[id(data)] = (data, geometry, form_factor_sq, tensor_context)
        return geometry, form_factor_sq, tensor_context

    def _build_tensor_context(self, geometry: Any, data: PointData4D) -> tuple[Any, Any]:
        from .tensor_rpa import build_tensor_structure, cartesian_qhat_per_point

        matrix = data.metadata.get("rlu_to_inv_angstrom_matrix")
        if matrix is None:
            raise ValueError(
                f"heisenberg_rpa component {self.component.name!r} has anisotropic "
                "terms enabled but the fit points lack an RLU->inverse-angstrom "
                "matrix; the data group needs lattice parameters"
            )
        structure = build_tensor_structure(
            geometry,
            self.site_positions,
            orbits=self.orbits,
            lattice=self._lattice,
            anisotropy=self._anisotropy,
            sia=self._sia,
            site_rotations=self._site_rotations,
            dipole=self._dipole,
        )
        q_hat = cartesian_qhat_per_point(geometry, np.asarray(matrix, dtype=float))
        return structure, q_hat

    def _j_values(self, params: dict[str, float]) -> dict[str, float]:
        return {label: float(params[key]) for label, key in self.j_keys.items()}

    def _tensor_values(self, params: dict[str, float]) -> dict[str, float]:
        return {label: float(params[key]) for label, key in self.tensor_keys.items()}

    def _closure_context(self) -> tuple[Any, Any]:
        """BZ-grid geometry (and tensor structure) for the closure integrals.

        Built once per evaluator: the grid depends only on the component's
        site network and closure config, never on a dataset.
        """

        if self._closure_bz_context is None:
            from .sum_rules import bz_sample_hkl

            grid_size = self.closure_spec.bz_grid if self.closure_spec is not None else 12
            grid = bz_sample_hkl(grid_size)
            bz_geometry = build_rpa_geometry(
                grid[:, 0], grid[:, 1], grid[:, 2], self.site_positions, self.orbits
            )
            bz_structure = None
            if self.tensor_mode:
                from .tensor_rpa import build_tensor_structure

                bz_structure = build_tensor_structure(
                    bz_geometry,
                    self.site_positions,
                    orbits=self.orbits,
                    lattice=self._lattice,
                    anisotropy=self._anisotropy,
                    sia=self._sia,
                    site_rotations=self._site_rotations,
                    dipole=self._dipole,
                )
            self._closure_bz_context = (bz_geometry, bz_structure)
        return self._closure_bz_context

    def _closure_moment_model(
        self, params: dict[str, float], field: np.ndarray | None
    ) -> Any:
        """Moment backend for the closure solver, cached per parameter point."""

        from .closures import TierAMoments, TierBMoments

        bz_geometry, bz_structure = self._closure_context()
        if self.zeeman_mode:
            from .tensor_rpa import (
                MU_B_MEV_PER_T,
                assemble_tensor_exchange,
                zeeman_cartesian_propagator,
            )

            magnitude = float(np.linalg.norm(field)) if field is not None else 0.0
            b_hat = (
                field / magnitude if magnitude > 0 else np.array([0.0, 0.0, 1.0])
            )
            g_factor = float(params[self._zeeman_keys["g_factor"]])
            chi_perp = float(params[self._zeeman_keys["chi_perp_ratio"]])
            gamma_perp = float(params[self._zeeman_keys["gamma_perp_ratio"]])
            omega_larmor = g_factor * MU_B_MEV_PER_T * magnitude
            tensor_values = self._tensor_values(params)
            key = (
                "tier_b",
                tuple(sorted(tensor_values.items())),
                omega_larmor,
                chi_perp,
                gamma_perp,
                tuple(np.asarray(b_hat, dtype=float)),
            )
            model = self._closure_moment_cache.get(key)
            if model is None:
                exchange = assemble_tensor_exchange(bz_structure, tensor_values)

                def builder(omega, chi0, gamma0):
                    return zeeman_cartesian_propagator(
                        omega,
                        b_hat,
                        chi0=chi0,
                        gamma0=gamma0,
                        omega_larmor=omega_larmor,
                        chi_perp_ratio=chi_perp,
                        gamma_perp_ratio=gamma_perp,
                    )

                model = TierBMoments(
                    exchange,
                    bz_geometry.n_sites,
                    builder,
                    omega_points=self.closure_spec.omega_points,
                    static_response_bound=max(1.0, chi_perp),
                )
        elif self.tensor_mode:
            from .tensor_rpa import assemble_tensor_exchange

            tensor_values = self._tensor_values(params)
            key = ("tier_a_tensor", tuple(sorted(tensor_values.items())))
            model = self._closure_moment_cache.get(key)
            if model is None:
                exchange = assemble_tensor_exchange(bz_structure, tensor_values)
                model = TierAMoments(
                    np.linalg.eigvalsh(exchange),
                    bz_geometry.n_sites,
                    isotropic_components=1,
                )
        else:
            j_values = self._j_values(params)
            key = ("tier_a", tuple(sorted(j_values.items())))
            model = self._closure_moment_cache.get(key)
            if model is None:
                exchange = rpa_exchange_matrix(bz_geometry, j_values)
                model = TierAMoments(
                    np.linalg.eigvalsh(exchange),
                    bz_geometry.n_sites,
                    isotropic_components=3,
                )
        if len(self._closure_moment_cache) > 8:
            self._closure_moment_cache.clear()
        self._closure_moment_cache[key] = model
        return model

    def _solve_closure(
        self,
        params: dict[str, float],
        temperature: Any,
        field: np.ndarray | None,
    ) -> Any:
        """Solve the configured closure at this dataset's (T, B), cached."""

        from .closures import solve_closure

        unique_t = np.unique(np.atleast_1d(np.asarray(temperature, dtype=float)))
        if unique_t.size != 1:
            raise ValueError(
                "sum-rule closures need a single temperature per dataset; "
                "per-point temperatures are only supported for magnetization "
                "datasets"
            )
        t = float(unique_t[0])
        relevant = sorted(
            set(self.j_keys.values())
            | set(self.tensor_keys.values())
            | set(self._closure_keys.values())
            | {self.chi0_key, self.gamma0_key}
            | (set(self._zeeman_keys.values()) if self.zeeman_mode else set())
        )
        key = (
            t,
            tuple(np.asarray(field, dtype=float)) if field is not None else None,
            tuple(float(params[k]) for k in relevant),
        )
        cached = self._closure_result_cache.get(key)
        if cached is not None:
            return cached
        model = self._closure_moment_model(params, field)
        result = solve_closure(
            self.closure_spec,
            model,
            chi0=float(params[self.chi0_key]),
            gamma0=float(params[self.gamma0_key]),
            temperature_K=t,
            params={
                label: float(params[key_])
                for label, key_ in self._closure_keys.items()
            },
        )
        if len(self._closure_result_cache) > 256:
            self._closure_result_cache.clear()
        self._closure_result_cache[key] = result
        return result

    def value(self, data: PointData4D, params: dict[str, float]) -> np.ndarray:
        temperature = _dataset_temperature(data)
        # Validate the applied field before the instability guard below, so a
        # missing field raises its actionable error instead of being caught and
        # turned into the 1e6 sentinel.
        field = _dataset_magnetic_field(data) if self.zeeman_mode else None
        geometry, form_factor_sq, tensor_context = self._geometry(data)
        try:
            chi0 = float(params[self.chi0_key])
            lambda_shift = 0.0
            if self.closure_spec is not None:
                closure = self._solve_closure(params, temperature, field)
                chi0 = closure.chi0_eff
                lambda_shift = closure.lambda_shift
            if self.zeeman_mode:
                from .tensor_rpa import (
                    tensor_rpa_zeeman_unpolarized_chipp,
                    zeeman_cartesian_propagator,
                    MU_B_MEV_PER_T,
                )

                structure, q_hat = tensor_context
                magnitude = float(np.linalg.norm(field))
                b_hat = field / magnitude if magnitude > 0 else np.array([0.0, 0.0, 1.0])
                g_factor = float(params[self._zeeman_keys["g_factor"]])
                energy = np.asarray(data.E, dtype=float)
                propagator = zeeman_cartesian_propagator(
                    energy,
                    b_hat,
                    chi0=chi0,
                    gamma0=float(params[self.gamma0_key]),
                    omega_larmor=g_factor * MU_B_MEV_PER_T * magnitude,
                    chi_perp_ratio=float(params[self._zeeman_keys["chi_perp_ratio"]]),
                    gamma_perp_ratio=float(params[self._zeeman_keys["gamma_perp_ratio"]]),
                )
                chipp = tensor_rpa_zeeman_unpolarized_chipp(
                    structure, geometry, energy, q_hat, propagator,
                    param_values=self._tensor_values(params),
                    lambda_shift=lambda_shift,
                )
                polarization = 1.0
            elif self.tensor_mode:
                from .tensor_rpa import tensor_rpa_unpolarized_chipp

                structure, q_hat = tensor_context
                chipp = tensor_rpa_unpolarized_chipp(
                    structure,
                    geometry,
                    np.asarray(data.E, dtype=float),
                    q_hat,
                    chi0=chi0,
                    gamma0=float(params[self.gamma0_key]),
                    param_values=self._tensor_values(params),
                    lambda_shift=lambda_shift,
                )
                # The unpolarized channel already carries the polarization
                # average, so no extra scalar polarization factor here.
                polarization = 1.0
            else:
                chipp = heisenberg_rpa_chipp(
                    geometry,
                    data.E,
                    chi0=chi0,
                    gamma0=float(params[self.gamma0_key]),
                    j_values=self._j_values(params),
                    lambda_shift=lambda_shift,
                )
                polarization = ISOTROPIC_POLARIZATION
        except ValueError:
            # Unphysical trial parameters (RPA instability, non-positive
            # chi0/gamma0, unsatisfiable closure). Optimizers probe these while
            # searching; a huge finite misfit steers them back without
            # aborting the fit.
            return np.full(data.size, 1e6, dtype=float)
        return intensity_from_chipp(
            chipp,
            data.E,
            temperature,
            scale=float(params[self.scale_key]),
            form_factor_sq=form_factor_sq,
            polarization=polarization,
        )

    def _static_chi_grid_and_q0(
        self, params: dict[str, float], chi0: float, lambda_shift: float
    ) -> tuple[float, float]:
        """Uniform static susceptibility at Q=0 and its peak over the BZ grid.

        Kramers-Kronig of the relaxational modes closes to their static
        amplitude, so ``chi(Q, 0) = sum_nu w_nu chi0_eff/(1 - lambda' chi0_eff)``
        with uniform mode weights. The Q=0 value is the (zero-field) bulk
        susceptibility; the peak locates the incipient ordering vector.
        Computed from the same J(Q) eigenstructure the closure uses (zero-field
        limit in Zeeman mode).
        """

        from .sum_rules import static_chi_modes

        bz_geometry, bz_structure = self._closure_context()
        q0_geometry = build_rpa_geometry(
            [0.0], [0.0], [0.0], self.site_positions, self.orbits
        )

        def eig_weights(geometry: Any) -> tuple[FloatArray, FloatArray]:
            if self.tensor_mode or self.zeeman_mode:
                from .tensor_rpa import assemble_tensor_exchange, build_tensor_structure

                structure = (
                    bz_structure
                    if geometry is bz_geometry
                    else build_tensor_structure(
                        geometry,
                        self.site_positions,
                        orbits=self.orbits,
                        lattice=self._lattice,
                        anisotropy=self._anisotropy,
                        sia=self._sia,
                        site_rotations=self._site_rotations,
                        dipole=self._dipole,
                    )
                )
                exchange = assemble_tensor_exchange(structure, self._tensor_values(params))
                lam, modes = np.linalg.eigh(exchange)
                n_sites = structure.n_sites
                amps = modes.reshape(geometry.n_q, n_sites, 3, 3 * n_sites).sum(axis=1)
                weights = (np.abs(amps) ** 2).sum(axis=1) / n_sites
            else:
                exchange = rpa_exchange_matrix(geometry, self._j_values(params))
                lam, modes = np.linalg.eigh(exchange)
                weights = np.abs(modes.sum(axis=1)) ** 2 / geometry.n_sites
            return lam, weights

        lam_grid, w_grid = eig_weights(bz_geometry)
        lam_q0, w_q0 = eig_weights(q0_geometry)
        chi_grid = static_chi_modes(lam_grid, w_grid, chi0=chi0, lambda_shift=lambda_shift)
        chi_q0 = static_chi_modes(lam_q0, w_q0, chi0=chi0, lambda_shift=lambda_shift)
        return float(chi_q0[0]), float(np.max(chi_grid))

    # Default energy cutoff for the moment integral when no closure is active
    # (diagnostics still report an effective moment). Overridden by the
    # closure's configured cutoff when present.
    _DEFAULT_DIAGNOSTIC_CUTOFF_MEV = 100.0

    def diagnostics(self, data: PointData4D, params: dict[str, float]) -> dict[str, float]:
        """Post-fit physics diagnostics for one dataset (see docs/theory_notes).

        Returns a flat JSON-safe dict: the effective fluctuating moment
        ``mu_eff_sq`` (its zero-point/thermal split), the closure internals
        ``lambda_shift``/``chi0_eff``, ``chi0_gamma0``, the distance to the RPA
        instability, and the static susceptibility at Q=0 and its BZ peak. On
        unphysical parameters it returns a minimal record flagged ``unstable``.
        """

        temperature = _dataset_temperature(data)
        field = _dataset_magnetic_field(data) if self.zeeman_mode else None
        t_values = np.unique(np.atleast_1d(np.asarray(temperature, dtype=float)))
        t = float(t_values[0]) if t_values.size else float("nan")
        chi0 = float(params[self.chi0_key])
        gamma0 = float(params[self.gamma0_key])
        lambda_shift = 0.0
        cutoff = (
            self.closure_spec.energy_cutoff_mev
            if self.closure_spec is not None
            else self._DEFAULT_DIAGNOSTIC_CUTOFF_MEV
        )
        record: dict[str, float] = {"temperature": t, "gamma0": gamma0}
        try:
            if self.closure_spec is not None:
                closure = self._solve_closure(params, temperature, field)
                chi0 = closure.chi0_eff
                lambda_shift = closure.lambda_shift
            model = self._closure_moment_model(params, field)
            zero_point, thermal = model.moment(
                chi0=chi0,
                gamma0=gamma0,
                temperature_K=t,
                cutoff_mev=cutoff,
                lambda_shift=lambda_shift,
            )
            lam_max = float(model._lambda_max)
            chi_q0, chi_peak = self._static_chi_grid_and_q0(params, chi0, lambda_shift)
        except ValueError:
            record.update({"chi0_eff": chi0, "unstable": 1.0})
            return record
        record.update(
            {
                "chi0_eff": chi0,
                "chi0_gamma0": chi0 * gamma0,
                "lambda_shift": lambda_shift,
                "mu_eff_sq": zero_point + thermal,
                "m2_zero_point": zero_point,
                "m2_thermal": thermal,
                "distance_to_instability": 1.0 - (lam_max - lambda_shift) * chi0,
                "chi_static_q0": chi_q0,
                "chi_static_qpeak": chi_peak,
            }
        )
        return record

    def gradients(self, data: PointData4D, params: dict[str, float]) -> dict[str, np.ndarray]:
        """Return ``d(intensity)/d(param)`` keyed by qualified parameter name."""

        temperature = _dataset_temperature(data)
        geometry, form_factor_sq, _tensor_context = self._geometry(data)
        scale = float(params[self.scale_key])
        try:
            chipp, chipp_grads = heisenberg_rpa_chipp_and_gradients(
                geometry,
                data.E,
                chi0=float(params[self.chi0_key]),
                gamma0=float(params[self.gamma0_key]),
                j_values=self._j_values(params),
            )
        except ValueError:
            # Sentinel-penalty region: the value is a constant with no
            # parameter dependence, so every column is zero.
            zero = np.zeros(data.size, dtype=float)
            columns = {self.scale_key: zero, self.chi0_key: zero, self.gamma0_key: zero}
            columns.update({key: zero for key in self.j_keys.values()})
            return columns
        # I = scale * (2/3) * |f|^2 * chipp / bose(E, T), linear in chipp. The
        # scale column is I evaluated at unit scale; d(I)/d(chipp) is I with
        # chipp replaced by 1 (the Bose/form-factor/scale prefactor). Both are
        # computed directly -- never by dividing by chipp, which would be
        # singular at nodes where chipp = 0 but the sensitivity is finite.
        ones = np.ones(data.size, dtype=float)
        scale_column = np.asarray(
            intensity_from_chipp(
                chipp,
                data.E,
                temperature,
                scale=1.0,
                form_factor_sq=form_factor_sq,
                polarization=ISOTROPIC_POLARIZATION,
            ),
            dtype=float,
        )
        d_intensity_d_chipp = np.asarray(
            intensity_from_chipp(
                ones,
                data.E,
                temperature,
                scale=scale,
                form_factor_sq=form_factor_sq,
                polarization=ISOTROPIC_POLARIZATION,
            ),
            dtype=float,
        )
        columns = {self.scale_key: scale_column}
        columns[self.chi0_key] = d_intensity_d_chipp * chipp_grads["chi0"]
        columns[self.gamma0_key] = d_intensity_d_chipp * chipp_grads["gamma0"]
        for label, key in self.j_keys.items():
            columns[key] = d_intensity_d_chipp * chipp_grads[label]
        return columns


def _heisenberg_rpa_factory(component: Any) -> ModelFunction:
    return _RpaComponentEvaluator(component).value


def compute_component_diagnostics(
    component: Any, data: PointData4D, params: Mapping[str, float]
) -> dict[str, float] | None:
    """Post-fit physics diagnostics for one ``heisenberg_rpa`` component.

    Returns a flat JSON-safe metrics dict (see
    :meth:`_RpaComponentEvaluator.diagnostics`), or ``None`` when the component
    is not a ``heisenberg_rpa`` model or cannot be evaluated on ``data``
    (missing crystal config, incompatible dataset). ``params`` are fully
    resolved qualified parameter values.
    """

    if getattr(component, "type", None) != "heisenberg_rpa":
        return None
    try:
        evaluator = _RpaComponentEvaluator(component)
        return evaluator.diagnostics(data, dict(params))
    except (ValueError, KeyError):
        return None


def _heisenberg_rpa_jacobian_factory(component: Any) -> "ModelJacobian | None":
    # Analytic gradients are implemented for the scalar Heisenberg path only;
    # components with anisotropic tensor terms fall back to finite differences
    # (the fit engine gates on every component providing a Jacobian). Sum-rule
    # closures make chi0_eff/lambda an implicit function of every parameter,
    # so they also use the finite-difference fallback.
    config = component.config if isinstance(component.config, dict) else {}
    if _component_has_tensor_terms(config):
        return None
    from .closures import ClosureSpec

    if ClosureSpec.from_config(config) is not None:
        return None
    return _RpaComponentEvaluator(component).gradients


@dataclass(frozen=True)
class ModelTypeInfo:
    """Fit-engine registration for one model component type.

    ``data_types`` lists the dataset data types the model can be applied to;
    ``("*",)`` means the model works with any dataset. ``dynamic_parameters``
    optionally derives additional parameter names from a component's
    configuration (e.g. one exchange constant per bond orbit).
    """

    parameters: tuple[str, ...]
    data_types: tuple[str, ...]
    factory: Callable[[Any], ModelFunction]
    dynamic_parameters: Callable[[Any], tuple[str, ...]] | None = None
    jacobian_factory: Callable[[Any], ModelJacobian] | None = None
    """Optional analytic-Jacobian builder. When every component applied to a
    dataset provides one, the optimizer uses exact gradients instead of finite
    differences; otherwise it silently falls back."""


def component_parameter_names(component: Any) -> tuple[str, ...]:
    """Return all parameter names of a component, static plus config-derived."""

    info = MODEL_TYPE_REGISTRY[component.type]
    if info.dynamic_parameters is None:
        return info.parameters
    return (*info.parameters, *info.dynamic_parameters(component))


MODEL_TYPE_REGISTRY: dict[str, ModelTypeInfo] = {
    "constant_background": ModelTypeInfo(
        parameters=("constant",),
        data_types=("*",),
        factory=_constant_background_factory,
        jacobian_factory=_constant_background_jacobian_factory,
    ),
    "linear_background": ModelTypeInfo(
        parameters=("c0", "c1"),
        data_types=("single_crystal_inelastic", "powder_inelastic"),
        factory=_linear_background_factory,
        jacobian_factory=_linear_background_jacobian_factory,
    ),
    "single_q_paramagnon": ModelTypeInfo(
        parameters=("amplitude", "q0_h", "q0_k", "q0_l", "kappa", "omega_sf"),
        data_types=("single_crystal_inelastic",),
        factory=_single_q_paramagnon_factory,
    ),
    "local_relaxational": ModelTypeInfo(
        parameters=("scale", "chi_loc", "gamma"),
        data_types=("single_crystal_inelastic", "powder_inelastic"),
        factory=_local_relaxational_factory,
    ),
    "mmp_relaxational": ModelTypeInfo(
        parameters=("scale", "chi_pk", "xi", "omega_sf", "q0_h", "q0_k", "q0_l"),
        data_types=("single_crystal_inelastic",),
        factory=_mmp_relaxational_factory,
    ),
    "heisenberg_rpa": ModelTypeInfo(
        parameters=("scale", "chi0", "gamma0"),
        data_types=("single_crystal_inelastic",),
        factory=_heisenberg_rpa_factory,
        dynamic_parameters=heisenberg_rpa_parameter_labels,
        jacobian_factory=_heisenberg_rpa_jacobian_factory,
    ),
}


def model_supports_data_type(model_type: str, data_type: str) -> bool:
    """Return whether a registered model type can fit a dataset data type."""

    info = MODEL_TYPE_REGISTRY.get(model_type)
    if info is None:
        return False
    if "*" in info.data_types:
        return True
    return (data_type or FALLBACK_DATA_TYPE) in info.data_types


@dataclass(frozen=True)
class FitDatasetInput:
    """One prepared dataset offered to the fit compiler.

    ``data`` must already reflect masks and any rebinning. Fixed scale factors
    should already be applied; fitted scale factors should be passed as
    ``scale_value`` with ``scale_vary=True`` so the compiler can add a dataset
    scale parameter.
    """

    name: str
    data: PointData4D
    weight: float = 1.0
    data_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    scale_value: float = 1.0
    scale_vary: bool = False


@dataclass(frozen=True)
class ParameterInstance:
    """Where one optimizer parameter came from."""

    name: str
    component: str
    parameter: str
    scope: str
    datasets: tuple[str, ...]


@dataclass
class CompiledFitProblem:
    """A FitProblem plus the bookkeeping to map results back to components."""

    problem: FitProblem
    parameter_instances: dict[str, ParameterInstance]
    components_by_dataset: dict[str, list[str]]
    skipped_datasets: list[str]

    def instances_for(self, component: str, parameter: str) -> list[ParameterInstance]:
        """Return all optimizer instances of one component parameter."""

        return [
            instance
            for instance in self.parameter_instances.values()
            if instance.component == component and instance.parameter == parameter
        ]


def sharing_mode(component: Any, parameter: str) -> str:
    """Return the sharing mode of one component parameter.

    An explicit ``sharing`` entry wins; otherwise the legacy ``global_fit``
    boolean maps ``True`` to ``"global"`` and ``False`` to ``"per_dataset"``.
    """

    entry = component.sharing.get(parameter) if isinstance(component.sharing, dict) else None
    if isinstance(entry, dict) and entry.get("mode") in SHARING_MODES:
        return str(entry["mode"])
    return "global" if component.global_fit.get(parameter, True) else "per_dataset"


def dataset_scale_parameter_name(dataset_name: str) -> str:
    """Return the optimizer parameter name for a fitted dataset scale."""

    return f"dataset_scale[{dataset_name}]"


def parameter_limits(component: Any, parameter: str) -> tuple[float | None, float | None]:
    """Return ``(min, max)`` bounds for one component parameter."""

    raw = component.limits.get(parameter) if isinstance(component.limits, dict) else None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None, None
    lower = None if raw[0] in (None, "") else float(raw[0])
    upper = None if raw[1] in (None, "") else float(raw[1])
    return lower, upper


def _parameter_value(component: Any, parameter: str) -> float:
    raw = component.parameters.get(parameter, 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"parameter {parameter!r} of component {component.name!r} has "
            f"non-numeric value {raw!r}"
        ) from exc


def _tie_keys(component: Any, parameter: str, dataset_names: Sequence[str]) -> dict[str, str]:
    """Return dataset name -> tie key for per_dataset/grouped instancing."""

    mode = sharing_mode(component, parameter)
    if mode == "per_dataset":
        return {name: name for name in dataset_names}
    entry = component.sharing.get(parameter, {}) if isinstance(component.sharing, dict) else {}
    groups = entry.get("groups", {}) if isinstance(entry, dict) else {}
    return {name: str(groups.get(name, name)) for name in dataset_names}


def compile_fit_problem(
    components: Sequence[Any],
    datasets: Sequence[FitDatasetInput],
    *,
    description: str = "",
) -> CompiledFitProblem:
    """Compile model components and prepared datasets into a FitProblem.

    Only enabled components are used. A component applies to a dataset when
    the dataset is listed in ``applies_to`` (or ``applies_to`` is ``None``)
    and the model type supports the dataset's data type. Datasets that no
    component applies to are excluded from the problem and reported in
    ``skipped_datasets``. Parameters of components that apply to no dataset
    are not emitted, so the optimizer never sees insensitive parameters.
    """

    if not datasets:
        raise ValueError("compile_fit_problem requires at least one dataset")
    dataset_names = [dataset.name for dataset in datasets]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("dataset names must be unique")

    active = [component for component in components if getattr(component, "enabled", True)]
    seen: set[str] = set()
    for component in active:
        if component.type not in MODEL_TYPE_REGISTRY:
            raise ValueError(f"model type {component.type!r} is not registered for fitting")
        if component.name in seen:
            raise ValueError(f"duplicate model component name {component.name!r}")
        seen.add(component.name)

    applicable: dict[str, list[str]] = {}
    components_by_dataset: dict[str, list[str]] = {name: [] for name in dataset_names}
    for component in active:
        names: list[str] = []
        for dataset in datasets:
            if component.applies_to is not None and dataset.name not in component.applies_to:
                continue
            if not model_supports_data_type(component.type, dataset.data_type):
                continue
            names.append(dataset.name)
            components_by_dataset[dataset.name].append(component.name)
        applicable[component.name] = names

    fitted = [dataset for dataset in datasets if components_by_dataset[dataset.name]]
    skipped = [name for name in dataset_names if not components_by_dataset[name]]
    if not fitted:
        raise ValueError("no dataset is matched by any enabled model component")

    specs: list[ParameterSpec] = []
    instances: dict[str, ParameterInstance] = {}
    bindings: dict[str, dict[str, ParameterBinding]] = {name: {} for name in dataset_names}

    def emit(spec: ParameterSpec, instance: ParameterInstance) -> None:
        if spec.name in instances:
            raise ValueError(f"duplicate fit parameter name {spec.name!r}")
        specs.append(spec)
        instances[spec.name] = instance

    for component in active:
        component_datasets = applicable[component.name]
        if not component_datasets:
            continue
        for parameter in component_parameter_names(component):
            qualified = qualified_parameter_name(component.name, parameter)
            value = _parameter_value(component, parameter)
            vary = bool(component.fit_parameters.get(parameter, False))
            lower, upper = parameter_limits(component, parameter)
            mode = sharing_mode(component, parameter)
            if mode == "global":
                emit(
                    ParameterSpec(name=qualified, value=value, min=lower, max=upper, vary=vary),
                    ParameterInstance(
                        name=qualified,
                        component=component.name,
                        parameter=parameter,
                        scope="global",
                        datasets=tuple(component_datasets),
                    ),
                )
                continue
            keys = _tie_keys(component, parameter, component_datasets)
            for key in dict.fromkeys(keys.values()):
                name = instanced_parameter_name(component.name, parameter, key)
                emit(
                    ParameterSpec(name=name, value=value, min=lower, max=upper, vary=vary),
                    ParameterInstance(
                        name=name,
                        component=component.name,
                        parameter=parameter,
                        scope=key,
                        datasets=tuple(
                            dataset for dataset, tie in keys.items() if tie == key
                        ),
                    ),
                )
            for dataset_name, key in keys.items():
                bindings[dataset_name][qualified] = instanced_parameter_name(
                    component.name, parameter, key
                )

    scale_parameters: dict[str, str] = {}
    for dataset in fitted:
        if not dataset.scale_vary:
            continue
        name = dataset_scale_parameter_name(dataset.name)
        emit(
            ParameterSpec(
                name=name,
                value=float(dataset.scale_value),
                vary=True,
                description=f"Scale factor for dataset {dataset.name}",
            ),
            ParameterInstance(
                name=name,
                component="dataset",
                parameter="scale_factor",
                scope=dataset.name,
                datasets=(dataset.name,),
            ),
        )
        scale_parameters[dataset.name] = name

    derived = _compile_constraints(active, applicable, specs, instances)

    fit_datasets: list[FitDataset] = []
    for dataset in fitted:
        components_here = [
            component for component in active if dataset.name in applicable[component.name]
        ]
        evaluators = [
            MODEL_TYPE_REGISTRY[component.type].factory(component)
            for component in components_here
        ]
        # An analytic Jacobian is available for the dataset only when *every*
        # component on it provides one; otherwise the optimizer falls back to
        # finite differences for the whole problem.
        jacobian_factories = [
            MODEL_TYPE_REGISTRY[component.type].jacobian_factory
            for component in components_here
        ]
        model_jacobian = None
        if (
            components_here
            and dataset.name not in scale_parameters
            and all(factory is not None for factory in jacobian_factories)
        ):
            # A registry entry may have a jacobian factory that still declines
            # (returns None) for a particular component configuration (e.g. a
            # heisenberg_rpa component with anisotropic tensor terms). Only use
            # analytic Jacobians when every component actually yields one.
            built = [factory(component) for factory, component in zip(jacobian_factories, components_here)]
            if all(jacobian is not None for jacobian in built):
                model_jacobian = _additive_jacobian(built)
        fit_datasets.append(
            FitDataset(
                name=dataset.name,
                data=dataset.data,
                weight=float(dataset.weight),
                parameter_bindings=dict(bindings[dataset.name]),
                model=_additive_model(evaluators),
                data_scale_parameter=scale_parameters.get(dataset.name),
                model_jacobian=model_jacobian,
                metadata=dict(dataset.metadata),
            )
        )

    problem = FitProblem(
        datasets=fit_datasets,
        model=None,
        parameter_specs=tuple(specs),
        derived_parameters=tuple(derived),
        description=description,
    )
    return CompiledFitProblem(
        problem=problem,
        parameter_instances=instances,
        components_by_dataset=components_by_dataset,
        skipped_datasets=skipped,
    )


def _additive_model(evaluators: Sequence[ModelFunction]) -> ModelFunction:
    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        total = np.zeros(data.size, dtype=float)
        for evaluate in evaluators:
            total += np.asarray(evaluate(data, params), dtype=float)
        return total

    return model


def _additive_jacobian(jacobians: Sequence[ModelJacobian]) -> ModelJacobian:
    """Merge component Jacobians; each contributes its own qualified names."""

    def jacobian(data: PointData4D, params: dict[str, float]) -> dict[str, np.ndarray]:
        columns: dict[str, np.ndarray] = {}
        for evaluate in jacobians:
            for key, column in evaluate(data, params).items():
                existing = columns.get(key)
                arr = np.asarray(column, dtype=float)
                columns[key] = arr if existing is None else existing + arr
        return columns

    return jacobian


def _compile_constraints(
    components: Sequence[Any],
    applicable: dict[str, list[str]],
    specs: list[ParameterSpec],
    instances: dict[str, ParameterInstance],
) -> list[DerivedParameter]:
    """Reparameterize inequality constraints as bounded offset parameters."""

    derived: list[DerivedParameter] = []
    specs_by_name = {spec.name: spec for spec in specs}
    for component in components:
        if not applicable.get(component.name):
            continue
        for constraint in getattr(component, "constraints", []) or []:
            parameter = str(constraint.get("parameter", ""))
            op = str(constraint.get("op", ">="))
            reference = constraint.get("reference")
            qualified = qualified_parameter_name(component.name, parameter)
            target = specs_by_name.get(qualified)
            if target is None:
                if any(name.startswith(f"{qualified}[") for name in specs_by_name):
                    raise ValueError(
                        f"constraint on {qualified!r} requires global sharing mode"
                    )
                raise ValueError(f"constraint references unknown parameter {qualified!r}")
            if op not in (">=", "<="):
                raise ValueError(f"unsupported constraint operator {op!r} on {qualified!r}")
            if target.min is not None or target.max is not None:
                raise ValueError(
                    f"parameter {qualified!r} cannot have both limits and a constraint"
                )
            if not target.vary:
                continue

            if isinstance(reference, str):
                reference_spec = specs_by_name.get(reference)
                if reference_spec is None:
                    raise ValueError(
                        f"constraint on {qualified!r} references unknown parameter "
                        f"{reference!r}"
                    )
                base: str | float = reference
                reference_value = reference_spec.value
            else:
                base = float(reference)
                reference_value = float(reference)

            sign = 1.0 if op == ">=" else -1.0
            offset_name = f"{qualified}{CONSTRAINT_OFFSET_SUFFIX}"
            offset_value = max(sign * (target.value - reference_value), 0.0)
            specs.remove(target)
            del specs_by_name[qualified]
            del instances[qualified]
            offset_spec = ParameterSpec(
                name=offset_name,
                value=offset_value,
                min=0.0,
                max=None,
                vary=True,
                description=f"offset enforcing {qualified} {op} {reference}",
            )
            specs.append(offset_spec)
            specs_by_name[offset_name] = offset_spec
            instances[offset_name] = ParameterInstance(
                name=offset_name,
                component=component.name,
                parameter=f"{parameter}{CONSTRAINT_OFFSET_SUFFIX}",
                scope="global",
                datasets=tuple(applicable[component.name]),
            )
            derived.append(
                DerivedParameter(name=qualified, base=base, offset=offset_name, sign=sign)
            )
    return derived
