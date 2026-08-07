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
    or the qualified name of another global parameter. Exact constraints use
    ``{"parameter": p, "op": "=", "expression": "..."}``; the dependent
    parameter is removed from the optimizer and derived from the expression.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any

import numpy as np

from .cross_section import (
    MILLIBARN_PER_BARN,
    bose_denominator,
    cross_section_from_chipp,
    intensity_from_chipp,
    kf_over_ki,
    magnetic_moment_factor,
    quasistatic_cross_section_from_chi,
)
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
    parameter_expression_names,
)
from .heat_capacity import debye_heat_capacity, low_temperature_heat_capacity
from .magnetization import curie_weiss_susceptibility
from .model_registry import MODEL_TYPE_REGISTRY, validate_model_component
from .quantities import convert_quantity
from .spin_fluctuations import (
    build_rpa_geometry,
    conserved_ferromagnetic_susceptibility,
    generalized_paramagnon_chipp,
    generalized_paramagnon_susceptibility,
    heisenberg_rpa_chipp,
    heisenberg_rpa_chipp_and_gradients,
    heisenberg_rpa_susceptibility,
    local_relaxational_chipp,
    local_relaxational_susceptibility,
    mmp_chipp,
    mmp_susceptibility,
    paramagnon_spatial_kernel,
    reduce_site_network,
    reduce_site_network_with_tensors,
    rpa_exchange_matrix,
)
from .susceptibility import (
    ScalarSusceptibilityResponse,
    coupled_scalar_susceptibility,
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


def _point_temperature(data: PointData4D) -> np.ndarray:
    if data.temperature is None:
        raise ValueError("model requires a per-point temperature axis")
    if np.isscalar(data.temperature):
        return np.full(data.size, float(data.temperature), dtype=float)
    return np.asarray(data.temperature, dtype=float)


def _heat_capacity_observable(data: PointData4D, molar_heat_capacity: np.ndarray) -> np.ndarray:
    quantity = str(data.metadata.get("quantity_type", "heat_capacity"))
    unit = str(data.metadata.get("unit", "mJ/(mol K)"))
    values = np.asarray(molar_heat_capacity, dtype=float)
    if quantity == "heat_capacity_over_temperature":
        values = values / _point_temperature(data)
        if unit not in {"", "mJ/(mol K^2)"}:
            raise ValueError(f"heat-capacity-over-temperature model cannot output {unit!r}")
        return values
    if quantity != "heat_capacity":
        raise ValueError("heat-capacity model requires C or C/T data")
    if unit in {"", "mJ/(mol K)"}:
        return values
    return convert_quantity(values, "heat_capacity", "mJ/(mol K)", unit)


def _debye_heat_capacity_factory(component: Any) -> ModelFunction:
    theta_key = qualified_parameter_name(component.name, "debye_temperature")
    count_key = qualified_parameter_name(component.name, "oscillator_count")

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        values = debye_heat_capacity(
            _point_temperature(data),
            float(params[theta_key]),
            float(params[count_key]),
        )
        return _heat_capacity_observable(data, values)

    return model


def _low_temperature_heat_capacity_factory(component: Any) -> ModelFunction:
    gamma_key = qualified_parameter_name(component.name, "sommerfeld_gamma")
    beta_key = qualified_parameter_name(component.name, "debye_beta")

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        values = low_temperature_heat_capacity(
            _point_temperature(data), float(params[gamma_key]), float(params[beta_key])
        )
        return _heat_capacity_observable(data, values)

    return model


def _curie_weiss_factory(component: Any) -> ModelFunction:
    curie_key = qualified_parameter_name(component.name, "curie_constant")
    theta_key = qualified_parameter_name(component.name, "theta_CW")

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        if data.metadata.get("quantity_type") != "bulk_susceptibility":
            raise ValueError("Curie-Weiss model requires a bulk-susceptibility channel")
        temperature = _point_temperature(data)
        values = curie_weiss_susceptibility(
            temperature,
            float(params[curie_key]),
            float(params[theta_key]),
        )
        unit = str(data.metadata.get("unit", "cm^3/mol"))
        if unit == "cm^3/mol":
            return values
        return convert_quantity(values, "bulk_susceptibility", "cm^3/mol", unit)

    return model


def _electronic_structure_factory(component: Any) -> ModelFunction:
    """Return a guard for a calculation-only electronic-structure component."""

    def model(_data: PointData4D, _params: dict[str, float]) -> np.ndarray:
        raise ValueError(
            f"{component.name!r} calculates electronic structure but does not "
            "produce a dataset observable until an electronic-response model is added"
        )

    return model


def _lindhard_unbound_factory(component: Any) -> ModelFunction:
    """Guard direct construction when the sibling-component context is absent."""

    def model(_data: PointData4D, _params: dict[str, float]) -> np.ndarray:
        raise ValueError(
            f"{component.name!r} requires its referenced tight-binding component"
        )

    return model


def _lindhard_factory(
    component: Any,
    components: Mapping[str, Any],
    *,
    dressing_component: Any | None = None,
    dressing_kind: str = "bare",
) -> ModelFunction:
    """Build a bare or interaction-dressed response evaluator."""

    from .electronic_backends import validate_electronic_backend
    from .electronic_interactions import (
        correlated_basis_indices,
        hubbard_hund_spin_vertex,
        matrix_interaction_vertex,
        project_implicit_spin_response,
        rpa_dress_implicit_spin_probe_stoner,
        rpa_dress_susceptibility,
        scalar_stoner_vertex,
    )
    from .electronic_normalization import (
        resolve_electronic_response_normalization,
    )
    from .electronic_response import (
        ElectronicResponseCache,
        _interpolation_certificate,
        bare_lindhard_susceptibility,
        bare_spin_susceptibility,
        chemical_potential_for_filling,
        has_orbital_magnetic_form_factors,
        implicit_spin_probe_susceptibility,
        neutron_spin_contraction,
        orbital_pair_operator_basis,
        response_k_mesh,
    )
    from .electronic_structure import electronic_model_from_component, k_mesh

    config = component.config if isinstance(component.config, dict) else {}
    source_name = str(config.get("electronic_component", "")).strip()
    source = components.get(source_name) if source_name else None
    if source is None and not source_name:
        candidates = [
            item
            for item in components.values()
            if getattr(item, "type", None) == "tight_binding"
            and bool(getattr(item, "enabled", True))
        ]
        source = candidates[0] if len(candidates) == 1 else None
    if source is None or getattr(source, "type", None) != "tight_binding":
        raise ValueError(
            f"Lindhard component {component.name!r} must reference an enabled "
            "tight-binding component; an empty reference is automatic only "
            "when exactly one compatible component is enabled"
        )
    base_model = electronic_model_from_component(source)
    mesh = k_mesh(
        base_model,
        config.get("response_mesh", [16, 16, 16]),
        shift=config.get("response_mesh_shift", [0.5, 0.5, 0.5]),
        symmetry="full",
    )
    eta_key = qualified_parameter_name(component.name, "broadening")
    source_parameter_keys = {
        parameter: qualified_parameter_name(source.name, parameter)
        for parameter in tight_binding_parameter_names(source)
    }
    backend = str(config.get("response_backend", "auto"))
    workers = int(config.get("response_workers", 0))
    batch_bytes = int(
        float(config.get("response_max_batch_mb", 256.0)) * 1024**2
    )
    transition_batch_bytes = int(
        float(config.get("response_transition_max_batch_mb", 256.0))
        * 1024**2
    )
    transition_backend = str(
        config.get("response_transition_backend", "auto")
    )
    response_cache = ElectronicResponseCache(
        max_bytes=int(float(config.get("response_cache_mb", 512.0)) * 1024**2),
        max_entries=int(config.get("response_cache_entries", 64)),
    )
    powder_orientations = int(config.get("powder_orientations", 50))
    certificate_lock = RLock()
    normalization_cache: dict[tuple[str, str], Any] = {}
    dressing_config = (
        dressing_component.config
        if dressing_component is not None
        and isinstance(dressing_component.config, dict)
        else {}
    )
    dressing_keys = (
        {
            name: qualified_parameter_name(dressing_component.name, name)
            for name in component_parameter_names(dressing_component)
        }
        if dressing_component is not None
        else {}
    )
    # Pole and stability diagnostics are configured identically for every
    # dressing kind, so resolve them once instead of at each call site.
    dressing_diagnostics = {
        "singular_tolerance": float(
            dressing_config.get("singular_tolerance", 1.0e-12)
        ),
        "near_pole_tolerance": float(
            dressing_config.get("near_pole_tolerance", 1.0e-3)
        ),
        "static_warning_margin": float(
            dressing_config.get("static_stability_warning_margin", 0.05)
        ),
        "reject_sampled_static_instability": bool(
            dressing_config.get("reject_sampled_static_instability", False)
        ),
    }
    validated_digests: set[str] = set()

    def resolved_model(params: Mapping[str, float]):
        values = {
            parameter: float(params[key])
            for parameter, key in source_parameter_keys.items()
        }
        resolved = (
            base_model.with_parameters(**values, energy_unit="meV")
            if values
            else base_model
        )
        if (
            bool(config.get("response_validate_backend", True))
            and backend != "numpy"
            and resolved.content_digest not in validated_digests
        ):
            count = min(
                int(config.get("response_backend_probe_points", 8)),
                mesh.reduced_coordinates.shape[0],
            )
            probe_indices = np.linspace(
                0,
                mesh.reduced_coordinates.shape[0] - 1,
                count,
                dtype=int,
            )
            validation = validate_electronic_backend(
                resolved,
                mesh.reduced_coordinates[probe_indices],
                backend,
                workers=workers,
                max_batch_bytes=batch_bytes,
                relative_tolerance=float(
                    config.get("response_backend_rtol", 1.0e-10)
                ),
                absolute_tolerance_meV=float(
                    config.get("response_backend_atol_meV", 1.0e-8)
                ),
                require_requested_backend=True,
            )
            if not validation.passed:
                raise RuntimeError(
                    "electronic response backend validation failed: "
                    f"{validation.reason}; max absolute error "
                    f"{validation.max_absolute_error_meV:g} meV"
                )
            validated_digests.add(resolved.content_digest)
        return resolved

    def chemical_potential(model: Any, temperature: float) -> float:
        mode = str(config.get("chemical_potential_mode", "source"))
        if mode == "source":
            return float(source.config.get("chemical_potential_meV", 0.0))
        return chemical_potential_for_filling(
            model,
            mesh,
            float(config.get("filling_per_cell", 1.0)),
            temperature_K=temperature,
            backend=backend,
            workers=workers,
            max_batch_bytes=batch_bytes,
            cache=response_cache,
        )

    def response_normalization(
        _model: Any,
        data: PointData4D,
        *,
        bulk: bool = False,
    ):
        convention = data.metadata.get("spectral_observable")
        if bulk:
            target_basis = "per_formula_unit"
            target_label = "f.u."
        elif isinstance(convention, Mapping):
            target_basis = str(
                convention.get("normalization_basis", "per_model_cell")
            )
            target_label = str(convention.get("normalization_label", "") or "")
        else:
            # Legacy fit points did not declare a normalization basis. Preserve
            # their historical model-cell ordinate rather than guessing.
            target_basis = "per_model_cell"
            target_label = ""
        key = (target_basis, target_label)
        with certificate_lock:
            cached = normalization_cache.get(key)
            if cached is not None:
                return cached
            resolved = resolve_electronic_response_normalization(
                base_model,
                config,
                crystal=(
                    source.config.get("crystal")
                    if isinstance(source.config, Mapping)
                    else None
                ),
                target_basis=target_basis,
                target_label=target_label,
            )
            normalization_cache[key] = resolved
            return resolved

    def response_at_points(
        model: Any,
        q_reduced: np.ndarray,
        energy: np.ndarray,
        temperature: np.ndarray,
        eta: float,
        params: Mapping[str, float],
    ) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
        def certify_dressed_response(
            bare_result: Any,
            dressed_result: Any,
            vertex: Any,
            evaluation_mesh: Any,
            selected_q: np.ndarray,
            selected_energy: np.ndarray,
            common: Mapping[str, Any],
            *,
            direct_bare_evaluator: Callable[..., Any] = bare_spin_susceptibility,
            dressing_evaluator: Callable[[Any], Any] | None = None,
        ) -> Any:
            certificate = bare_result.provenance.get(
                "q_interpolation_certificate"
            )
            if (
                not isinstance(certificate, Mapping)
                or certificate.get("status") != "certified"
            ):
                return dressed_result
            evaluation = bare_result.provenance.get("q_evaluation", {})
            validation_indices = np.asarray(
                evaluation.get("validation_points", ()),
                dtype=np.intp,
            )
            if validation_indices.size == 0:
                return dressed_result
            direct_settings = {
                **dict(common),
                "q_evaluation": "direct",
                "q_interpolation_rtol": 0.0,
                "q_interpolation_atol": 0.0,
                "q_interpolation_mesh": None,
            }
            direct_bare = direct_bare_evaluator(
                model,
                selected_q[validation_indices],
                selected_energy[validation_indices],
                evaluation_mesh,
                **direct_settings,
            )
            direct_dressed = (
                dressing_evaluator(direct_bare)
                if dressing_evaluator is not None
                else rpa_dress_susceptibility(
                    direct_bare,
                    vertex,
                    **dressing_diagnostics,
                )
            )
            interpolated_values = dressed_result.values_per_meV_cell[
                validation_indices
            ]
            direct_values = direct_dressed.values_per_meV_cell
            difference = np.abs(interpolated_values - direct_values)
            rtol = float(certificate.get("relative_tolerance", 0.0))
            atol = float(certificate.get("absolute_tolerance", 0.0))
            scale = max(
                float(np.max(np.abs(direct_values))),
                atol,
                np.finfo(float).tiny,
            )
            maximum_absolute = float(np.max(difference))
            maximum_relative = float(maximum_absolute / scale)
            passed = bool(np.all(difference <= atol + rtol * scale))
            updated = {
                key: copy.deepcopy(value)
                for key, value in certificate.items()
                if key != "certificate_digest"
            }
            updated.update(
                {
                    "certified_quantity": dressed_result.response_kind,
                    "bare_certificate_digest": certificate.get(
                        "certificate_digest"
                    ),
                    "bare_maximum_absolute_error": certificate.get(
                        "maximum_absolute_error"
                    ),
                    "bare_maximum_relative_error": certificate.get(
                        "maximum_relative_error"
                    ),
                    "maximum_absolute_error": maximum_absolute,
                    "maximum_relative_error": maximum_relative,
                    "dressed_validation_passed": passed,
                }
            )
            if passed:
                dressed_certificate = _interpolation_certificate(updated)
                return replace(
                    dressed_result,
                    provenance={
                        **dict(dressed_result.provenance),
                        "q_interpolation_certificate": dressed_certificate,
                    },
                )
            policy = str(common.get("q_evaluation", "auto"))
            if policy == "interpolated":
                raise ValueError(
                    "interaction-dressed q interpolation did not satisfy the "
                    "requested tolerance; maximum absolute error "
                    f"{maximum_absolute:g}, maximum relative error "
                    f"{maximum_relative:g}"
                )
            exact_bare = direct_bare_evaluator(
                model,
                selected_q,
                selected_energy,
                evaluation_mesh,
                **direct_settings,
            )
            exact_dressed = (
                dressing_evaluator(exact_bare)
                if dressing_evaluator is not None
                else rpa_dress_susceptibility(
                    exact_bare,
                    vertex,
                    **dressing_diagnostics,
                )
            )
            updated.update(
                {
                    "status": "exact_fallback",
                    "dressed_validation_passed": False,
                }
            )
            fallback_certificate = _interpolation_certificate(updated)
            return replace(
                exact_dressed,
                provenance={
                    **dict(exact_dressed.provenance),
                    "q_interpolation_certificate": fallback_certificate,
                    "q_evaluation": {
                        **dict(evaluation),
                        "resolved_policy": "exact_fallback",
                        "interpolated_points": 0,
                        "direct_off_mesh_points": int(selected_q.shape[0]),
                    },
                },
            )

        tensor = np.empty((q_reduced.shape[0], 3, 3), dtype=np.complex128)
        certificates: list[dict[str, Any]] = []
        for value in np.unique(temperature):
            selected = np.flatnonzero(temperature == value)
            mu = chemical_potential(model, float(value))
            evaluation_mesh = response_k_mesh(
                model,
                config.get("response_mesh", [16, 16, 16]),
                q_reduced[selected],
                shift=config.get("response_mesh_shift", [0.5, 0.5, 0.5]),
                symmetry=str(config.get("response_symmetry", "auto")),
                operator_kind=(
                    "orbital_pair_matrix"
                    if dressing_kind == "hubbard_hund"
                    else "implicit_isotropic_spin"
                ),
            )
            common = {
                "temperature_K": float(value),
                "chemical_potential_meV": mu,
                "broadening_meV": eta,
                "backend": backend,
                "workers": workers,
                "max_batch_bytes": batch_bytes,
                "transition_max_batch_bytes": transition_batch_bytes,
                "transition_backend": transition_backend,
                "cache": response_cache,
                "q_evaluation": str(
                    config.get("response_q_evaluation", "auto")
                ),
                "q_interpolation_rtol": float(
                    config.get("response_q_interpolation_rtol", 0.0)
                ),
                "q_interpolation_atol": float(
                    config.get("response_q_interpolation_atol", 0.0)
                ),
                "q_interpolation_mesh": (
                    None
                    if not config.get("response_q_interpolation_mesh", [])
                    else config.get("response_q_interpolation_mesh")
                ),
                "q_validation_points": int(
                    config.get("response_q_validation_points", 8)
                ),
            }
            if dressing_kind == "hubbard_hund":
                shells = dressing_config.get("correlated_shells", [])
                if isinstance(shells, str):
                    shells = [
                        item.strip() for item in shells.split(",") if item.strip()
                    ]
                basis_indices = correlated_basis_indices(model, shells)
                operators = orbital_pair_operator_basis(model, list(basis_indices))
                bare = bare_lindhard_susceptibility(
                    model,
                    q_reduced[selected],
                    energy[selected],
                    evaluation_mesh,
                    operators,
                    **common,
                )
                vertex = hubbard_hund_spin_vertex(
                    model,
                    basis_indices,
                    U=float(params[dressing_keys["U"]]),
                    U_prime=float(params[dressing_keys["U_prime"]]),
                    J_H=float(params[dressing_keys["J_H"]]),
                    J_pair=float(params[dressing_keys["J_pair"]]),
                    rotationally_invariant=bool(
                        dressing_config.get("rotationally_invariant", True)
                    ),
                    energy_unit="eV",
                )
                result = project_implicit_spin_response(
                    rpa_dress_susceptibility(bare, vertex, **dressing_diagnostics),
                    model,
                    basis_indices,
                )
            else:
                profiled_probe = has_orbital_magnetic_form_factors(model)
                if profiled_probe and model.spin_operators is None:
                    bare_result = implicit_spin_probe_susceptibility(
                        model,
                        q_reduced[selected],
                        energy[selected],
                        evaluation_mesh,
                        **common,
                    )
                else:
                    bare_result = bare_spin_susceptibility(
                        model,
                        q_reduced[selected],
                        energy[selected],
                        evaluation_mesh,
                        **common,
                    )
                result = bare_result
                if profiled_probe and model.spin_operators is not None and dressing_kind != "bare":
                    raise ValueError(
                        "orbital form factors with RPA currently require an "
                        "implicit-spin tight-binding model"
                    )
                if profiled_probe and model.spin_operators is None and dressing_kind == "bare":
                    scalar = bare_result.values_per_meV_cell[:, 0, 0]
                    isotropic = np.zeros(
                        (scalar.size, 3, 3), dtype=np.complex128
                    )
                    isotropic[:, np.arange(3), np.arange(3)] = scalar[:, None]
                    result = replace(
                        bare_result,
                        values_per_meV_cell=isotropic,
                        operator_labels=("Sx", "Sy", "Sz"),
                        conjugate_indices=(0, 1, 2),
                    )
                elif profiled_probe and model.spin_operators is None and dressing_kind == "stoner":
                    interaction = float(params[dressing_keys["I"]])
                    vertex = matrix_interaction_vertex(
                        bare_result.operator_labels,
                        np.asarray(((0.0, 0.0), (0.0, 2.0 * interaction))),
                        energy_unit="eV",
                        channel="spin",
                    )

                    def dress_probe_stoner(
                        response: Any,
                        interaction_value: float = interaction,
                    ) -> Any:
                        return rpa_dress_implicit_spin_probe_stoner(
                            response,
                            interaction_value,
                            energy_unit="eV",
                            **dressing_diagnostics,
                        )

                    dressed_mixed = dress_probe_stoner(bare_result)
                    dressed_mixed = certify_dressed_response(
                        bare_result,
                        dressed_mixed,
                        vertex,
                        evaluation_mesh,
                        q_reduced[selected],
                        energy[selected],
                        common,
                        direct_bare_evaluator=implicit_spin_probe_susceptibility,
                        dressing_evaluator=dress_probe_stoner,
                    )
                    scalar = dressed_mixed.values_per_meV_cell[:, 0, 0]
                    isotropic = np.zeros(
                        (scalar.size, 3, 3), dtype=np.complex128
                    )
                    isotropic[:, np.arange(3), np.arange(3)] = scalar[:, None]
                    result = replace(
                        dressed_mixed,
                        values_per_meV_cell=isotropic,
                        operator_labels=("Sx", "Sy", "Sz"),
                        conjugate_indices=(0, 1, 2),
                        response_kind="rpa_scalar_stoner_orbital_probe",
                    )
                elif dressing_kind == "stoner":
                    vertex = scalar_stoner_vertex(
                        result.operator_labels,
                        float(params[dressing_keys["I"]]),
                        energy_unit="eV",
                    )
                    result = rpa_dress_susceptibility(
                        result, vertex, **dressing_diagnostics
                    )
                    result = certify_dressed_response(
                        bare_result,
                        result,
                        vertex,
                        evaluation_mesh,
                        q_reduced[selected],
                        energy[selected],
                        common,
                    )
                elif profiled_probe and model.spin_operators is None and dressing_kind == "matrix":
                    matrix = np.asarray(
                        dressing_config.get("vertex_matrix", np.eye(3)),
                        dtype=np.complex128,
                    )
                    vertex = matrix_interaction_vertex(
                        ("Sx", "Sy", "Sz"),
                        float(params[dressing_keys["scale"]]) * matrix,
                        energy_unit="eV",
                        channel=str(dressing_config.get("channel", "spin")),
                    )

                    def dress_probe_matrix(
                        response: Any,
                        interaction_vertex: Any = vertex,
                    ) -> Any:
                        gamma = np.asarray(interaction_vertex.values_meV)
                        mixed = response.values_per_meV_cell
                        probe = mixed[:, 0, 0]
                        probe_total = mixed[:, 0, 1]
                        total_probe = mixed[:, 1, 0]
                        total = mixed[:, 1, 1]
                        identity = np.eye(3, dtype=np.complex128)
                        denominator = (
                            identity[None, ...] - total[:, None, None] * gamma
                        )
                        dressed_vertex = np.linalg.solve(
                            denominator,
                            np.broadcast_to(gamma, denominator.shape),
                        )
                        isotropic = probe[:, None, None] * identity[None, ...]
                        isotropic = isotropic + (
                            probe_total * total_probe
                        )[:, None, None] * dressed_vertex
                        return replace(
                            response,
                            values_per_meV_cell=isotropic,
                            operator_labels=("Sx", "Sy", "Sz"),
                            conjugate_indices=(0, 1, 2),
                            response_kind="rpa_matrix_orbital_probe",
                            provenance={
                                **dict(response.provenance),
                                "interaction": interaction_vertex.to_dict(),
                                "dressing": {
                                    "equation": (
                                        "chi_PP + chi_PS Gamma "
                                        "(I-chi_SS Gamma)^-1 chi_SP"
                                    ),
                                    "form_factor_outside_denominator": True,
                                },
                            },
                        )

                    result = dress_probe_matrix(bare_result)
                    result = certify_dressed_response(
                        bare_result,
                        result,
                        vertex,
                        evaluation_mesh,
                        q_reduced[selected],
                        energy[selected],
                        common,
                        direct_bare_evaluator=implicit_spin_probe_susceptibility,
                        dressing_evaluator=dress_probe_matrix,
                    )
                elif dressing_kind == "matrix":
                    matrix = np.asarray(
                        dressing_config.get("vertex_matrix", np.eye(3)),
                        dtype=np.complex128,
                    )
                    vertex = matrix_interaction_vertex(
                        result.operator_labels,
                        float(params[dressing_keys["scale"]]) * matrix,
                        energy_unit="eV",
                        channel=str(dressing_config.get("channel", "spin")),
                    )
                    result = rpa_dress_susceptibility(
                        result, vertex, **dressing_diagnostics
                    )
                    result = certify_dressed_response(
                        bare_result,
                        result,
                        vertex,
                        evaluation_mesh,
                        q_reduced[selected],
                        energy[selected],
                        common,
                    )
            tensor[selected] = result.values_per_meV_cell
            certificate = result.provenance.get(
                "q_interpolation_certificate"
            )
            if isinstance(certificate, Mapping):
                certificates.append(
                    {
                        **copy.deepcopy(dict(certificate)),
                        "temperature_K": float(value),
                    }
                )
        return tensor, tuple(certificates)

    def record_interpolation_certificates(
        data: PointData4D,
        params: Mapping[str, float],
        certificates: tuple[dict[str, Any], ...],
    ) -> None:
        """Persist certificates only for the component's current live state."""

        current_values = {
            **{
                key: float(source.parameters[parameter])
                for parameter, key in source_parameter_keys.items()
            },
            eta_key: float(component.parameters.get("broadening", 1.0)),
            **{
                key: float(dressing_component.parameters[name])
                for name, key in dressing_keys.items()
            },
        }
        if any(
            key not in params
            or not np.isclose(
                float(params[key]),
                value,
                rtol=1.0e-13,
                atol=1.0e-13,
            )
            for key, value in current_values.items()
        ):
            return
        dataset_name = str(
            data.metadata.get("fit_dataset_name", "dataset")
        )
        with certificate_lock:
            stored = component.config.get(
                "response_q_interpolation_certificates",
                {},
            )
            by_dataset = (
                copy.deepcopy(dict(stored))
                if isinstance(stored, Mapping)
                else {}
            )
            if certificates:
                by_dataset[dataset_name] = {
                    "status": (
                        "certified"
                        if all(
                            item.get("status") == "certified"
                            for item in certificates
                        )
                        else "exact_fallback"
                    ),
                    "certificates": list(certificates),
                }
            else:
                by_dataset.pop(dataset_name, None)
            component.config[
                "response_q_interpolation_certificates"
            ] = by_dataset

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        electronic_model = resolved_model(params)
        eta = float(params[eta_key])
        temperatures = np.broadcast_to(
            np.asarray(_dataset_temperature(data), dtype=float),
            (data.size,),
        )
        data_type = str(data.metadata.get("data_type", ""))
        bulk = data_type == "magnetization"
        powder = data_type in {"powder_inelastic", "powder_elastic"} or bool(
            data.metadata.get("powder_q_modulus_axis")
        )
        if bulk:
            q_reduced = np.zeros((data.size, 3), dtype=float)
            tensor, certificates = response_at_points(
                electronic_model,
                q_reduced,
                np.zeros(data.size),
                temperatures,
                eta,
                params,
            )
            record_interpolation_certificates(data, params, certificates)
            chi = np.asarray(
                np.trace(tensor.real, axis1=1, axis2=2) / 3.0,
                dtype=float,
            )
            chi = response_normalization(
                electronic_model,
                data,
                bulk=True,
            ).apply(chi)
            return _scalar_bulk_observable(
                data,
                chi,
                g_factor=float(config.get("bulk_g_factor", 2.0)),
                magnetic_ions_per_formula_unit=1.0,
            )

        if powder:
            from .fitting import q_modulus_inv_angstrom

            directions = _powder_sphere_directions(powder_orientations)
            q_modulus = np.asarray(q_modulus_inv_angstrom(data), dtype=float)
            q_cartesian = (
                q_modulus[:, None, None] * directions[None, :, :]
            ).reshape(-1, 3)
            q_reduced = q_cartesian @ np.linalg.inv(
                electronic_model.reciprocal_lattice
            ).T
            repeated_energy = np.repeat(
                np.zeros(data.size) if _is_elastic_dataset(data) else data.E,
                powder_orientations,
            )
            repeated_temperature = np.repeat(temperatures, powder_orientations)
            tensor, certificates = response_at_points(
                electronic_model,
                q_reduced,
                repeated_energy,
                repeated_temperature,
                eta,
                params,
            )
            record_interpolation_certificates(data, params, certificates)
            from .electronic_response import SusceptibilityResult

            shell_response = SusceptibilityResult(
                q_reduced=q_reduced,
                energy_meV=repeated_energy,
                values_per_meV_cell=tensor,
                operator_labels=("Sx", "Sy", "Sz"),
                conjugate_indices=(0, 1, 2),
                model_digest=electronic_model.content_digest,
                temperature_K=float(repeated_temperature[0]),
                chemical_potential_meV=chemical_potential(
                    electronic_model,
                    float(repeated_temperature[0]),
                ),
                broadening_meV=eta,
            )
            contracted = neutron_spin_contraction(
                shell_response,
                electronic_model.reciprocal_lattice,
            ).reshape(data.size, powder_orientations).mean(axis=1)
        else:
            coordinates = np.column_stack((data.H, data.K, data.L)).astype(float)
            if _metadata_coordinate_units_are_inv_angstrom(data.metadata):
                q_cartesian = coordinates
                q_reduced = q_cartesian @ np.linalg.inv(
                    electronic_model.reciprocal_lattice
                ).T
            else:
                matrix = data.metadata.get("rlu_to_inv_angstrom_matrix")
                if matrix is None:
                    q_reduced = coordinates
                else:
                    q_cartesian = coordinates @ np.asarray(matrix, dtype=float).T
                    q_reduced = q_cartesian @ np.linalg.inv(
                        electronic_model.reciprocal_lattice
                    ).T
            point_energy = (
                np.zeros(data.size)
                if _is_elastic_dataset(data)
                else np.asarray(data.E)
            )
            tensor, certificates = response_at_points(
                electronic_model,
                q_reduced,
                point_energy,
                temperatures,
                eta,
                params,
            )
            record_interpolation_certificates(data, params, certificates)
            from .electronic_response import SusceptibilityResult

            response = SusceptibilityResult(
                q_reduced=q_reduced,
                energy_meV=point_energy,
                values_per_meV_cell=tensor,
                operator_labels=("Sx", "Sy", "Sz"),
                conjugate_indices=(0, 1, 2),
                model_digest=electronic_model.content_digest,
                temperature_K=float(temperatures[0]),
                chemical_potential_meV=chemical_potential(
                    electronic_model,
                    float(temperatures[0]),
                ),
                broadening_meV=eta,
            )
            contracted = neutron_spin_contraction(
                response,
                electronic_model.reciprocal_lattice,
            )

        contracted = response_normalization(
            electronic_model,
            data,
        ).apply(contracted)

        if _is_elastic_dataset(data):
            return _quasistatic_model_observable(
                data,
                np.asarray(contracted.real),
                form_factor_sq=1.0,
                polarization=1.0,
            )
        return _spectral_model_observable(
            data,
            np.asarray(contracted.imag),
            form_factor_sq=1.0,
            polarization=1.0,
        )

    if dressing_kind == "bare":
        return model

    def stable_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        try:
            return model(data, params)
        except np.linalg.LinAlgError:
            # RPA trial parameters can cross a sampled pole while an optimizer
            # explores. Match the Heisenberg-RPA behavior by returning a large
            # finite residual target instead of aborting the fit.
            return np.full(data.size, 1.0e6, dtype=float)

    return stable_model


def _rpa_unbound_factory(component: Any) -> ModelFunction:
    """Guard direct construction when the bare-response context is absent."""

    def model(_data: PointData4D, _params: dict[str, float]) -> np.ndarray:
        raise ValueError(
            f"{component.name!r} requires its referenced Lindhard component"
        )

    return model


def _rpa_factory(
    component: Any,
    components: Mapping[str, Any],
    *,
    dressing_kind: str,
) -> ModelFunction:
    config = component.config if isinstance(component.config, dict) else {}
    response_name = str(config.get("response_component", "")).strip()
    response = components.get(response_name)
    if response is None or getattr(response, "type", None) != "lindhard":
        raise ValueError(
            f"RPA component {component.name!r} must reference an enabled "
            "Lindhard component"
        )
    return _lindhard_factory(
        response,
        components,
        dressing_component=component,
        dressing_kind=dressing_kind,
    )


def _stoner_rpa_factory(
    component: Any,
    components: Mapping[str, Any],
) -> ModelFunction:
    return _rpa_factory(component, components, dressing_kind="stoner")


def _matrix_rpa_factory(
    component: Any,
    components: Mapping[str, Any],
) -> ModelFunction:
    return _rpa_factory(component, components, dressing_kind="matrix")


def _hubbard_hund_rpa_factory(
    component: Any,
    components: Mapping[str, Any],
) -> ModelFunction:
    return _rpa_factory(component, components, dressing_kind="hubbard_hund")


def _coupled_susceptibility_factory(
    component: Any,
    components: Mapping[str, Any],
) -> ModelFunction:
    """Build a bilinearly coupled pair of complex scalar responses."""

    config = component.config if isinstance(component.config, dict) else {}
    names = (
        str(config.get("response_a", "")).strip(),
        str(config.get("response_b", "")).strip(),
    )
    sources = []
    providers = []
    for name in names:
        source = components.get(name)
        if source is None:
            raise ValueError(
                f"coupled susceptibility {component.name!r} references missing "
                f"component {name!r}"
            )
        definition = MODEL_TYPE_REGISTRY.get(str(source.type))
        if definition is None or definition.susceptibility_factory is None:
            raise ValueError(
                f"component {name!r} of type {source.type!r} does not expose "
                "a composable complex susceptibility"
            )
        sources.append(source)
        providers.append(definition.susceptibility_factory(source, components))
    coupling_key = qualified_parameter_name(component.name, "coupling")
    tolerance = float(config.get("singular_tolerance", 1.0e-12))

    def complex_response(
        data: PointData4D, params: Mapping[str, float]
    ) -> ScalarSusceptibilityResponse:
        response_data = (
            data.with_updates(E=np.zeros(data.size, dtype=float))
            if _is_elastic_dataset(data)
            else data
        )
        response_a = providers[0](response_data, params)
        response_b = providers[1](response_data, params)
        chi = coupled_scalar_susceptibility(
            response_a.chi,
            response_b.chi,
            coupling=float(params[coupling_key]),
            amplitude_a=response_a.form_factor,
            amplitude_b=response_b.form_factor,
            singular_tolerance=tolerance,
        )
        return ScalarSusceptibilityResponse(
            chi,
            np.ones(data.size, dtype=float),
            f"{sources[0].name}<->{sources[1].name}",
        )

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        try:
            response = complex_response(data, params).chi
        except (ValueError, np.linalg.LinAlgError):
            return np.full(data.size, 1.0e6, dtype=float)
        if _is_elastic_dataset(data):
            return _quasistatic_model_observable(
                data,
                response.real,
                form_factor_sq=1.0,
                polarization=ISOTROPIC_POLARIZATION,
            )
        return _spectral_model_observable(
            data,
            response.imag,
            form_factor_sq=1.0,
            polarization=ISOTROPIC_POLARIZATION,
        )

    model.susceptibility_response = complex_response  # type: ignore[attr-defined]
    return model


def _coupled_susceptibility_unbound_factory(component: Any) -> ModelFunction:
    def model(_data: PointData4D, _params: dict[str, float]) -> np.ndarray:
        raise ValueError(
            f"{component.name!r} requires its two referenced susceptibility components"
        )

    return model


def _coupled_susceptibility_provider_factory(
    component: Any,
    components: Mapping[str, Any],
) -> Callable[[PointData4D, Mapping[str, float]], ScalarSusceptibilityResponse]:
    model = _coupled_susceptibility_factory(component, components)
    return model.susceptibility_response  # type: ignore[attr-defined, no-any-return]


def _validate_coupled_susceptibility_component(component: Any) -> None:
    config = component.config if isinstance(component.config, dict) else {}
    response_a = str(config.get("response_a", "")).strip()
    response_b = str(config.get("response_b", "")).strip()
    if not response_a or not response_b:
        raise ValueError("response_a and response_b must name response components")
    if response_a == response_b:
        raise ValueError("response_a and response_b must be different components")
    tolerance = float(config.get("singular_tolerance", 1.0e-12))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("singular_tolerance must be finite and positive")


def _validate_electronic_sampling_config(
    config: Mapping[str, Any],
    prefix: str,
) -> None:
    mode = str(config.get(f"{prefix}_sampling_mode", "automatic"))
    if mode not in {"automatic", "manual"}:
        raise ValueError(f"{prefix}_sampling_mode must be automatic or manual")
    accuracy = str(config.get(f"{prefix}_sampling_accuracy", "standard"))
    if accuracy not in {"preview", "standard", "high", "custom"}:
        raise ValueError(
            f"{prefix}_sampling_accuracy must be preview, standard, high, or custom"
        )
    custom = float(config.get(f"{prefix}_sampling_custom_rtol", 0.01))
    if not np.isfinite(custom) or custom <= 0.0:
        raise ValueError(
            f"{prefix}_sampling_custom_rtol must be finite and positive"
        )
    if int(config.get(f"{prefix}_sampling_max_refinements", 7)) < 3:
        raise ValueError(
            f"{prefix}_sampling_max_refinements must be at least three"
        )
    if int(config.get(f"{prefix}_sampling_max_mesh_points", 1)) < 1:
        raise ValueError(
            f"{prefix}_sampling_max_mesh_points must be positive"
        )
    certificate = config.get(f"{prefix}_sampling_certificate", {})
    if not isinstance(certificate, Mapping):
        raise ValueError(f"{prefix}_sampling_certificate must be a mapping")


def _validate_lindhard_component(component: Any) -> None:
    config = component.config if isinstance(component.config, dict) else {}
    _validate_electronic_sampling_config(config, "response")
    if int(config.get("response_sampling_points_per_dataset", 32)) < 1:
        raise ValueError(
            "response_sampling_points_per_dataset must be positive"
        )
    mesh = tuple(int(value) for value in config.get("response_mesh", ()))
    if len(mesh) not in {1, 2, 3} or any(value < 1 for value in mesh):
        raise ValueError("response_mesh must contain one to three positive sizes")
    shift = tuple(float(value) for value in config.get("response_mesh_shift", ()))
    if len(shift) not in {1, 2, 3} or any(not np.isfinite(value) for value in shift):
        raise ValueError("response_mesh_shift must contain finite mesh offsets")
    if str(config.get("response_symmetry", "auto")) not in {
        "auto",
        "full",
        "reduced",
    }:
        raise ValueError("response_symmetry must be auto, full, or reduced")
    if str(config.get("response_q_evaluation", "auto")) not in {
        "auto",
        "direct",
        "commensurate",
        "interpolated",
    }:
        raise ValueError(
            "response_q_evaluation must be auto, direct, commensurate, or "
            "interpolated"
        )
    q_rtol = float(config.get("response_q_interpolation_rtol", 0.0))
    q_atol = float(config.get("response_q_interpolation_atol", 0.0))
    if not np.isfinite(q_rtol) or q_rtol < 0.0:
        raise ValueError(
            "response_q_interpolation_rtol must be finite and nonnegative"
        )
    if not np.isfinite(q_atol) or q_atol < 0.0:
        raise ValueError(
            "response_q_interpolation_atol must be finite and nonnegative"
        )
    if (
        str(config.get("response_q_evaluation", "auto")) == "interpolated"
        and q_rtol == 0.0
        and q_atol == 0.0
    ):
        raise ValueError(
            "interpolated response_q_evaluation requires a positive "
            "interpolation tolerance"
        )
    q_mesh = tuple(
        int(value)
        for value in config.get("response_q_interpolation_mesh", ())
    )
    if q_mesh and (len(q_mesh) not in {1, 2, 3} or any(value < 1 for value in q_mesh)):
        raise ValueError(
            "response_q_interpolation_mesh must be empty or contain one to "
            "three positive sizes"
        )
    if int(config.get("response_q_validation_points", 8)) < 1:
        raise ValueError("response_q_validation_points must be positive")
    certificates = config.get("response_q_interpolation_certificates", {})
    if not isinstance(certificates, Mapping):
        raise ValueError(
            "response_q_interpolation_certificates must be a mapping"
        )
    if str(config.get("chemical_potential_mode", "source")) not in {
        "source",
        "filling",
    }:
        raise ValueError("chemical_potential_mode must be source or filling")
    filling = float(config.get("filling_per_cell", 1.0))
    if not np.isfinite(filling) or filling <= 0.0:
        raise ValueError("filling_per_cell must be finite and positive")
    if str(config.get("response_backend", "auto")) not in {
        "auto",
        "numpy",
        "threaded",
        "cupy",
    }:
        raise ValueError(
            "response_backend must be auto, numpy, threaded, or cupy"
        )
    if int(config.get("response_backend_probe_points", 8)) < 1:
        raise ValueError("response_backend_probe_points must be positive")
    backend_rtol = float(config.get("response_backend_rtol", 1.0e-10))
    backend_atol = float(config.get("response_backend_atol_meV", 1.0e-8))
    if not np.isfinite(backend_rtol) or backend_rtol < 0.0:
        raise ValueError("response_backend_rtol must be finite and nonnegative")
    if not np.isfinite(backend_atol) or backend_atol < 0.0:
        raise ValueError(
            "response_backend_atol_meV must be finite and nonnegative"
        )
    if int(config.get("response_workers", 0)) < 0:
        raise ValueError("response_workers must be nonnegative")
    if str(config.get("response_transition_backend", "auto")) not in {
        "auto",
        "numpy",
        "numba",
    }:
        raise ValueError(
            "response_transition_backend must be auto, numpy, or numba"
        )
    memory = float(config.get("response_max_batch_mb", 256.0))
    if not np.isfinite(memory) or memory <= 0.0:
        raise ValueError("response_max_batch_mb must be finite and positive")
    transition_memory = float(
        config.get("response_transition_max_batch_mb", 256.0)
    )
    if not np.isfinite(transition_memory) or transition_memory <= 0.0:
        raise ValueError(
            "response_transition_max_batch_mb must be finite and positive"
        )
    cache_memory = float(config.get("response_cache_mb", 512.0))
    if not np.isfinite(cache_memory) or cache_memory < 0.0:
        raise ValueError("response_cache_mb must be finite and nonnegative")
    if int(config.get("response_cache_entries", 64)) < 0:
        raise ValueError("response_cache_entries must be nonnegative")
    if int(config.get("powder_orientations", 50)) < 6:
        raise ValueError("powder_orientations must be at least 6")
    formula_mode = str(
        config.get("formula_units_mode", "manual")
    ).strip().lower()
    if formula_mode not in {"auto", "manual"}:
        raise ValueError("formula_units_mode must be auto or manual")
    if formula_mode == "manual":
        formula_units = float(config.get("formula_units_per_cell", 1.0))
        if not np.isfinite(formula_units) or formula_units <= 0.0:
            raise ValueError(
                "formula_units_per_cell must be finite and positive"
            )
    magnetic_mode = str(
        config.get("magnetic_normalization_mode", "auto")
    ).strip().lower()
    if magnetic_mode not in {"auto", "manual"}:
        raise ValueError("magnetic_normalization_mode must be auto or manual")
    if magnetic_mode == "manual":
        magnetic_count = float(
            config.get("magnetic_centers_per_model_cell", 1.0)
        )
        if not np.isfinite(magnetic_count) or magnetic_count <= 0.0:
            raise ValueError(
                "magnetic_centers_per_model_cell must be finite and positive"
            )
    plot_q = np.asarray(config.get("plot_q_reduced", ()), dtype=float)
    if plot_q.shape != (3,) or np.any(~np.isfinite(plot_q)):
        raise ValueError("plot_q_reduced must contain three finite coordinates")
    plot_min = float(config.get("plot_energy_min_meV", -100.0))
    plot_max = float(config.get("plot_energy_max_meV", 100.0))
    if not np.isfinite(plot_min) or not np.isfinite(plot_max) or plot_min >= plot_max:
        raise ValueError("plot energy limits must be finite and increasing")
    if int(config.get("plot_energy_points", 401)) < 2:
        raise ValueError("plot_energy_points must be at least 2")
    plot_temperature = float(config.get("plot_temperature_K", 10.0))
    if not np.isfinite(plot_temperature) or plot_temperature < 0.0:
        raise ValueError("plot_temperature_K must be finite and nonnegative")
    if (
        str(config.get("chemical_potential_mode", "source")) == "filling"
        and plot_temperature == 0.0
    ):
        raise ValueError(
            "filling-based plot chemical potential requires positive temperature"
        )
    mesh_scales = np.asarray(
        config.get("convergence_mesh_scales", ()),
        dtype=float,
    )
    broadening_scales = np.asarray(
        config.get("convergence_broadening_scales", ()),
        dtype=float,
    )
    if (
        mesh_scales.ndim != 1
        or mesh_scales.size < 1
        or np.any(~np.isfinite(mesh_scales))
        or np.any(mesh_scales <= 0.0)
    ):
        raise ValueError("convergence_mesh_scales must be positive and finite")
    if (
        broadening_scales.ndim != 1
        or broadening_scales.size < 1
        or np.any(~np.isfinite(broadening_scales))
        or np.any(broadening_scales <= 0.0)
    ):
        raise ValueError(
            "convergence_broadening_scales must be positive and finite"
        )
    if int(config.get("convergence_energy_points", 9)) < 1:
        raise ValueError("convergence_energy_points must be positive")
    relative_floor = float(config.get("convergence_relative_floor", 1.0e-12))
    if not np.isfinite(relative_floor) or relative_floor <= 0.0:
        raise ValueError("convergence_relative_floor must be finite and positive")
    _validate_scalar_bulk_config(component)


def _validate_rpa_component(component: Any) -> None:
    config = component.config if isinstance(component.config, dict) else {}
    if not str(config.get("response_component", "")).strip():
        raise ValueError("response_component must name a Lindhard component")
    tolerance = float(config.get("singular_tolerance", 1.0e-12))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("singular_tolerance must be finite and positive")
    near_pole = float(config.get("near_pole_tolerance", 1.0e-3))
    if not np.isfinite(near_pole) or near_pole <= tolerance:
        raise ValueError(
            "near_pole_tolerance must be finite and exceed singular_tolerance"
        )
    warning_margin = float(
        config.get("static_stability_warning_margin", 0.05)
    )
    if not np.isfinite(warning_margin) or warning_margin < 0.0:
        raise ValueError(
            "static_stability_warning_margin must be finite and nonnegative"
        )


def _validate_stoner_rpa_component(component: Any) -> None:
    _validate_rpa_component(component)
    interaction = float(component.parameters.get("I", np.nan))
    if not np.isfinite(interaction):
        raise ValueError("Stoner interaction I must be finite")


def _validate_matrix_rpa_component(component: Any) -> None:
    _validate_rpa_component(component)
    matrix = np.asarray(component.config.get("vertex_matrix", ()), dtype=complex)
    if matrix.shape != (3, 3) or np.any(~np.isfinite(matrix)):
        raise ValueError("vertex_matrix must be a finite 3 by 3 matrix")
    if not np.allclose(matrix, matrix.conj().T, rtol=0.0, atol=1.0e-12):
        raise ValueError("vertex_matrix must be Hermitian")
    if not str(component.config.get("channel", "spin")).strip():
        raise ValueError("interaction channel cannot be empty")
    scale = float(component.parameters.get("scale", np.nan))
    if not np.isfinite(scale):
        raise ValueError("matrix interaction scale must be finite")


def _validate_hubbard_hund_rpa_component(component: Any) -> None:
    _validate_rpa_component(component)
    config = component.config if isinstance(component.config, dict) else {}
    shells = config.get("correlated_shells", [])
    if not isinstance(shells, (str, list, tuple)):
        raise ValueError("correlated_shells must be a list or comma-separated string")
    for name in ("U", "U_prime", "J_H", "J_pair"):
        if not np.isfinite(float(component.parameters.get(name, np.nan))):
            raise ValueError(f"Hubbard-Hund parameter {name} must be finite")


def tight_binding_parameter_names(component: Any) -> tuple[str, ...]:
    """Return active named Hamiltonian coefficients for shared fit machinery."""

    from .electronic_builder import tight_binding_parameter_names as names

    return names(component)


def _validate_tight_binding_config(component: Any) -> None:
    """Validate canonical electronic energies and the presentation unit."""

    from .electronic_structure import normalize_electronic_energy_unit

    config = component.config
    _validate_electronic_sampling_config(config, "dos")
    normalize_electronic_energy_unit(config.get("electronic_energy_unit", "eV"))
    hopping_parameterization = str(
        config.get("hopping_parameterization", "slater_koster")
    )
    if hopping_parameterization not in {"slater_koster", "general"}:
        raise ValueError(
            "hopping_parameterization must be 'slater_koster' or 'general'"
        )
    path_convention = str(config.get("band_path_convention", "hinuma"))
    if path_convention not in {
        "hinuma",
        "setyawan_curtarolo",
        "manual",
    }:
        raise ValueError(
            "band_path_convention must be hinuma, "
            "setyawan_curtarolo, or manual"
        )
    electronic_backend = str(config.get("electronic_backend", "auto"))
    if electronic_backend not in {"auto", "numpy", "threaded", "cupy"}:
        raise ValueError(
            "electronic_backend must be auto, numpy, threaded, or cupy"
        )
    electronic_workers = int(config.get("electronic_workers", 0))
    if electronic_workers < 0:
        raise ValueError("electronic_workers must be zero or positive")
    electronic_max_batch_mb = float(
        config.get("electronic_max_batch_mb", 256.0)
    )
    if (
        not np.isfinite(electronic_max_batch_mb)
        or electronic_max_batch_mb <= 0.0
    ):
        raise ValueError("electronic_max_batch_mb must be positive and finite")
    if not isinstance(config.get("band_path_metadata", {}), Mapping):
        raise ValueError("band_path_metadata must be a mapping")
    point_density = float(config.get("band_points_per_inv_angstrom", 80.0))
    if not np.isfinite(point_density) or point_density <= 0.0:
        raise ValueError(
            "band_points_per_inv_angstrom must be positive and finite"
        )
    dos_symmetry = str(config.get("dos_symmetry", "auto"))
    if dos_symmetry not in {
        "auto",
        "full",
        "reduced",
    }:
        raise ValueError("dos_symmetry must be auto, full, or reduced")
    dos_method = str(config.get("dos_method", "gaussian"))
    if dos_method not in {"gaussian", "tetrahedron"}:
        raise ValueError("dos_method must be gaussian or tetrahedron")
    fermi_mesh_mode = str(config.get("fermi_mesh_mode", "spacing"))
    if fermi_mesh_mode not in {"spacing", "size"}:
        raise ValueError("fermi_mesh_mode must be spacing or size")
    fermi_spacing = float(
        config.get("fermi_spacing_inv_angstrom", 0.025)
    )
    if not np.isfinite(fermi_spacing) or fermi_spacing <= 0.0:
        raise ValueError(
            "fermi_spacing_inv_angstrom must be positive and finite"
        )
    energy_names = (
        "chemical_potential_meV",
        "dos_energy_min_meV",
        "dos_energy_max_meV",
        "dos_broadening_meV",
        "fermi_energy_meV",
    )
    energies = {name: float(config.get(name, 0.0)) for name in energy_names}
    if any(not np.isfinite(value) for value in energies.values()):
        raise ValueError("tight-binding canonical energy settings must be finite")
    if energies["dos_energy_min_meV"] >= energies["dos_energy_max_meV"]:
        raise ValueError("dos_energy_min_meV must be below dos_energy_max_meV")
    if (
        dos_method == "gaussian"
        and energies["dos_broadening_meV"] <= 0.0
    ):
        raise ValueError("Gaussian dos_broadening_meV must be positive")
    from .electronic_builder import (
        HoppingInvariant,
        OnsiteInvariant,
        OrbitalManifold,
        tight_binding_parameter_terms,
    )

    manifolds = [
        OrbitalManifold.from_dict(item)
        for item in config.get("orbital_manifolds", ())
    ]
    if len({item.label for item in manifolds}) != len(manifolds):
        raise ValueError("tight-binding orbital manifold labels must be unique")
    for item in config.get("onsite_terms", ()):
        OnsiteInvariant.from_dict(item)
    cutoff = float(config.get("hopping_cutoff_angstrom", 0.0))
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("hopping_cutoff_angstrom must be finite and nonnegative")
    for item in config.get("hopping_candidates", ()):
        HoppingInvariant.from_dict(item)
    for item in config.get("hopping_terms", ()):
        HoppingInvariant.from_dict(item)
    for term in tight_binding_parameter_terms(component):
        name = term.identifier
        if name not in component.parameters:
            raise ValueError(
                f"tight-binding parameter {name!r} is missing shared value state"
            )
        if not np.isclose(float(component.parameters[name]), term.value_meV):
            raise ValueError(
                f"tight-binding parameter {name!r} disagrees with its builder term"
            )
        raw_limits = component.limits.get(name, (None, None))
        limits = (
            tuple(raw_limits)
            if isinstance(raw_limits, (list, tuple)) and len(raw_limits) == 2
            else (None, None)
        )
        if limits != term.bounds_meV:
            raise ValueError(
                f"tight-binding limits for {name!r} disagree with its builder term"
            )
        if bool(component.fit_parameters.get(name, False)) != term.fit:
            raise ValueError(
                f"tight-binding fit selection for {name!r} disagrees with its "
                "builder term"
            )


ISOTROPIC_POLARIZATION = 2.0
"""Polarization factor for one component of an isotropic susceptibility.

The scalar spin-fluctuation kernels return
``chi_xx = chi_yy = chi_zz`` rather than their three-component trace.
Contracting that response with ``delta_ab - Qhat_a Qhat_b`` therefore gives
``P = 2``. A scalar defined as the trace would instead use ``P = 2/3``.
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


def _spectral_model_observable(
    data: PointData4D,
    chipp: np.ndarray,
    *,
    form_factor_sq: float | np.ndarray,
    polarization: float | np.ndarray,
    legacy: str = "intensity",
    bose_denominator_values: np.ndarray | None = None,
) -> np.ndarray:
    """Map a model's canonical ``chi''`` to the selected dataset channel.

    ``bose_denominator_values`` optionally supplies the dataset's already
    evaluated ``1 - exp(-E/kT)``. It depends only on the fitted points, never
    on the model parameters, so evaluators that call this once per optimizer
    iteration can hoist it out of the loop (see
    :meth:`_RpaComponentEvaluator._bose_denominator`). The arithmetic is
    unchanged either way.
    """

    convention = data.metadata.get("spectral_observable")
    if not isinstance(convention, dict):
        if legacy == "chipp":
            return np.asarray(chipp, dtype=float)
        return intensity_from_chipp(
            chipp,
            data.E,
            _dataset_temperature(data),
            form_factor_sq=form_factor_sq,
            polarization=polarization,
            bose_denominator_values=bose_denominator_values,
        )
    # Model kernels return a spin-operator response. Convert it to the
    # dataset's declared response convention exactly once.
    g_value = convention.get("g_factor", 2.0)
    g_factor = 2.0 if g_value in (None, "") else float(g_value)
    target_moment_unit = str(convention.get("moment_unit", "mu_B_squared"))
    if convention.get("fit_representation") == "chi_double_prime":
        response = np.asarray(chipp, dtype=float)
        if target_moment_unit == "mu_B_squared":
            response = response * magnetic_moment_factor("spin_squared", g_factor)
        elif target_moment_unit != "spin_squared":
            raise ValueError(
                "spectral moment_unit must be 'mu_B_squared' or 'spin_squared'"
            )
        return response

    kinematic: float | np.ndarray = 1.0
    if convention.get("kf_ki_state") == "included":
        incident = convention.get("incident_energy_meV")
        final = convention.get("final_energy_meV")
        kinematic = kf_over_ki(
            data.E,
            incident_energy_meV=None if incident in (None, "") else float(incident),
            final_energy_meV=None if final in (None, "") else float(final),
        )
    values = cross_section_from_chipp(
        chipp,
        data.E,
        _dataset_temperature(data),
        form_factor_sq=form_factor_sq,
        polarization=polarization,
        kf_ki=kinematic,
        # Model chi'' is a spin response regardless of the stored data
        # channel's display convention.
        moment_unit="spin_squared",
        g_factor=g_factor,
        bose_denominator_values=bose_denominator_values,
    )
    unit = str(convention.get("unit", ""))
    if unit.startswith("mbarn/"):
        values = values * MILLIBARN_PER_BARN
    return np.asarray(values, dtype=float)


def _quasistatic_model_observable(
    data: PointData4D,
    chi_static: np.ndarray,
    *,
    form_factor_sq: float | np.ndarray,
    polarization: float | np.ndarray,
) -> np.ndarray:
    """Map static spin susceptibility to energy-integrated elastic intensity."""

    convention = data.metadata.get("spectral_observable")
    g_factor = 2.0
    unit = ""
    if isinstance(convention, dict):
        g_value = convention.get("g_factor", 2.0)
        g_factor = 2.0 if g_value in (None, "") else float(g_value)
        unit = str(convention.get("unit", ""))
    values = quasistatic_cross_section_from_chi(
        chi_static,
        _dataset_temperature(data),
        form_factor_sq=form_factor_sq,
        polarization=polarization,
        moment_unit="spin_squared",
        g_factor=g_factor,
    )
    if unit.startswith("mbarn/"):
        values = values * MILLIBARN_PER_BARN
    return np.asarray(values, dtype=float)


def _is_elastic_dataset(data: PointData4D) -> bool:
    return str(data.metadata.get("data_type", "")) in {
        "single_crystal_elastic",
        "powder_elastic",
    }


def _dataset_magnetic_field(data: PointData4D) -> np.ndarray:
    """Return the applied field of a fitted dataset, validating it.

    Zeeman-enabled models require a Cartesian Tesla 3-vector on every fitted
    dataset (stamped from the dataset's Conditions settings).
    """

    field_vector = data.magnetic_field
    if field_vector is None or not np.all(np.isfinite(np.asarray(field_vector, dtype=float))):
        raise ValueError(
            "this model has the Zeeman term enabled and requires a valid "
            "applied magnetic field for every fitted dataset: set the field "
            "magnitude and direction in the dataset details (Conditions), "
            "and make sure the data group has lattice "
            "parameters to orient the direction"
        )
    return np.asarray(field_vector, dtype=float)


def _scalar_bulk_observable(
    data: PointData4D,
    chi_uniform: float | np.ndarray,
    *,
    g_factor: float,
    magnetic_ions_per_formula_unit: float,
) -> np.ndarray:
    """Convert uniform static spin susceptibility to a bulk fit channel.

    ``chi_uniform`` is one Cartesian spin-susceptibility component in
    meV^-1 per magnetic ion. Relative channels retain the historical model
    units. Absolute channels are converted to molar CGS or rationalized SI
    susceptibility, or to the requested linear-response moment.
    """

    from .sum_rules import (
        EMU_PER_MOL_PER_MODEL_CHI,
        EMU_PER_MOL_PER_MU_B,
        OERSTED_PER_TESLA,
    )

    metadata = data.metadata if isinstance(data.metadata, dict) else {}
    quantity_type = str(metadata.get("quantity_type", "magnetic_moment"))
    if quantity_type not in {
        "magnetic_moment",
        "magnetization",
        "bulk_susceptibility",
    }:
        raise ValueError(f"bulk model cannot predict quantity type {quantity_type!r}")
    g_value = float(g_factor)
    sites_per_fu = float(magnetic_ions_per_formula_unit)
    if not np.isfinite(g_value) or g_value <= 0.0:
        raise ValueError("bulk g factor must be finite and positive")
    if not np.isfinite(sites_per_fu) or sites_per_fu <= 0.0:
        raise ValueError(
            "magnetic ions per formula unit must be finite and positive"
        )
    chi = np.broadcast_to(np.asarray(chi_uniform, dtype=float), (data.size,))
    if np.any(~np.isfinite(chi)):
        raise ValueError("uniform static susceptibility must be finite")

    values = g_value**2 * chi
    if quantity_type != "bulk_susceptibility":
        raw_field = data.magnetic_field
        if raw_field is None:
            signed_field = np.zeros(data.size, dtype=float)
        else:
            field_vectors = np.asarray(raw_field, dtype=float)
            if field_vectors.ndim == 1:
                field_vectors = np.broadcast_to(field_vectors, (data.size, 3))
            axis = np.asarray(
                metadata.get("field_direction_cartesian", []), dtype=float
            )
            if axis.shape == (3,) and np.all(np.isfinite(axis)) and np.linalg.norm(axis) > 0:
                axis = axis / np.linalg.norm(axis)
                signed_field = field_vectors @ axis
            else:
                # Older projects did not store the fixed measurement axis.
                # Preserve the field magnitude while taking its sign from the
                # Cartesian component with the largest range.
                component = int(np.argmax(np.max(np.abs(field_vectors), axis=0)))
                signed_field = np.linalg.norm(field_vectors, axis=1) * np.sign(
                    field_vectors[:, component]
                )
        values = values * signed_field

    if not bool(metadata.get("absolute_units")):
        return np.asarray(values, dtype=float)

    target_unit = str(metadata.get("unit", ""))
    if quantity_type == "bulk_susceptibility":
        values = values * EMU_PER_MOL_PER_MODEL_CHI * sites_per_fu
        if target_unit == "m^3/mol":
            values = convert_quantity(
                values, "bulk_susceptibility", "cm^3/mol", target_unit
            )
        elif target_unit != "cm^3/mol":
            raise ValueError(
                "absolute bulk susceptibility must use cm^3/mol or m^3/mol"
            )
        return np.asarray(values, dtype=float)

    mass_g = float(metadata.get("sample_mass_mg", 0.0)) / 1000.0
    molar_mass = float(metadata.get("molar_mass_g_mol", 0.0))
    if molar_mass <= 0.0 or mass_g <= 0.0:
        raise ValueError(
            "absolute magnetic moment requires a positive sample mass and molar mass"
        )
    moles = mass_g / molar_mass
    values = (
        values
        * EMU_PER_MOL_PER_MODEL_CHI
        * moles
        * OERSTED_PER_TESLA
        * sites_per_fu
    )
    if target_unit == "emu/mol":
        values = values / moles
    elif target_unit == "mu_B/f.u.":
        values = values / (moles * EMU_PER_MOL_PER_MU_B)
    elif target_unit == "A m^2":
        values = convert_quantity(values, "magnetic_moment", "emu", target_unit)
    elif target_unit not in {"", "emu"}:
        raise ValueError(
            "absolute magnetic moment must use emu, emu/mol, mu_B/f.u., or A m^2"
        )
    return np.asarray(values, dtype=float)


def _form_factor_sq_from_config(component: Any, data: PointData4D) -> float | np.ndarray:
    """Return ``|f(Q)|^2`` from a component's ``ion`` / coefficient config.

    Returns 1.0 when the component configures no form factor. Computing
    ``|Q|`` in inverse angstrom requires lattice or UB metadata on the data.
    ``form_factor_g_J`` selects the dipole approximation; its default of 2.0 is
    the spin-only ``<j0>`` form factor.
    """

    amplitude = _form_factor_from_config(component, data)
    return np.asarray(amplitude, dtype=float) ** 2


def _form_factor_from_config(component: Any, data: PointData4D) -> float | np.ndarray:
    """Return the signed magnetic form-factor amplitude for a component."""

    config = component.config if isinstance(component.config, dict) else {}
    from .form_factors import (
        magnetic_form_factor_profile,
        normalized_form_factor_mode,
    )

    if normalized_form_factor_mode(config) == "none":
        return 1.0
    from .fitting import q_modulus_inv_angstrom

    q = q_modulus_inv_angstrom(data)
    return magnetic_form_factor_profile(q, config)


def _powder_sphere_directions(count: int) -> np.ndarray:
    """Return deterministic equal-area directions for a powder average."""

    count = max(int(count), 6)
    index = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.column_stack(
        [radius * np.cos(azimuth), radius * np.sin(azimuth), z]
    )


def _powder_reciprocal_matrix(
    data: PointData4D,
    lattice: Mapping[str, Any] | None,
) -> np.ndarray:
    """Return the matrix mapping model HKL to Cartesian inverse angstroms."""

    matrix = data.metadata.get("rlu_to_inv_angstrom_matrix")
    if matrix is None and isinstance(lattice, Mapping):
        from .fitting import reciprocal_basis_from_lattice_parameters

        required = ("a", "b", "c")
        if all(name in lattice for name in required):
            matrix = reciprocal_basis_from_lattice_parameters(
                float(lattice["a"]),
                float(lattice["b"]),
                float(lattice["c"]),
                float(lattice.get("alpha", 90.0)),
                float(lattice.get("beta", 90.0)),
                float(lattice.get("gamma", 90.0)),
            )
    result = np.asarray(matrix, dtype=float) if matrix is not None else np.empty(0)
    if (
        result.shape != (3, 3)
        or not np.all(np.isfinite(result))
        or abs(float(np.linalg.det(result))) < 1.0e-14
    ):
        raise ValueError(
            "powder Heisenberg RPA evaluation requires valid crystal lattice "
            "parameters on the model or workspace"
        )
    return result


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


def _lattice_reciprocal_matrix(
    data: PointData4D,
    lattice: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Resolve a reciprocal-basis matrix from data or model lattice metadata."""

    matrix = data.metadata.get("rlu_to_inv_angstrom_matrix")
    if matrix is None and isinstance(lattice, Mapping):
        from .fitting import reciprocal_basis_from_lattice_parameters

        if all(name in lattice for name in ("a", "b", "c")):
            matrix = reciprocal_basis_from_lattice_parameters(
                float(lattice["a"]),
                float(lattice["b"]),
                float(lattice["c"]),
                float(lattice.get("alpha", 90.0)),
                float(lattice.get("beta", 90.0)),
                float(lattice.get("gamma", 90.0)),
            )
    if matrix is None and not _metadata_coordinate_units_are_inv_angstrom(data.metadata):
        matrix = _resolve_q_transform(data)
    result = np.asarray(matrix, dtype=float) if matrix is not None else np.empty(0)
    if (
        result.shape != (3, 3)
        or not np.all(np.isfinite(result))
        or abs(float(np.linalg.det(result))) < 1.0e-14
    ):
        raise ValueError(
            "generalized paramagnon evaluation requires valid lattice or "
            "reciprocal-basis metadata"
        )
    return result


def _periodic_cartesian_offset_candidates(
    q_rlu: np.ndarray,
    center_rlu: np.ndarray,
    reciprocal_matrix: np.ndarray,
) -> np.ndarray:
    """Return nearby Cartesian offsets to reciprocal images of one center."""

    fractional = np.asarray(q_rlu, dtype=float) - np.asarray(center_rlu, dtype=float)
    base = np.rint(fractional)
    shifts = np.asarray(
        [
            (i, j, k)
            for i in (-1.0, 0.0, 1.0)
            for j in (-1.0, 0.0, 1.0)
            for k in (-1.0, 0.0, 1.0)
        ],
        dtype=float,
    )
    return (
        fractional[:, None, :] - base[:, None, :] - shifts[None, :, :]
    ) @ reciprocal_matrix.T


def _generalized_paramagnon_kernels(
    q_cartesian: np.ndarray,
    q_rlu: np.ndarray,
    *,
    reciprocal_matrix: np.ndarray,
    centers_rlu: np.ndarray,
    correlation_cholesky: np.ndarray,
    spatial_power: float,
    periodic: bool,
) -> np.ndarray:
    """Return one spatial kernel per point and symmetry-related center."""

    kernels = []
    for center in centers_rlu:
        if periodic:
            offsets = _periodic_cartesian_offset_candidates(
                q_rlu, center, reciprocal_matrix
            )
            image_kernels = paramagnon_spatial_kernel(
                offsets,
                correlation_cholesky_angstrom=correlation_cholesky,
                spatial_power=spatial_power,
            )
            kernels.append(np.min(image_kernels, axis=1))
        else:
            offsets = q_cartesian - center @ reciprocal_matrix.T
            kernels.append(
                paramagnon_spatial_kernel(
                    offsets,
                    correlation_cholesky_angstrom=correlation_cholesky,
                    spatial_power=spatial_power,
                )
            )
    return np.column_stack(kernels)


def _generalized_paramagnon_factory(component: Any) -> ModelFunction:
    """Build the generalized relaxational/propagating paramagnon evaluator."""

    name = component.name
    parameter_names = (
        "chi_peak",
        "gamma0",
        "relaxation_power",
        "inverse_mode_energy_sq",
        "xi_x",
        "xi_y",
        "xi_z",
        "xi_yx",
        "xi_zx",
        "xi_zy",
        "q0_h",
        "q0_k",
        "q0_l",
    )
    keys = {
        parameter: qualified_parameter_name(name, parameter)
        for parameter in parameter_names
    }
    config = component.config if isinstance(component.config, dict) else {}
    lattice = config.get("lattice")
    center_offsets = np.asarray(
        config.get("center_offsets", [[0.0, 0.0, 0.0]]), dtype=float
    )
    spatial_power = float(config.get("spatial_power", 2.0))
    periodic = bool(config.get("periodic", True))
    combination = str(config.get("center_combination", "sum")).strip().lower()
    powder_orientations = max(int(config.get("powder_orientations", 50)), 6)

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        bulk = data.metadata.get("data_type") == "magnetization"
        reciprocal = _lattice_reciprocal_matrix(data, lattice)
        powder = (
            data.metadata.get("data_type") in {"powder_inelastic", "powder_elastic"}
            or bool(data.metadata.get("powder_q_modulus_axis"))
        )
        orientation_count = 1
        if powder:
            from .fitting import q_modulus_inv_angstrom

            orientation_count = powder_orientations
            directions = _powder_sphere_directions(orientation_count)
            q_modulus = np.asarray(q_modulus_inv_angstrom(data), dtype=float)
            q_cartesian = (
                q_modulus[:, None, None] * directions[None, :, :]
            ).reshape(-1, 3)
            q_rlu = q_cartesian @ np.linalg.inv(reciprocal).T
            energy = np.repeat(np.asarray(data.E, dtype=float), orientation_count)
        else:
            coordinates = np.column_stack([data.H, data.K, data.L]).astype(float)
            if _metadata_coordinate_units_are_inv_angstrom(data.metadata):
                q_cartesian = coordinates
                q_rlu = q_cartesian @ np.linalg.inv(reciprocal).T
            else:
                q_rlu = coordinates
                q_cartesian = q_rlu @ reciprocal.T
            energy = np.asarray(data.E, dtype=float)

        q0 = np.asarray(
            [
                params[keys["q0_h"]],
                params[keys["q0_k"]],
                params[keys["q0_l"]],
            ],
            dtype=float,
        )
        centers = q0[None, :] + center_offsets
        cholesky = np.asarray(
            [
                [params[keys["xi_x"]], 0.0, 0.0],
                [params[keys["xi_yx"]], params[keys["xi_y"]], 0.0],
                [
                    params[keys["xi_zx"]],
                    params[keys["xi_zy"]],
                    params[keys["xi_z"]],
                ],
            ],
            dtype=float,
        )
        kernels = _generalized_paramagnon_kernels(
            q_cartesian,
            q_rlu,
            reciprocal_matrix=reciprocal,
            centers_rlu=centers,
            correlation_cholesky=cholesky,
            spatial_power=spatial_power,
            periodic=periodic,
        )
        if combination == "nearest":
            kernels = np.min(kernels, axis=1, keepdims=True)

        if bulk or _is_elastic_dataset(data):
            response = np.sum(float(params[keys["chi_peak"]]) / kernels, axis=1)
        else:
            response = np.sum(
                generalized_paramagnon_chipp(
                    kernels,
                    energy[:, None],
                    chi_peak=float(params[keys["chi_peak"]]),
                    gamma0=float(params[keys["gamma0"]]),
                    relaxation_power=float(params[keys["relaxation_power"]]),
                    inverse_mode_energy_sq=float(
                        params[keys["inverse_mode_energy_sq"]]
                    ),
                ),
                axis=1,
            )
        if powder:
            response = response.reshape(data.size, orientation_count).mean(axis=1)
        if bulk:
            return _scalar_bulk_observable(
                data,
                response,
                g_factor=float(config.get("bulk_g_factor", 2.0)),
                magnetic_ions_per_formula_unit=float(
                    config.get("magnetic_ions_per_formula_unit", 1.0)
                ),
            )
        if _is_elastic_dataset(data):
            return _quasistatic_model_observable(
                data,
                response,
                form_factor_sq=_form_factor_sq_from_config(component, data),
                polarization=ISOTROPIC_POLARIZATION,
            )
        return _spectral_model_observable(
            data,
            response,
            form_factor_sq=_form_factor_sq_from_config(component, data),
            polarization=ISOTROPIC_POLARIZATION,
        )

    return model


def _conserved_ferromagnetic_factory(component: Any) -> ModelFunction:
    """Build the clean or diffusive conserved-ferromagnetic response."""

    name = component.name
    parameter_names = (
        "chi_uniform",
        "gamma_scale",
        "xi_x",
        "xi_y",
        "xi_z",
        "xi_yx",
        "xi_zx",
        "xi_zy",
        "q0_h",
        "q0_k",
        "q0_l",
    )
    keys = {
        parameter: qualified_parameter_name(name, parameter)
        for parameter in parameter_names
    }
    config = component.config if isinstance(component.config, dict) else {}
    lattice = config.get("lattice")
    center_offsets = np.asarray(
        config.get("center_offsets", [[0.0, 0.0, 0.0]]), dtype=float
    )
    periodic = bool(config.get("periodic", True))
    combination = str(config.get("center_combination", "nearest")).strip().lower()
    powder_orientations = max(int(config.get("powder_orientations", 50)), 6)
    damping_power = 1.0 if str(config.get("damping_kind", "clean")) == "clean" else 2.0

    def complex_response(
        data: PointData4D, params: Mapping[str, float]
    ) -> tuple[np.ndarray, int]:
        reciprocal = _lattice_reciprocal_matrix(data, lattice)
        powder = (
            data.metadata.get("data_type") in {"powder_inelastic", "powder_elastic"}
            or bool(data.metadata.get("powder_q_modulus_axis"))
        )
        orientation_count = 1
        if powder:
            from .fitting import q_modulus_inv_angstrom

            orientation_count = powder_orientations
            directions = _powder_sphere_directions(orientation_count)
            q_modulus = np.asarray(q_modulus_inv_angstrom(data), dtype=float)
            q_cartesian = (
                q_modulus[:, None, None] * directions[None, :, :]
            ).reshape(-1, 3)
            q_rlu = q_cartesian @ np.linalg.inv(reciprocal).T
            source_energy = (
                np.zeros(data.size, dtype=float)
                if _is_elastic_dataset(data)
                or data.metadata.get("data_type") == "magnetization"
                else np.asarray(data.E, dtype=float)
            )
            energy = np.repeat(source_energy, orientation_count)
        else:
            coordinates = np.column_stack([data.H, data.K, data.L]).astype(float)
            if _metadata_coordinate_units_are_inv_angstrom(data.metadata):
                q_cartesian = coordinates
                q_rlu = q_cartesian @ np.linalg.inv(reciprocal).T
            else:
                q_rlu = coordinates
                q_cartesian = q_rlu @ reciprocal.T
            energy = (
                np.zeros(data.size, dtype=float)
                if _is_elastic_dataset(data)
                or data.metadata.get("data_type") == "magnetization"
                else np.asarray(data.E, dtype=float)
            )
        q0 = np.asarray(
            [params[keys["q0_h"]], params[keys["q0_k"]], params[keys["q0_l"]]],
            dtype=float,
        )
        centers = q0[None, :] + center_offsets
        cholesky = np.asarray(
            [
                [params[keys["xi_x"]], 0.0, 0.0],
                [params[keys["xi_yx"]], params[keys["xi_y"]], 0.0],
                [params[keys["xi_zx"]], params[keys["xi_zy"]], params[keys["xi_z"]]],
            ],
            dtype=float,
        )
        kernels = _generalized_paramagnon_kernels(
            q_cartesian,
            q_rlu,
            reciprocal_matrix=reciprocal,
            centers_rlu=centers,
            correlation_cholesky=cholesky,
            spatial_power=2.0,
            periodic=periodic,
        )
        if combination == "nearest":
            kernels = np.min(kernels, axis=1, keepdims=True)
        rho = np.sqrt(np.maximum(kernels - 1.0, 0.0))
        response = np.sum(
            conserved_ferromagnetic_susceptibility(
                rho,
                energy[:, None],
                chi_uniform=float(params[keys["chi_uniform"]]),
                gamma_scale=float(params[keys["gamma_scale"]]),
                damping_power=damping_power,
            ),
            axis=1,
        )
        return response, orientation_count

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        response, orientation_count = complex_response(data, params)
        if orientation_count > 1:
            response = response.reshape(data.size, orientation_count).mean(axis=1)
        if data.metadata.get("data_type") == "magnetization":
            return _scalar_bulk_observable(
                data,
                response.real,
                g_factor=float(config.get("bulk_g_factor", 2.0)),
                magnetic_ions_per_formula_unit=float(
                    config.get("magnetic_ions_per_formula_unit", 1.0)
                ),
            )
        if _is_elastic_dataset(data):
            return _quasistatic_model_observable(
                data,
                response.real,
                form_factor_sq=_form_factor_sq_from_config(component, data),
                polarization=ISOTROPIC_POLARIZATION,
            )
        return _spectral_model_observable(
            data,
            response.imag,
            form_factor_sq=_form_factor_sq_from_config(component, data),
            polarization=ISOTROPIC_POLARIZATION,
        )

    model.complex_response = complex_response  # type: ignore[attr-defined]
    return model


def _validate_scalar_bulk_config(component: Any) -> None:
    """Validate shared bulk-normalization settings on a scalar model."""

    config = component.config if isinstance(component.config, dict) else {}
    g_factor = float(config.get("bulk_g_factor", 2.0))
    sites_per_fu = float(config.get("magnetic_ions_per_formula_unit", 1.0))
    if not np.isfinite(g_factor) or g_factor <= 0.0:
        raise ValueError("bulk_g_factor must be finite and positive")
    if not np.isfinite(sites_per_fu) or sites_per_fu <= 0.0:
        raise ValueError(
            "magnetic_ions_per_formula_unit must be finite and positive"
        )


def _validate_generalized_paramagnon_component(component: Any) -> None:
    """Validate fixed generalized-paramagnon configuration."""

    _validate_scalar_bulk_config(component)
    config = component.config if isinstance(component.config, dict) else {}
    offsets = np.asarray(
        config.get("center_offsets", [[0.0, 0.0, 0.0]]), dtype=float
    )
    if offsets.ndim != 2 or offsets.shape[1] != 3 or offsets.shape[0] < 1:
        raise ValueError("center_offsets must be a nonempty list of [dH, dK, dL]")
    if np.any(~np.isfinite(offsets)):
        raise ValueError("center_offsets must contain only finite numbers")
    if np.unique(np.round(offsets, 12), axis=0).shape[0] != offsets.shape[0]:
        raise ValueError("center_offsets contains duplicate centers")
    if bool(config.get("periodic", True)):
        wrapped = offsets - np.floor(offsets)
        if np.unique(np.round(wrapped, 12), axis=0).shape[0] != offsets.shape[0]:
            raise ValueError(
                "center_offsets contains equivalent centers modulo reciprocal "
                "lattice vectors while periodic is true"
            )
    power = float(config.get("spatial_power", 2.0))
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("spatial_power must be finite and positive")
    if str(config.get("center_combination", "sum")).strip().lower() not in {
        "sum",
        "nearest",
    }:
        raise ValueError("center_combination must be 'sum' or 'nearest'")
    if int(config.get("powder_orientations", 50)) < 6:
        raise ValueError("powder_orientations must be at least 6")


def _validate_conserved_ferromagnetic_component(component: Any) -> None:
    """Validate the fixed conserved-ferromagnetic configuration."""

    _validate_generalized_paramagnon_component(component)
    config = component.config if isinstance(component.config, dict) else {}
    if str(config.get("damping_kind", "clean")) not in {"clean", "diffusive"}:
        raise ValueError("damping_kind must be 'clean' or 'diffusive'")


def _generalized_paramagnon_component_diagnostics(
    component: Any, data: PointData4D, params: Mapping[str, float]
) -> dict[str, Any]:
    """Return fit-safe dynamical and correlation-length diagnostics."""

    del data
    prefix = f"{component.name}."
    inertia = float(params[prefix + "inverse_mode_energy_sq"])
    gamma0 = float(params[prefix + "gamma0"])
    cholesky = np.asarray(
        [
            [params[prefix + "xi_x"], 0.0, 0.0],
            [params[prefix + "xi_yx"], params[prefix + "xi_y"], 0.0],
            [
                params[prefix + "xi_zx"],
                params[prefix + "xi_zy"],
                params[prefix + "xi_z"],
            ],
        ],
        dtype=float,
    )
    principal = np.linalg.svd(cholesky, compute_uv=False)
    record: dict[str, Any] = {
        "chi_static_qpeak": float(params[prefix + "chi_peak"]),
        "gamma0": gamma0,
        "relaxation_power": float(params[prefix + "relaxation_power"]),
        "xi_principal_min": float(np.min(principal)),
        "xi_principal_max": float(np.max(principal)),
        "propagating": float(inertia > 0.0),
    }
    if inertia > 0.0:
        mode_energy = 1.0 / np.sqrt(inertia)
        damping = 1.0 / (inertia * gamma0)
        record.update(
            {
                "mode_energy_qpeak": float(mode_energy),
                "dho_damping_qpeak": float(damping),
                "damping_ratio_qpeak": float(damping / (2.0 * mode_energy)),
            }
        )
    return record


def _local_relaxational_factory(component: Any) -> ModelFunction:
    name = component.name
    chi_key = qualified_parameter_name(name, "chi_loc")
    gamma_key = qualified_parameter_name(name, "gamma")
    config = component.config if isinstance(component.config, dict) else {}

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        if data.metadata.get("data_type") == "magnetization":
            return _scalar_bulk_observable(
                data,
                float(params[chi_key]),
                g_factor=float(config.get("bulk_g_factor", 2.0)),
                magnetic_ions_per_formula_unit=float(
                    config.get("magnetic_ions_per_formula_unit", 1.0)
                ),
            )
        if _is_elastic_dataset(data):
            return _quasistatic_model_observable(
                data,
                np.full(data.size, float(params[chi_key]), dtype=float),
                form_factor_sq=_form_factor_sq_from_config(component, data),
                polarization=ISOTROPIC_POLARIZATION,
            )
        chipp = local_relaxational_chipp(
            data.E,
            chi_loc=float(params[chi_key]),
            gamma=float(params[gamma_key]),
        )
        return _spectral_model_observable(
            data,
            chipp,
            form_factor_sq=_form_factor_sq_from_config(component, data),
            polarization=ISOTROPIC_POLARIZATION,
        )

    return model


def _mmp_relaxational_factory(component: Any) -> ModelFunction:
    name = component.name
    keys = {
        parameter: qualified_parameter_name(name, parameter)
        for parameter in ("chi_pk", "xi", "omega_sf", "q0_h", "q0_k", "q0_l")
    }
    config = component.config if isinstance(component.config, dict) else {}

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        q0 = (
            float(params[keys["q0_h"]]),
            float(params[keys["q0_k"]]),
            float(params[keys["q0_l"]]),
        )
        q_offset_sq = _q_offset_sq_inv_angstrom(data, q0)
        chi_static = float(params[keys["chi_pk"]]) / (
            1.0 + float(params[keys["xi"]]) ** 2 * q_offset_sq
        )
        if data.metadata.get("data_type") == "magnetization":
            return _scalar_bulk_observable(
                data,
                chi_static,
                g_factor=float(config.get("bulk_g_factor", 2.0)),
                magnetic_ions_per_formula_unit=float(
                    config.get("magnetic_ions_per_formula_unit", 1.0)
                ),
            )
        if _is_elastic_dataset(data):
            return _quasistatic_model_observable(
                data,
                chi_static,
                form_factor_sq=_form_factor_sq_from_config(component, data),
                polarization=ISOTROPIC_POLARIZATION,
            )
        chipp = mmp_chipp(
            q_offset_sq,
            data.E,
            chi_pk=float(params[keys["chi_pk"]]),
            xi=float(params[keys["xi"]]),
            omega_sf=float(params[keys["omega_sf"]]),
        )
        return _spectral_model_observable(
            data,
            chipp,
            form_factor_sq=_form_factor_sq_from_config(component, data),
            polarization=ISOTROPIC_POLARIZATION,
        )

    return model


def _local_susceptibility_factory(
    component: Any, _components: Mapping[str, Any]
) -> Callable[[PointData4D, Mapping[str, float]], ScalarSusceptibilityResponse]:
    chi_key = qualified_parameter_name(component.name, "chi_loc")
    gamma_key = qualified_parameter_name(component.name, "gamma")

    def response(
        data: PointData4D, params: Mapping[str, float]
    ) -> ScalarSusceptibilityResponse:
        chi = local_relaxational_susceptibility(
            data.E,
            chi_loc=float(params[chi_key]),
            gamma=float(params[gamma_key]),
        )
        return ScalarSusceptibilityResponse(
            chi,
            _form_factor_from_config(component, data),
            component.name,
        )

    return response


def _mmp_susceptibility_factory(
    component: Any, _components: Mapping[str, Any]
) -> Callable[[PointData4D, Mapping[str, float]], ScalarSusceptibilityResponse]:
    keys = {
        parameter: qualified_parameter_name(component.name, parameter)
        for parameter in ("chi_pk", "xi", "omega_sf", "q0_h", "q0_k", "q0_l")
    }

    def response(
        data: PointData4D, params: Mapping[str, float]
    ) -> ScalarSusceptibilityResponse:
        q0 = tuple(float(params[keys[name]]) for name in ("q0_h", "q0_k", "q0_l"))
        chi = mmp_susceptibility(
            _q_offset_sq_inv_angstrom(data, q0),
            data.E,
            chi_pk=float(params[keys["chi_pk"]]),
            xi=float(params[keys["xi"]]),
            omega_sf=float(params[keys["omega_sf"]]),
        )
        return ScalarSusceptibilityResponse(
            chi,
            _form_factor_from_config(component, data),
            component.name,
        )

    return response


def _generalized_paramagnon_susceptibility_factory(
    component: Any, _components: Mapping[str, Any]
) -> Callable[[PointData4D, Mapping[str, float]], ScalarSusceptibilityResponse]:
    names = (
        "chi_peak", "gamma0", "relaxation_power", "inverse_mode_energy_sq",
        "xi_x", "xi_y", "xi_z", "xi_yx", "xi_zx", "xi_zy",
        "q0_h", "q0_k", "q0_l",
    )
    keys = {name: qualified_parameter_name(component.name, name) for name in names}
    config = component.config if isinstance(component.config, dict) else {}
    reciprocal_lattice = config.get("lattice")
    offsets = np.asarray(config.get("center_offsets", [[0.0, 0.0, 0.0]]), dtype=float)
    periodic = bool(config.get("periodic", True))
    combination = str(config.get("center_combination", "sum")).strip().lower()
    spatial_power = float(config.get("spatial_power", 2.0))

    def response(
        data: PointData4D, params: Mapping[str, float]
    ) -> ScalarSusceptibilityResponse:
        reciprocal = _lattice_reciprocal_matrix(data, reciprocal_lattice)
        coordinates = np.column_stack([data.H, data.K, data.L]).astype(float)
        if _metadata_coordinate_units_are_inv_angstrom(data.metadata):
            q_cartesian = coordinates
            q_rlu = q_cartesian @ np.linalg.inv(reciprocal).T
        else:
            q_rlu = coordinates
            q_cartesian = q_rlu @ reciprocal.T
        q0 = np.asarray(
            [params[keys["q0_h"]], params[keys["q0_k"]], params[keys["q0_l"]]],
            dtype=float,
        )
        cholesky = np.asarray(
            [
                [params[keys["xi_x"]], 0.0, 0.0],
                [params[keys["xi_yx"]], params[keys["xi_y"]], 0.0],
                [params[keys["xi_zx"]], params[keys["xi_zy"]], params[keys["xi_z"]]],
            ],
            dtype=float,
        )
        kernels = _generalized_paramagnon_kernels(
            q_cartesian,
            q_rlu,
            reciprocal_matrix=reciprocal,
            centers_rlu=q0[None, :] + offsets,
            correlation_cholesky=cholesky,
            spatial_power=spatial_power,
            periodic=periodic,
        )
        if combination == "nearest":
            kernels = np.min(kernels, axis=1, keepdims=True)
        chi = np.sum(
            generalized_paramagnon_susceptibility(
                kernels,
                np.asarray(data.E, dtype=float)[:, None],
                chi_peak=float(params[keys["chi_peak"]]),
                gamma0=float(params[keys["gamma0"]]),
                relaxation_power=float(params[keys["relaxation_power"]]),
                inverse_mode_energy_sq=float(params[keys["inverse_mode_energy_sq"]]),
            ),
            axis=1,
        )
        return ScalarSusceptibilityResponse(
            chi,
            _form_factor_from_config(component, data),
            component.name,
        )

    return response


def _conserved_ferromagnetic_susceptibility_factory(
    component: Any, _components: Mapping[str, Any]
) -> Callable[[PointData4D, Mapping[str, float]], ScalarSusceptibilityResponse]:
    model = _conserved_ferromagnetic_factory(component)
    complex_response = model.complex_response  # type: ignore[attr-defined]

    def response(
        data: PointData4D, params: Mapping[str, float]
    ) -> ScalarSusceptibilityResponse:
        chi, orientation_count = complex_response(data, params)
        if orientation_count != 1:
            raise ValueError(
                "coupled susceptibility currently requires single-crystal source data"
            )
        return ScalarSusceptibilityResponse(
            chi,
            _form_factor_from_config(component, data),
            component.name,
        )

    return response


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
        # Magnetic sites in the user's cell (before primitive reduction).
        # A complete chemical crystal converts this count to sites per formula
        # unit; legacy magnetic-only structures retain the historical fallback.
        self._n_magnetic_sites = len(site_positions)
        self._magnetic_sites_per_formula_unit: float | None = None
        try:
            from .crystal import infer_crystal_formula_units

            formula = infer_crystal_formula_units(
                config.get("crystal") or {},
                cell_multiplicity=1,
            )
            inferred = (
                self._n_magnetic_sites
                / formula.conventional_formula_units
            )
            if np.isfinite(inferred) and inferred > 0.0:
                self._magnetic_sites_per_formula_unit = float(inferred)
        except (ImportError, KeyError, TypeError, ValueError):
            # Legacy manually entered magnetic models may not contain the
            # complete chemical structure needed for formula-unit inference.
            pass
        self.labels = heisenberg_rpa_orbit_labels(component)
        self.j_keys = {label: qualified_parameter_name(name, label) for label in self.labels}
        self.chi0_key = qualified_parameter_name(name, "chi0")
        self.gamma0_key = qualified_parameter_name(name, "gamma0")
        self.inertia_key = qualified_parameter_name(name, "inverse_mode_energy_sq")
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
        self._closure_moment_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._closure_result_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._closure_bz_context: Any = None
        self._bulk_q0_context: Any = None
        self._bulk_q0_cache: tuple[tuple[Any, ...], Any] | None = None
        self._lattice = (config.get("crystal") or {}).get("lattice")
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
        self._geometry_cache: dict[int, tuple[PointData4D, Any, Any, Any, int]] = {}
        # Same identity-checked pattern for the dataset's Bose denominator: it
        # is a function of (E, T) alone, so it is constant across a fit while
        # being one of the most expensive per-evaluation array operations.
        self._bose_cache: dict[int, tuple[PointData4D, np.ndarray]] = {}
        self._intensity_per_chipp_cache: dict[int, tuple[PointData4D, np.ndarray]] = {}

    def _bose_denominator(self, data: PointData4D) -> np.ndarray | None:
        """Cached ``1 - exp(-E/kT)`` for ``data``, or ``None`` when unusable.

        Returns ``None`` for datasets whose channel convention does not apply
        the Bose factor, so the caller simply forwards ``None`` and the
        conversion behaves exactly as before.
        """

        convention = data.metadata.get("spectral_observable")
        if isinstance(convention, dict) and (
            convention.get("fit_representation") == "chi_double_prime"
        ):
            return None
        cached = self._bose_cache.get(id(data))
        if cached is not None and cached[0] is data:
            return cached[1]
        if len(self._bose_cache) > 32:
            self._bose_cache.clear()
        values = bose_denominator(data.E, _dataset_temperature(data))
        self._bose_cache[id(data)] = (data, values)
        return values

    def _intensity_per_chipp(
        self, data: PointData4D, form_factor_sq: float | np.ndarray
    ) -> np.ndarray:
        """Cached ``d(intensity)/d(chi'')``, the channel map's linear factor."""

        cached = self._intensity_per_chipp_cache.get(id(data))
        if cached is not None and cached[0] is data:
            return cached[1]
        if len(self._intensity_per_chipp_cache) > 32:
            self._intensity_per_chipp_cache.clear()
        factor = np.asarray(
            _spectral_model_observable(
                data,
                np.ones(data.size, dtype=float),
                form_factor_sq=form_factor_sq,
                polarization=ISOTROPIC_POLARIZATION,
                bose_denominator_values=self._bose_denominator(data),
            ),
            dtype=float,
        )
        self._intensity_per_chipp_cache[id(data)] = (data, factor)
        return factor

    def _geometry(self, data: PointData4D) -> tuple[Any, Any, Any, int]:
        cached = self._geometry_cache.get(id(data))
        if cached is not None and cached[0] is data:
            return cached[1], cached[2], cached[3], cached[4]
        if len(self._geometry_cache) > 32:
            self._geometry_cache.clear()
        powder = (
            data.metadata.get("data_type") in {"powder_inelastic", "powder_elastic"}
            or bool(data.metadata.get("powder_q_modulus_axis"))
        )
        powder_count = 1
        reciprocal_matrix = None
        if powder:
            from .fitting import q_modulus_inv_angstrom

            powder_count = max(int(self.config.get("powder_orientations", 50)), 6)
            reciprocal_matrix = _powder_reciprocal_matrix(data, self._lattice)
            directions = _powder_sphere_directions(powder_count)
            q_modulus = np.asarray(q_modulus_inv_angstrom(data), dtype=float)
            # Powder points sharing a |Q| generate exactly the same sphere of
            # sampled directions, so collapse |Q| first and orient only the
            # distinct moduli. A powder S(|Q|, E) map has one |Q| per momentum
            # bin repeated over every energy bin, so this shrinks the grid the
            # geometry has to deduplicate by the number of energy bins -- the
            # sampled directions, the resulting HKL values, and the per-point
            # mapping below are unchanged.
            unique_modulus, modulus_inverse = np.unique(q_modulus, return_inverse=True)
            modulus_inverse = np.asarray(modulus_inverse, dtype=np.intp).ravel()
            q_cartesian = (
                unique_modulus[:, None, None] * directions[None, :, :]
            ).reshape(-1, 3)
            hkl = q_cartesian @ np.linalg.inv(reciprocal_matrix).T
            geometry = build_rpa_geometry(
                hkl[:, 0],
                hkl[:, 1],
                hkl[:, 2],
                self.site_positions,
                self.orbits,
            )
            # Re-expand to one row per (fitted point, orientation), keeping the
            # orientation-major layout that _powder_average averages over.
            geometry = replace(
                geometry,
                point_index=np.ascontiguousarray(
                    geometry.point_index.reshape(unique_modulus.size, powder_count)[
                        modulus_inverse
                    ].ravel()
                ),
            )
        else:
            geometry = build_rpa_geometry(
                data.H,
                data.K,
                data.L,
                self.site_positions,
                self.orbits,
            )
        form_factor_sq = _form_factor_sq_from_config(self.component, data)
        tensor_context = None
        if self.tensor_mode:
            tensor_context = self._build_tensor_context(
                geometry,
                data,
                reciprocal_matrix=reciprocal_matrix,
            )
        self._geometry_cache[id(data)] = (
            data,
            geometry,
            form_factor_sq,
            tensor_context,
            powder_count,
        )
        return geometry, form_factor_sq, tensor_context, powder_count

    def _build_tensor_context(
        self,
        geometry: Any,
        data: PointData4D,
        *,
        reciprocal_matrix: np.ndarray | None = None,
    ) -> tuple[Any, Any]:
        from .tensor_rpa import build_tensor_structure, cartesian_qhat_per_point

        matrix = (
            reciprocal_matrix
            if reciprocal_matrix is not None
            else data.metadata.get("rlu_to_inv_angstrom_matrix")
        )
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

    @staticmethod
    def _powder_average(values: np.ndarray, data_size: int, count: int) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if count == 1:
            return result
        return np.mean(result.reshape(data_size, count), axis=1)

    def _inertia(self, params: Mapping[str, float]) -> float:
        """Return inertia, treating pre-inertia project components as zero."""

        return float(
            params.get(
                self.inertia_key,
                self.component.parameters.get("inverse_mode_energy_sq", 0.0),
            )
        )

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
        self._closure_moment_cache[key] = model
        self._closure_moment_cache.move_to_end(key)
        while len(self._closure_moment_cache) > 16:
            self._closure_moment_cache.popitem(last=False)
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
        key = self._closure_cache_key(params, t, field)
        cached = self._closure_result_cache.get(key)
        if cached is not None:
            self._closure_result_cache.move_to_end(key)
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
        self._closure_result_cache[key] = result
        self._closure_result_cache.move_to_end(key)
        while len(self._closure_result_cache) > 8192:
            self._closure_result_cache.popitem(last=False)
        return result

    def _bulk_q0_modes(self, params: dict[str, float]) -> Any:
        """Return the Q=0 eigensystem, once per exchange parameter vector."""

        if self.tensor_mode or self.zeeman_mode:
            values = self._tensor_values(params)
            key = ("tensor", *sorted(values.items()))
        else:
            values = self._j_values(params)
            key = ("scalar", *sorted(values.items()))
        if self._bulk_q0_cache is not None and self._bulk_q0_cache[0] == key:
            return self._bulk_q0_cache[1]

        if self._bulk_q0_context is None:
            geometry = build_rpa_geometry(
                [0.0], [0.0], [0.0], self.site_positions, self.orbits
            )
            structure = None
            if self.tensor_mode or self.zeeman_mode:
                from .tensor_rpa import build_tensor_structure

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
            self._bulk_q0_context = (geometry, structure)
        geometry, structure = self._bulk_q0_context

        if structure is not None:
            from .tensor_rpa import assemble_tensor_exchange

            exchange = assemble_tensor_exchange(structure, values)
            lam, modes = np.linalg.eigh(exchange)
            n_sites = structure.n_sites
            amps = modes.reshape(1, n_sites, 3, 3 * n_sites).sum(axis=1)[0]
            result = ("tensor", lam[0], amps, n_sites)
        else:
            exchange = rpa_exchange_matrix(geometry, values)
            lam, modes = np.linalg.eigh(exchange)
            weights = np.abs(modes.sum(axis=1)) ** 2 / geometry.n_sites
            result = ("scalar", lam, weights)
        self._bulk_q0_cache = (key, result)
        return result

    def _closure_cache_key(
        self,
        params: dict[str, float],
        temperature: float,
        field: np.ndarray | None,
    ) -> tuple[Any, ...]:
        """Exact inputs that can change a solved closure result.

        Under TAC, the model's ``chi0`` is only a numerical root seed and is
        deliberately excluded: changing it cannot change the physical root.
        This also prevents a formally unidentifiable fitted ``chi0`` from
        invalidating a full temperature-sweep cache.
        """

        relevant = (
            set(self.j_keys.values())
            | set(self.tensor_keys.values())
            | set(self._closure_keys.values())
            | {self.gamma0_key}
            | (set(self._zeeman_keys.values()) if self.zeeman_mode else set())
        )
        if self.closure_spec.mode != "tac":
            relevant.add(self.chi0_key)
        return (
            float(temperature),
            tuple(np.asarray(field, dtype=float)) if field is not None else None,
            tuple(float(params[key]) for key in sorted(relevant)),
        )

    def _solve_closures_at_temperatures(
        self,
        params: dict[str, float],
        temperatures: np.ndarray,
        field: np.ndarray | None,
    ) -> dict[float, Any]:
        """Solve a closure over a temperature sweep, batching Tier-A TAC."""

        unique = np.unique(np.asarray(temperatures, dtype=float))
        if unique.size == 0:
            return {}
        cached_results: dict[float, Any] = {}
        missing: list[float] = []
        for temperature in unique:
            key = self._closure_cache_key(params, float(temperature), field)
            cached = self._closure_result_cache.get(key)
            if cached is None:
                missing.append(float(temperature))
            else:
                cached_results[float(temperature)] = cached
        if not missing:
            return cached_results
        model = self._closure_moment_model(params, field)
        from .closures import (
            TierAMoments,
            solve_onsager_temperatures,
            solve_scr_temperatures,
            solve_tac_temperatures,
        )

        if not isinstance(model, TierAMoments):
            cached_results.update(
                {
                    temperature: self._solve_closure(params, temperature, field)
                    for temperature in missing
                }
            )
            return cached_results
        common = {
            "model": model,
            "gamma0": float(params[self.gamma0_key]),
            "temperature_K": np.asarray(missing, dtype=float),
            "cutoff_mev": self.closure_spec.energy_cutoff_mev,
        }
        if self.closure_spec.mode == "onsager":
            target = (
                float(
                    params.get(
                        self._closure_keys.get("m2_total", ""),
                        self.closure_spec.moment_target,
                    )
                )
                if self.closure_spec.moment_mode == "fitted"
                else self.closure_spec.moment_target
            )
            results = solve_onsager_temperatures(
                **common,
                chi0=float(params[self.chi0_key]),
                target=target,
            )
        elif self.closure_spec.mode == "scr":
            results = solve_scr_temperatures(
                **common,
                chi0_bare=float(params[self.chi0_key]),
                mode_coupling_u=float(
                    params.get(
                        self._closure_keys.get("mode_coupling_u", ""), 0.0
                    )
                ),
            )
        else:
            target = (
                float(
                    params.get(
                        self._closure_keys.get("total_amplitude", ""),
                        self.closure_spec.moment_target,
                    )
                )
                if self.closure_spec.moment_mode == "fitted"
                else self.closure_spec.moment_target
            )
            results = solve_tac_temperatures(
                **common,
                total_amplitude=target,
                guess=float(params[self.chi0_key]),
            )
        for temperature, result in zip(missing, results, strict=True):
            cached_results[temperature] = result
            self._closure_result_cache[
                self._closure_cache_key(params, temperature, field)
            ] = result
        if len(self._closure_result_cache) > 16384:
            self._closure_result_cache.clear()
        return cached_results

    def _bulk_static_chi(
        self,
        params: dict[str, float],
        chi0: float,
        lambda_shift: float,
        b_hat: np.ndarray | None,
    ) -> float:
        """Uniform static susceptibility at Q=0 (model units) per site.

        For the tensor/Zeeman paths this is the longitudinal component along
        ``b_hat`` (the magnetization direction) when a field is set, else the
        powder trace/3; for the scalar path the isotropic value. The Onsager
        ``lambda_shift`` enters as the rigid eigenvalue shift. This is the
        longitudinal static limit -- the field's gyrotropic effect on the
        *dynamic* response does not shift the static magnetization.
        """

        from .sum_rules import static_chi_modes

        mode, *payload = self._bulk_q0_modes(params)
        if mode == "tensor":
            lam, amps, n_sites = payload
            denom = 1.0 - (lam - lambda_shift) * chi0
            if np.any(denom <= 0.0):
                raise ValueError("RPA instability at Q=0")
            chi_modes = chi0 / denom
            if b_hat is not None:
                weight = np.abs(np.asarray(b_hat, dtype=float) @ amps) ** 2 / n_sites
            else:
                weight = (np.abs(amps) ** 2).sum(axis=0) / (3.0 * n_sites)
            return float((weight * chi_modes).sum())
        lam, weights = payload
        return float(static_chi_modes(lam, weights, chi0=chi0, lambda_shift=lambda_shift)[0])

    def _magnetization_value(
        self, data: PointData4D, params: dict[str, float]
    ) -> np.ndarray:
        """Predict the bulk moment M(T, B) for a magnetization dataset.

        ``M = g^2 * chi_uniform(T, B) * B`` (longitudinal), with the
        closure making ``chi_uniform`` field-dependent (nonlinear M(B)) through
        the reaction field. Dataset calibration is applied outside the model.
        In absolute mode the physical emu/mol constant and the sample's molar
        amount set the conversion. Closures are solved once per distinct
        (T, B).
        """

        n = data.size
        temperature = np.broadcast_to(
            np.atleast_1d(np.asarray(_dataset_temperature(data), dtype=float)), (n,)
        )
        raw_field = data.magnetic_field
        if raw_field is None:
            field_vecs = np.zeros((n, 3))
        elif np.ndim(raw_field) == 1:
            field_vecs = np.broadcast_to(np.asarray(raw_field, dtype=float), (n, 3))
        else:
            field_vecs = np.asarray(raw_field, dtype=float)
        b_mag = np.linalg.norm(field_vecs, axis=1)

        g_factor = (
            float(params[self._zeeman_keys["g_factor"]]) if self.zeeman_mode else 2.0
        )

        # Group by quantities that can actually change chi. In particular,
        # small MPMS field readback variations must not turn an otherwise
        # one-dimensional chi(T) calculation into one solve per raw row.
        key_columns: list[np.ndarray] = []
        if self.closure_spec is not None:
            key_columns.append(temperature)
        if self.zeeman_mode and self.closure_spec is not None:
            key_columns.extend(field_vecs[:, axis] for axis in range(3))
        elif self.tensor_mode or self.zeeman_mode:
            directions = np.zeros_like(field_vecs)
            nonzero = b_mag > 0.0
            directions[nonzero] = field_vecs[nonzero] / b_mag[nonzero, None]
            key_columns.extend(directions[:, axis] for axis in range(3))
        keys = np.round(
            np.column_stack(key_columns) if key_columns else np.zeros((n, 1)),
            9,
        )
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        chi_values = np.empty(n, dtype=float)
        invalid = np.zeros(n, dtype=bool)
        closure_by_temperature: dict[float, Any] = {}
        if self.closure_spec is not None and not self.zeeman_mode:
            closure_by_temperature = self._solve_closures_at_temperatures(
                params, unique_keys[:, 0], None
            )
        for index, _row in enumerate(unique_keys):
            sel = inverse == index
            representative = int(np.flatnonzero(sel)[0])
            t = float(temperature[representative])
            field_vec = field_vecs[representative]
            magnitude = float(np.linalg.norm(field_vec))
            b_hat = field_vec / magnitude if magnitude > 0 else None
            try:
                chi0 = float(params[self.chi0_key])
                lambda_shift = 0.0
                if self.closure_spec is not None:
                    closure = (
                        self._solve_closure(params, t, field_vec)
                        if self.zeeman_mode
                        else closure_by_temperature[t]
                    )
                    chi0 = closure.chi0_eff
                    lambda_shift = closure.lambda_shift
                chi_uniform = self._bulk_static_chi(params, chi0, lambda_shift, b_hat)
            except ValueError:
                invalid[sel] = True
                chi_values[sel] = 0.0
                continue
            chi_values[sel] = chi_uniform
        sites_per_fu = float(
            (self.config.get("bulk") or {}).get("sites_per_fu")
            or self._magnetic_sites_per_formula_unit
            or self._n_magnetic_sites
        )
        out = _scalar_bulk_observable(
            data,
            chi_values,
            g_factor=g_factor,
            magnetic_ions_per_formula_unit=sites_per_fu,
        )
        out[invalid] = 1e6
        return out

    def value(self, data: PointData4D, params: dict[str, float]) -> np.ndarray:
        if isinstance(data.metadata, dict) and data.metadata.get("data_type") == "magnetization":
            return self._magnetization_value(data, params)
        temperature = _dataset_temperature(data)
        # Validate the applied field before the instability guard below, so a
        # missing field raises its actionable error instead of being caught and
        # turned into the 1e6 sentinel.
        field = _dataset_magnetic_field(data) if self.zeeman_mode else None
        geometry, form_factor_sq, tensor_context, powder_count = self._geometry(data)
        energy = (
            np.repeat(np.asarray(data.E, dtype=float), powder_count)
            if powder_count > 1
            else np.asarray(data.E, dtype=float)
        )
        elastic = _is_elastic_dataset(data)
        try:
            chi0 = float(params[self.chi0_key])
            lambda_shift = 0.0
            if self.closure_spec is not None:
                closure = self._solve_closure(params, temperature, field)
                chi0 = closure.chi0_eff
                lambda_shift = closure.lambda_shift
            if self.zeeman_mode:
                from .tensor_rpa import (
                    MU_B_MEV_PER_T,
                    tensor_rpa_zeeman_unpolarized_chipp,
                    tensor_zeeman_susceptibility,
                    unpolarized_static_chi,
                    zeeman_cartesian_propagator,
                )

                structure, q_hat = tensor_context
                magnitude = float(np.linalg.norm(field))
                b_hat = field / magnitude if magnitude > 0 else np.array([0.0, 0.0, 1.0])
                g_factor = float(params[self._zeeman_keys["g_factor"]])
                propagator = zeeman_cartesian_propagator(
                    np.zeros_like(energy) if elastic else energy,
                    b_hat,
                    chi0=chi0,
                    gamma0=float(params[self.gamma0_key]),
                    omega_larmor=g_factor * MU_B_MEV_PER_T * magnitude,
                    chi_perp_ratio=float(params[self._zeeman_keys["chi_perp_ratio"]]),
                    gamma_perp_ratio=float(params[self._zeeman_keys["gamma_perp_ratio"]]),
                )
                if elastic:
                    chi_tensor = tensor_zeeman_susceptibility(
                        structure,
                        geometry,
                        np.zeros_like(energy),
                        propagator,
                        param_values=self._tensor_values(params),
                        lambda_shift=lambda_shift,
                    )
                    response = unpolarized_static_chi(chi_tensor, q_hat)
                else:
                    response = tensor_rpa_zeeman_unpolarized_chipp(
                        structure, geometry, energy, q_hat, propagator,
                        param_values=self._tensor_values(params),
                        lambda_shift=lambda_shift,
                    )
                polarization = 1.0
            elif self.tensor_mode:
                from .tensor_rpa import (
                    tensor_rpa_unpolarized_chipp,
                    tensor_susceptibility,
                    unpolarized_static_chi,
                )

                structure, q_hat = tensor_context
                if elastic:
                    chi_tensor = tensor_susceptibility(
                        structure,
                        geometry,
                        np.zeros_like(energy),
                        chi0=chi0,
                        gamma0=float(params[self.gamma0_key]),
                        param_values=self._tensor_values(params),
                        lambda_shift=lambda_shift,
                        inverse_mode_energy_sq=self._inertia(params),
                    )
                    response = unpolarized_static_chi(chi_tensor, q_hat)
                else:
                    response = tensor_rpa_unpolarized_chipp(
                        structure,
                        geometry,
                        energy,
                        q_hat,
                        chi0=chi0,
                        gamma0=float(params[self.gamma0_key]),
                        param_values=self._tensor_values(params),
                        lambda_shift=lambda_shift,
                        inverse_mode_energy_sq=self._inertia(params),
                    )
                # The unpolarized channel already carries the polarization
                # average, so no extra scalar polarization factor here.
                polarization = 1.0
            else:
                if elastic:
                    from .sum_rules import static_chi_modes

                    exchange = rpa_exchange_matrix(geometry, self._j_values(params))
                    eigenvalues, modes = np.linalg.eigh(exchange)
                    weights = (
                        np.abs(modes.sum(axis=1)) ** 2 / geometry.n_sites
                    )
                    response = static_chi_modes(
                        eigenvalues,
                        weights,
                        chi0=chi0,
                        lambda_shift=lambda_shift,
                    )[geometry.point_index]
                else:
                    response = heisenberg_rpa_chipp(
                        geometry,
                        energy,
                        chi0=chi0,
                        gamma0=float(params[self.gamma0_key]),
                        j_values=self._j_values(params),
                        lambda_shift=lambda_shift,
                        inverse_mode_energy_sq=self._inertia(params),
                    )
                polarization = ISOTROPIC_POLARIZATION
        except ValueError:
            # Unphysical trial parameters (RPA instability, non-positive
            # chi0/gamma0, unsatisfiable closure). Optimizers probe these while
            # searching; a huge finite misfit steers them back without
            # aborting the fit.
            return np.full(data.size, 1e6, dtype=float)
        response = self._powder_average(response, data.size, powder_count)
        if elastic:
            return _quasistatic_model_observable(
                data,
                response,
                form_factor_sq=form_factor_sq,
                polarization=polarization,
            )
        return _spectral_model_observable(
            data,
            response,
            form_factor_sq=form_factor_sq,
            polarization=polarization,
            bose_denominator_values=self._bose_denominator(data),
        )

    def susceptibility_response(
        self, data: PointData4D, params: Mapping[str, float]
    ) -> ScalarSusceptibilityResponse:
        """Return the complex scalar response before neutron factors.

        The initial composable contract supports the scalar exchange network.
        Tensor and Zeeman models retain their full internal response and are
        rejected here until the coupling component has a tensor-valued block
        contract.
        """

        if self.tensor_mode or self.zeeman_mode:
            raise ValueError(
                "coupled susceptibility currently supports scalar Heisenberg-RPA sources only"
            )
        geometry, _form_factor_sq, _tensor_context, powder_count = self._geometry(data)
        if powder_count != 1:
            raise ValueError(
                "coupled susceptibility currently requires single-crystal source data"
            )
        chi0 = float(params[self.chi0_key])
        lambda_shift = 0.0
        if self.closure_spec is not None:
            temperature = _dataset_temperature(data)
            closure = self._solve_closure(dict(params), temperature, None)
            chi0 = closure.chi0_eff
            lambda_shift = closure.lambda_shift
        chi = heisenberg_rpa_susceptibility(
            geometry,
            data.E,
            chi0=chi0,
            gamma0=float(params[self.gamma0_key]),
            j_values=self._j_values(params),
            lambda_shift=lambda_shift,
            inverse_mode_energy_sq=self._inertia(params),
        )
        return ScalarSusceptibilityResponse(
            chi,
            _form_factor_from_config(self.component, data),
            self.component.name,
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

        def eig_weights(geometry: Any) -> tuple[np.ndarray, np.ndarray]:
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

    def _stability_metrics(
        self,
        model: Any,
        chi0: float,
        lambda_shift: float,
    ) -> dict[str, float]:
        """Return the smallest sampled RPA denominator and its BZ location."""

        if hasattr(model, "eigenvalues"):
            eigenvalues = np.asarray(model.eigenvalues, dtype=float)
        else:
            eigenvalues = np.linalg.eigvalsh(np.asarray(model.exchange))
        flat_index = int(np.nanargmax(eigenvalues))
        q_index, mode_index = np.unravel_index(flat_index, eigenvalues.shape)
        lambda_max = float(eigenvalues[q_index, mode_index])
        ratio = float((lambda_max - lambda_shift) * chi0)
        bz_geometry, _bz_structure = self._closure_context()
        critical_q = np.asarray(bz_geometry.unique_hkl[q_index], dtype=float)
        return {
            "stability_margin": 1.0 - ratio,
            "stability_ratio": ratio,
            "stability_lambda_max": lambda_max,
            "stability_q_h": float(critical_q[0]),
            "stability_q_k": float(critical_q[1]),
            "stability_q_l": float(critical_q[2]),
            "stability_mode_index": float(mode_index),
        }

    def diagnostics(self, data: PointData4D, params: dict[str, float]) -> dict[str, Any]:
        """Post-fit physics diagnostics for one dataset (see docs/theory_notes).

        Returns a flat JSON-safe dict: the effective fluctuating moment
        ``mu_eff_sq`` (its zero-point/thermal split), the closure internals
        ``lambda_shift``/``chi0_eff``, ``chi0_gamma0``, the smallest sampled RPA
        denominator, and the static susceptibility at Q=0 and its BZ peak.
        Stability metrics are computed before moment/static-response evaluation
        so they remain available for unstable parameters.
        """

        temperature = _dataset_temperature(data)
        field = _dataset_magnetic_field(data) if self.zeeman_mode else None
        t_values = np.unique(np.atleast_1d(np.asarray(temperature, dtype=float)))
        t = float(t_values[0]) if t_values.size else float("nan")
        chi0 = float(params[self.chi0_key])
        gamma0 = float(params[self.gamma0_key])
        inertia = self._inertia(params)
        lambda_shift = 0.0
        cutoff = (
            self.closure_spec.energy_cutoff_mev
            if self.closure_spec is not None
            else self._DEFAULT_DIAGNOSTIC_CUTOFF_MEV
        )
        record: dict[str, Any] = {
            "temperature": t,
            "gamma0": gamma0,
            "chi0_bare": chi0,
            "inverse_mode_energy_sq": inertia,
        }
        try:
            model = self._closure_moment_model(params, field)
            bare_stability = self._stability_metrics(model, chi0, 0.0)
        except (ValueError, np.linalg.LinAlgError) as exc:
            record.update({"unstable": 1.0, "diagnostic_error": str(exc)})
            return record
        record.update(
            {
                f"bare_{key}": value
                for key, value in bare_stability.items()
                if key in {"stability_margin", "stability_ratio"}
            }
        )
        try:
            if self.closure_spec is not None:
                closure = self._solve_closure(params, temperature, field)
                chi0 = closure.chi0_eff
                lambda_shift = closure.lambda_shift
        except ValueError as exc:
            record.update(bare_stability)
            record.update(
                {
                    "chi0_eff": float("nan"),
                    "lambda_shift": float("nan"),
                    "stability_uses_effective_closure": 0.0,
                    "closure_solution": 0.0,
                    "unstable": 1.0,
                    "diagnostic_error": str(exc),
                }
            )
            return record
        record.update(self._stability_metrics(model, chi0, lambda_shift))
        record.update(
            {
                "chi0_eff": chi0,
                "lambda_shift": lambda_shift,
                "stability_uses_effective_closure": 1.0,
                "closure_solution": 1.0,
            }
        )
        if inertia > 0.0:
            try:
                chi_q0, chi_peak = self._static_chi_grid_and_q0(
                    params, chi0, lambda_shift
                )
            except ValueError as exc:
                record.update({"unstable": 1.0, "diagnostic_error": str(exc)})
                return record
            record.update(
                {
                    "chi_static_q0": chi_q0,
                    "chi_static_qpeak": chi_peak,
                    "moment_integral_available": 0.0,
                    "unstable": 0.0,
                }
            )
            return record
        try:
            zero_point, thermal = model.moment(
                chi0=chi0,
                gamma0=gamma0,
                temperature_K=t,
                cutoff_mev=cutoff,
                lambda_shift=lambda_shift,
            )
            chi_q0, chi_peak = self._static_chi_grid_and_q0(params, chi0, lambda_shift)
        except ValueError as exc:
            record.update({"unstable": 1.0, "diagnostic_error": str(exc)})
            return record
        record.update(
            {
                "chi0_gamma0": chi0 * gamma0,
                "mu_eff_sq": zero_point + thermal,
                "m2_zero_point": zero_point,
                "m2_thermal": thermal,
                "chi_static_q0": chi_q0,
                "chi_static_qpeak": chi_peak,
                "moment_integral_available": 1.0,
                "unstable": 0.0,
            }
        )
        return record

    def gradients(self, data: PointData4D, params: dict[str, float]) -> dict[str, np.ndarray]:
        """Return ``d(intensity)/d(param)`` keyed by qualified parameter name."""

        _dataset_temperature(data)
        geometry, form_factor_sq, _tensor_context, powder_count = self._geometry(data)
        energy = (
            np.repeat(np.asarray(data.E, dtype=float), powder_count)
            if powder_count > 1
            else np.asarray(data.E, dtype=float)
        )
        try:
            chipp, chipp_grads = heisenberg_rpa_chipp_and_gradients(
                geometry,
                energy,
                chi0=float(params[self.chi0_key]),
                gamma0=float(params[self.gamma0_key]),
                j_values=self._j_values(params),
                inverse_mode_energy_sq=self._inertia(params),
            )
        except ValueError:
            # Sentinel-penalty region: the value is a constant with no
            # parameter dependence, so every column is zero.
            zero = np.zeros(data.size, dtype=float)
            columns = {
                self.chi0_key: zero,
                self.gamma0_key: zero,
                self.inertia_key: zero,
            }
            columns.update({key: zero for key in self.j_keys.values()})
            return columns
        chipp = self._powder_average(chipp, data.size, powder_count)
        chipp_grads = {
            name: self._powder_average(values, data.size, powder_count)
            for name, values in chipp_grads.items()
        }
        # Intensity is linear in chi''. Compute its derivative directly rather
        # than dividing by chi'', which would be singular at response nodes.
        # Nothing in it depends on the fitted parameters, so it is cached for
        # the lifetime of the dataset instead of rebuilt per Jacobian call.
        d_intensity_d_chipp = self._intensity_per_chipp(data, form_factor_sq)
        columns = {}
        columns[self.chi0_key] = d_intensity_d_chipp * chipp_grads["chi0"]
        columns[self.gamma0_key] = d_intensity_d_chipp * chipp_grads["gamma0"]
        columns[self.inertia_key] = (
            d_intensity_d_chipp * chipp_grads["inverse_mode_energy_sq"]
        )
        for label, key in self.j_keys.items():
            columns[key] = d_intensity_d_chipp * chipp_grads[label]
        return columns


def _heisenberg_rpa_factory(component: Any) -> ModelFunction:
    return _RpaComponentEvaluator(component).value


def _heisenberg_rpa_susceptibility_factory(
    component: Any, _components: Mapping[str, Any]
) -> Callable[[PointData4D, Mapping[str, float]], ScalarSusceptibilityResponse]:
    return _RpaComponentEvaluator(component).susceptibility_response


def _validate_heisenberg_rpa_component(component: Any) -> None:
    inertia = float(component.parameters.get("inverse_mode_energy_sq", 0.0))
    if not np.isfinite(inertia) or inertia < 0.0:
        raise ValueError("inverse_mode_energy_sq must be finite and nonnegative")
    may_be_inertial = inertia > 0.0 or bool(
        component.fit_parameters.get("inverse_mode_energy_sq", False)
    )
    if not may_be_inertial:
        return
    config = component.config if isinstance(component.config, dict) else {}
    from .closures import ClosureSpec

    if ClosureSpec.from_config(config) is not None:
        raise ValueError(
            "inertial Heisenberg RPA does not yet support sum-rule closures; "
            "fix inverse_mode_energy_sq at zero or disable the closure"
        )
    if _component_has_zeeman(config):
        raise ValueError(
            "inertial Heisenberg RPA does not yet support the Zeeman propagator; "
            "fix inverse_mode_energy_sq at zero or disable Zeeman coupling"
        )


def _heisenberg_rpa_component_diagnostics(
    component: Any, data: PointData4D, params: Mapping[str, float]
) -> dict[str, Any] | None:
    """Calculate post-fit physics diagnostics for a Heisenberg RPA component."""

    return _RpaComponentEvaluator(component).diagnostics(data, dict(params))


def _electronic_rpa_component_diagnostics(
    component: Any,
    _data: PointData4D,
    params: Mapping[str, float],
    components: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return sampled zero-energy pole diagnostics at the configured plot Q."""

    from .model_plots import electronic_rpa_energy_scan

    local_components = {}
    for name, value in components.items():
        local = copy.copy(value)
        local.parameters = dict(getattr(value, "parameters", {}))
        local.config = dict(getattr(value, "config", {}))
        local_components[name] = local
    for local in local_components.values():
        for parameter in getattr(local, "parameters", {}):
            key = qualified_parameter_name(local.name, parameter)
            if key in params:
                local.parameters[parameter] = float(params[key])
    local_component = local_components.get(component.name)
    if local_component is None:
        return None
    local_component.config["reject_sampled_static_instability"] = False
    response_name = str(
        local_component.config.get("response_component", "")
    ).strip()
    response = local_components.get(response_name)
    if response is None:
        return None
    response.config["plot_energy_min_meV"] = -1.0e-9
    response.config["plot_energy_max_meV"] = 1.0e-9
    response.config["plot_energy_points"] = 3
    result = electronic_rpa_energy_scan(
        local_component,
        local_components,
    )
    dressing = result.provenance.get("dressing", {})
    if not isinstance(dressing, Mapping):
        return None
    stability = dressing.get("static_stability", {})
    if not isinstance(stability, Mapping):
        return None
    q = np.asarray(result.Q_reduced[1], dtype=float)
    return {
        "stability_margin": float(stability.get("margin", np.nan)),
        "stability_ratio": float(stability.get("ratio", np.nan)),
        "stability_q_h": float(q[0]),
        "stability_q_k": float(q[1]),
        "stability_q_l": float(q[2]),
        "minimum_relative_singular_value": float(
            dressing.get("minimum_relative_singular_value", np.nan)
        ),
        "near_rpa_pole": float(bool(dressing.get("near_pole", False))),
        "near_rpa_instability": float(
            bool(stability.get("near_instability", False))
        ),
        "unstable": float(bool(stability.get("unstable", False))),
        "stability_sample_scope": "configured plot Q at E=0",
    }


def compute_component_diagnostics(
    component: Any,
    data: PointData4D,
    params: Mapping[str, float],
    *,
    components: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run a component's registered post-fit diagnostics, when available.

    ``params`` contains fully resolved qualified parameter values. Invalid or
    incomplete model configuration returns ``None`` so diagnostics never turn
    an otherwise successful fit into a failure.
    """

    info = MODEL_TYPE_REGISTRY.get(getattr(component, "type", None))
    if info is None or (
        info.diagnostics is None and info.context_diagnostics is None
    ):
        return None
    try:
        if info.context_diagnostics is not None and components is not None:
            result = info.context_diagnostics(
                component,
                data,
                params,
                components,
            )
        elif info.diagnostics is not None:
            result = info.diagnostics(component, data, params)
        else:
            return None
    except (ValueError, KeyError, np.linalg.LinAlgError):
        return None
    return None if result is None else dict(result)


def _heisenberg_rpa_jacobian_factory(component: Any) -> ModelJacobian | None:
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


def component_parameter_names(component: Any) -> tuple[str, ...]:
    """Return all parameter names of a component, static plus config-derived."""

    info = MODEL_TYPE_REGISTRY[component.type]
    if info.dynamic_parameters is None:
        return info.parameters
    return (*info.parameters, *info.dynamic_parameters(component))


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
    scale parameter. Datasets with the same nonempty ``scale_group`` share one
    fitted scale and must supply the same starting value.
    """

    name: str
    data: PointData4D
    weight: float = 1.0
    data_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    scale_value: float = 1.0
    scale_vary: bool = False
    scale_group: str | None = None


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

    Parameters without an explicit entry use global sharing.
    """

    entry = component.sharing.get(parameter) if isinstance(component.sharing, dict) else None
    if isinstance(entry, dict) and entry.get("mode") in SHARING_MODES:
        return str(entry["mode"])
    return "global"


def dataset_scale_parameter_name(dataset_name: str) -> str:
    """Return the optimizer parameter name for a fitted dataset scale."""

    return f"dataset_scale[{dataset_name}]"


def parameter_limits(component: Any, parameter: str) -> tuple[float | None, float | None]:
    """Return ``(min, max)`` bounds for one component parameter."""

    info = MODEL_TYPE_REGISTRY.get(component.type)
    default_lower = (
        dict(info.default_lower_bounds).get(parameter) if info is not None else None
    )
    raw = component.limits.get(parameter) if isinstance(component.limits, dict) else None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return default_lower, None
    lower = default_lower if raw[0] in (None, "") else float(raw[0])
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


def parameter_is_derived_by_closure(component: Any, parameter: str) -> bool:
    """Whether an active closure makes a stored model parameter non-fittable."""

    if (
        getattr(component, "type", None) == "hubbard_hund_rpa"
        and parameter in {"U_prime", "J_pair"}
    ):
        config = component.config if isinstance(component.config, dict) else {}
        return bool(config.get("rotationally_invariant", True))
    if getattr(component, "type", None) != "heisenberg_rpa" or parameter != "chi0":
        return False
    config = component.config if isinstance(component.config, dict) else {}
    closure = config.get("closure") if isinstance(config, dict) else None
    return isinstance(closure, dict) and str(closure.get("mode", "none")).lower() == "tac"


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
        validate_model_component(component)
        if component.name in seen:
            raise ValueError(f"duplicate model component name {component.name!r}")
        seen.add(component.name)

    observable_applicable: dict[str, list[str]] = {}
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
        observable_applicable[component.name] = names

    active_by_name = {component.name: component for component in active}
    applicable = {
        name: list(names) for name, names in observable_applicable.items()
    }
    dependencies: dict[str, tuple[str, ...]] = {}
    for component in active:
        definition = MODEL_TYPE_REGISTRY[component.type]
        references = []
        for field_name in definition.component_reference_fields:
            referenced_name = str(component.config.get(field_name, "")).strip()
            if not referenced_name:
                raise ValueError(
                    f"model component {component.name!r} requires a "
                    f"{field_name!r} component reference"
                )
            if referenced_name == component.name:
                raise ValueError(
                    f"model component {component.name!r} cannot depend on itself"
                )
            if referenced_name not in active_by_name:
                raise ValueError(
                    f"model component {component.name!r} references missing or "
                    f"disabled component {referenced_name!r}"
                )
            references.append(referenced_name)
        dependencies[component.name] = tuple(references)

    visiting: set[str] = set()
    visited: set[str] = set()

    def validate_dependency_graph(component_name: str) -> None:
        if component_name in visiting:
            raise ValueError(
                f"model component dependency cycle includes {component_name!r}"
            )
        if component_name in visited:
            return
        visiting.add(component_name)
        for dependency_name in dependencies[component_name]:
            validate_dependency_graph(dependency_name)
        visiting.remove(component_name)
        visited.add(component_name)

    for component in active:
        validate_dependency_graph(component.name)

    # A dressing replaces, rather than adds to, its referenced observable on
    # the dressing's dataset scope. The referenced component remains an
    # independently usable observable on any other selected datasets.
    for component in active:
        definition = MODEL_TYPE_REGISTRY[component.type]
        if not definition.consumes_referenced_observables:
            continue
        consumed_datasets = set(observable_applicable[component.name])
        for dependency_name in dependencies[component.name]:
            observable_applicable[dependency_name] = [
                name
                for name in observable_applicable[dependency_name]
                if name not in consumed_datasets
            ]
            for dataset_name in consumed_datasets:
                components_by_dataset[dataset_name] = [
                    name
                    for name in components_by_dataset[dataset_name]
                    if name != dependency_name
                ]

    # Propagate each observable's dataset scope back to all parameter providers.
    for component in active:
        dependent_names = set(observable_applicable[component.name])
        stack = list(dependencies[component.name])
        seen_dependencies: set[str] = set()
        while stack:
            dependency_name = stack.pop()
            if dependency_name in seen_dependencies:
                continue
            seen_dependencies.add(dependency_name)
            applicable[dependency_name] = sorted(
                set(applicable[dependency_name]) | dependent_names,
                key=dataset_names.index,
            )
            stack.extend(dependencies[dependency_name])

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
            vary = bool(component.fit_parameters.get(parameter, False)) and not (
                parameter_is_derived_by_closure(component, parameter)
            )
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
    scale_groups: dict[tuple[str, str], list[FitDatasetInput]] = {}
    for dataset in fitted:
        if dataset.scale_vary:
            key = (
                ("group", str(dataset.scale_group))
                if dataset.scale_group
                else ("dataset", dataset.name)
            )
            scale_groups.setdefault(key, []).append(dataset)
    for (kind, key), members in scale_groups.items():
        values = np.asarray([float(dataset.scale_value) for dataset in members])
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("fitted dataset scales must be finite and positive")
        if not np.allclose(values, values[0], rtol=1.0e-12, atol=1.0e-12):
            names = ", ".join(dataset.name for dataset in members)
            raise ValueError(
                f"datasets sharing scale group {key!r} have different starting "
                f"scales ({names}); set one common dataset scale first"
            )
        name = dataset_scale_parameter_name(
            f"group:{key}" if kind == "group" else key
        )
        emit(
            ParameterSpec(
                name=name,
                value=float(values[0]),
                min=0.0,
                vary=True,
                description=(
                    f"Shared scale for datasets {', '.join(d.name for d in members)}"
                    if len(members) > 1
                    else f"Scale factor for dataset {members[0].name}"
                ),
            ),
            ParameterInstance(
                name=name,
                component="dataset",
                parameter="scale_factor",
                scope=key,
                datasets=tuple(dataset.name for dataset in members),
            ),
        )
        for dataset in members:
            scale_parameters[dataset.name] = name

    derived = _compile_constraints(active, applicable, specs, instances)

    fit_datasets: list[FitDataset] = []
    # Context-aware evaluators own expensive immutable setup and bounded
    # scientific caches. Build one evaluator per observable component for the
    # whole compiled problem so compatible datasets and dataset groups reuse
    # the same electronic response context.
    context_evaluators: dict[str, ModelFunction] = {}
    for dataset in fitted:
        fit_data = dataset.data
        if (
            dataset.data_type
            and fit_data.metadata.get("data_type") != dataset.data_type
        ) or fit_data.metadata.get("fit_dataset_name") != dataset.name:
            point_metadata = dict(fit_data.metadata)
            if dataset.data_type:
                point_metadata["data_type"] = dataset.data_type
            point_metadata["fit_dataset_name"] = dataset.name
            fit_data = fit_data.with_updates(metadata=point_metadata)
        components_here = [
            component
            for component in active
            if dataset.name in observable_applicable[component.name]
        ]
        evaluators = []
        for component in components_here:
            definition = MODEL_TYPE_REGISTRY[component.type]
            if definition.context_factory is None:
                evaluator = definition.factory(component)
            else:
                evaluator = context_evaluators.get(component.name)
                if evaluator is None:
                    evaluator = definition.context_factory(
                        component,
                        active_by_name,
                    )
                    context_evaluators[component.name] = evaluator
            evaluators.append(evaluator)
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
            and getattr(dataset, "data_type", None)
            not in {"magnetization", "single_crystal_elastic", "powder_elastic"}
            and all(factory is not None for factory in jacobian_factories)
        ):
            # A registry entry may have a jacobian factory that still declines
            # (returns None) for a particular component configuration (e.g. a
            # heisenberg_rpa component with anisotropic tensor terms). Only use
            # analytic Jacobians when every component actually yields one.
            built = [
                factory(component)
                for factory, component in zip(
                    jacobian_factories,
                    components_here,
                    strict=True,
                )
            ]
            if all(jacobian is not None for jacobian in built):
                model_jacobian = _additive_jacobian(built)
        fit_datasets.append(
            FitDataset(
                name=dataset.name,
                data=fit_data,
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
    """Compile exact expressions and inequalities as derived parameters."""

    derived: list[DerivedParameter] = []
    specs_by_name = {spec.name: spec for spec in specs}
    original_specs_by_name = dict(specs_by_name)
    original_names = set(specs_by_name)
    constrained_targets: set[str] = set()
    for component in components:
        if not applicable.get(component.name):
            continue
        for constraint in getattr(component, "constraints", []) or []:
            parameter = str(constraint.get("parameter", ""))
            op = str(constraint.get("op", ">="))
            reference = constraint.get("reference")
            qualified = qualified_parameter_name(component.name, parameter)
            if qualified in constrained_targets:
                raise ValueError(f"parameter {qualified!r} has more than one constraint")
            target = specs_by_name.get(qualified)
            if target is None:
                if any(name.startswith(f"{qualified}[") for name in specs_by_name):
                    raise ValueError(
                        f"constraint on {qualified!r} requires global sharing mode"
                    )
                raise ValueError(f"constraint references unknown parameter {qualified!r}")
            if op not in ("=", ">=", "<="):
                raise ValueError(f"unsupported constraint operator {op!r} on {qualified!r}")
            configured_limits = (
                component.limits.get(parameter)
                if isinstance(component.limits, dict)
                else None
            )
            has_explicit_limits = (
                isinstance(configured_limits, (list, tuple))
                and len(configured_limits) == 2
                and any(value not in (None, "") for value in configured_limits)
            )
            if has_explicit_limits:
                raise ValueError(
                    f"parameter {qualified!r} cannot have both limits and a constraint"
                )
            if not target.vary:
                raise ValueError(
                    f"constrained parameter {qualified!r} must be enabled for fitting"
                )

            if op == "=":
                expression = str(constraint.get("expression", constraint.get("reference", ""))).strip()
                dependencies = parameter_expression_names(expression)
                unknown = [name for name in dependencies if name not in original_names]
                if unknown:
                    raise ValueError(
                        f"constraint on {qualified!r} references unknown parameter {unknown[0]!r}"
                    )
                specs.remove(target)
                del specs_by_name[qualified]
                del instances[qualified]
                constrained_targets.add(qualified)
                derived.append(
                    DerivedParameter(
                        name=qualified,
                        base=0.0,
                        expression=expression,
                        dependencies=dependencies,
                    )
                )
                continue

            if isinstance(reference, str):
                reference_spec = original_specs_by_name.get(reference)
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
            constrained_targets.add(qualified)
    return derived
