"""Electronic-response fit evaluators.

This internal module owns the expensive Lindhard and electronic-RPA evaluator
construction.  :mod:`nfit.fit_config` keeps compatibility wrappers for the
historical private factory names used by the model registry and extensions.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import replace
from threading import RLock
from typing import Any

import numpy as np

from .dataset import PointData4D
from .fitting import ModelFunction, _metadata_coordinate_units_are_inv_angstrom

_FIT_CONFIG_HELPER_NAMES = (
    "_dataset_temperature",
    "_is_elastic_dataset",
    "_powder_sphere_directions",
    "_quasistatic_model_observable",
    "_scalar_bulk_observable",
    "_spectral_model_observable",
    "component_parameter_names",
    "qualified_parameter_name",
    "tight_binding_parameter_names",
)
_FIT_CONFIG_CONTEXT: Mapping[str, Any] = {}


def configure_fit_helpers(context: Mapping[str, Any]) -> None:
    """Install the narrow set of shared fit helpers used by this evaluator."""

    missing = set(_FIT_CONFIG_HELPER_NAMES).difference(context)
    if missing:
        raise KeyError(f"missing electronic fit helper {sorted(missing)[0]!r}")
    global _FIT_CONFIG_CONTEXT
    _FIT_CONFIG_CONTEXT = context


def _fit_config_helper(name: str) -> Any:
    try:
        return _FIT_CONFIG_CONTEXT[name]
    except KeyError as exc:
        raise RuntimeError("electronic fit helpers have not been configured") from exc


def qualified_parameter_name(component_name: str, parameter: str) -> str:
    return _fit_config_helper("qualified_parameter_name")(component_name, parameter)


def tight_binding_parameter_names(component: Any) -> tuple[str, ...]:
    return _fit_config_helper("tight_binding_parameter_names")(component)


def component_parameter_names(component: Any) -> tuple[str, ...]:
    return _fit_config_helper("component_parameter_names")(component)


def _dataset_temperature(data: PointData4D) -> float | np.ndarray:
    return _fit_config_helper("_dataset_temperature")(data)


def _is_elastic_dataset(data: PointData4D) -> bool:
    return bool(_fit_config_helper("_is_elastic_dataset")(data))


def _powder_sphere_directions(count: int) -> np.ndarray:
    return _fit_config_helper("_powder_sphere_directions")(count)


def _scalar_bulk_observable(*args: Any, **kwargs: Any) -> np.ndarray:
    return _fit_config_helper("_scalar_bulk_observable")(*args, **kwargs)


def _quasistatic_model_observable(*args: Any, **kwargs: Any) -> np.ndarray:
    return _fit_config_helper("_quasistatic_model_observable")(*args, **kwargs)


def _spectral_model_observable(*args: Any, **kwargs: Any) -> np.ndarray:
    return _fit_config_helper("_spectral_model_observable")(*args, **kwargs)


def lindhard_factory(
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
    from .performance import operation_batch_bytes, scientific_memory_limit_bytes

    # Project recipes retain these legacy fields for compatibility, but GUI
    # execution uses the machine-local resource policy.
    workers = 0
    batch_bytes = operation_batch_bytes()
    transition_batch_bytes = operation_batch_bytes()
    transition_backend = str(
        config.get("response_transition_backend", "auto")
    )
    response_cache = ElectronicResponseCache(
        max_bytes=max(1, scientific_memory_limit_bytes() // 4),
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
