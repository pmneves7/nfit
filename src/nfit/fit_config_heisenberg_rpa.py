"""Heisenberg-RPA evaluator state used by :mod:`nfit.fit_config`.

The evaluator is isolated here because its geometry, closure, tensor, Zeeman,
and diagnostic paths form a cohesive subsystem.  Small delegates keep the
long-standing fit-config compatibility seams authoritative.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from .cross_section import bose_denominator
from .dataset import PointData4D
from .spin_fluctuations import (
    build_rpa_geometry,
    heisenberg_rpa_chipp,
    heisenberg_rpa_chipp_and_gradients,
    heisenberg_rpa_susceptibility,
    reduce_site_network,
    reduce_site_network_with_tensors,
    rpa_exchange_matrix,
)
from .susceptibility import ScalarSusceptibilityResponse

_FIT_CONFIG_HELPER_NAMES = (
    "ISOTROPIC_POLARIZATION",
    "_component_has_tensor_terms",
    "_component_has_zeeman",
    "_dataset_magnetic_field",
    "_dataset_temperature",
    "_form_factor_from_config",
    "_form_factor_sq_from_config",
    "_is_elastic_dataset",
    "_powder_reciprocal_matrix",
    "_powder_sphere_directions",
    "_quasistatic_model_observable",
    "_scalar_bulk_observable",
    "_spectral_model_observable",
    "heisenberg_rpa_orbit_labels",
    "heisenberg_rpa_parameter_labels",
    "qualified_parameter_name",
)
_FIT_CONFIG_CONTEXT: Mapping[str, Any] = {}


def configure_fit_helpers(context: Mapping[str, Any]) -> None:
    """Install the narrow set of shared fit helpers used by this evaluator."""

    missing = set(_FIT_CONFIG_HELPER_NAMES).difference(context)
    if missing:
        raise KeyError(f"missing Heisenberg-RPA fit helper {sorted(missing)[0]!r}")
    global _FIT_CONFIG_CONTEXT
    _FIT_CONFIG_CONTEXT = context


def _fit_config_helper(name: str) -> Any:
    try:
        return _FIT_CONFIG_CONTEXT[name]
    except KeyError as exc:
        raise RuntimeError("Heisenberg-RPA fit helpers have not been configured") from exc


def _isotropic_polarization() -> float:
    """Return the live public polarization convention from :mod:`fit_config`."""

    return _fit_config_helper("ISOTROPIC_POLARIZATION")


def qualified_parameter_name(component_name: str, parameter: str) -> str:
    return _fit_config_helper("qualified_parameter_name")(component_name, parameter)


def heisenberg_rpa_orbit_labels(component: Any) -> tuple[str, ...]:
    return _fit_config_helper("heisenberg_rpa_orbit_labels")(component)


def heisenberg_rpa_parameter_labels(component: Any) -> tuple[str, ...]:
    return _fit_config_helper("heisenberg_rpa_parameter_labels")(component)


def _component_has_tensor_terms(config: Mapping[str, Any]) -> bool:
    return bool(_fit_config_helper("_component_has_tensor_terms")(config))


def _component_has_zeeman(config: Mapping[str, Any]) -> bool:
    return bool(_fit_config_helper("_component_has_zeeman")(config))


def _dataset_temperature(data: PointData4D) -> float | np.ndarray:
    return _fit_config_helper("_dataset_temperature")(data)


def _dataset_magnetic_field(data: PointData4D) -> np.ndarray:
    return _fit_config_helper("_dataset_magnetic_field")(data)


def _powder_reciprocal_matrix(data: PointData4D, lattice: Any) -> np.ndarray:
    return _fit_config_helper("_powder_reciprocal_matrix")(data, lattice)


def _powder_sphere_directions(count: int) -> np.ndarray:
    return _fit_config_helper("_powder_sphere_directions")(count)


def _form_factor_sq_from_config(component: Any, data: PointData4D) -> np.ndarray:
    return _fit_config_helper("_form_factor_sq_from_config")(component, data)


def _form_factor_from_config(component: Any, data: PointData4D) -> float | np.ndarray:
    return _fit_config_helper("_form_factor_from_config")(component, data)


def _is_elastic_dataset(data: PointData4D) -> bool:
    return bool(_fit_config_helper("_is_elastic_dataset")(data))


def _scalar_bulk_observable(*args: Any, **kwargs: Any) -> np.ndarray:
    return _fit_config_helper("_scalar_bulk_observable")(*args, **kwargs)


def _quasistatic_model_observable(*args: Any, **kwargs: Any) -> np.ndarray:
    return _fit_config_helper("_quasistatic_model_observable")(*args, **kwargs)


def _spectral_model_observable(*args: Any, **kwargs: Any) -> np.ndarray:
    return _fit_config_helper("_spectral_model_observable")(*args, **kwargs)


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
                polarization=_isotropic_polarization(),
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
                polarization = _isotropic_polarization()
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
