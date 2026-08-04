"""Explicit normalization of electronic susceptibilities.

Lindhard and electronic-RPA kernels use normalized Brillouin-zone weights and
return the total response of one electronic model cell. This module resolves
the divisor needed to report that response per model cell, formula unit, or
represented magnetic center without coupling the divisor to a magnetic form
factor.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .crystal import CrystalFormulaUnits, infer_crystal_formula_units

FloatArray = NDArray[np.float64]
ELECTRONIC_NORMALIZATION_BASES = (
    "per_model_cell",
    "per_formula_unit",
    "per_magnetic_ion",
    "per_unit_cell",
    "unknown",
)
LINDHARD_COUPLING_FIELDS = frozenset(
    {
        "formula_units_mode",
        "formula_units_per_cell",
        "magnetic_normalization_mode",
        "magnetic_normalization_species",
        "magnetic_centers_per_model_cell",
        "form_factor_mode",
        "ion",
        "form_factor_coefficients",
        "form_factor_g_J",
        "form_factor_j2_coefficients",
        "form_factor_mixture",
        "bulk_g_factor",
    }
)


@dataclass(frozen=True)
class ElectronicResponseNormalization:
    """Resolved conversion from a model-cell response to a declared basis."""

    target_basis: str
    entities_per_model_cell: float
    target_label: str = ""
    formula_units_per_model_cell: float | None = None
    magnetic_reference_species: str = ""
    magnetic_centers_per_model_cell: float | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.target_basis not in ELECTRONIC_NORMALIZATION_BASES:
            raise ValueError(f"unknown electronic normalization basis {self.target_basis!r}")
        count = float(self.entities_per_model_cell)
        if not np.isfinite(count) or count <= 0.0:
            raise ValueError("entities_per_model_cell must be finite and positive")

    @property
    def scale_from_model_cell(self) -> float:
        """Multiplicative scale converting a model-cell response to the target."""

        return 1.0 / float(self.entities_per_model_cell)

    def apply(self, values: ArrayLike) -> FloatArray:
        """Return ``values`` normalized on the resolved target basis."""

        return np.asarray(values) * self.scale_from_model_cell

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready normalization provenance."""

        return {
            "source_basis": "per_model_cell",
            "target_basis": self.target_basis,
            "target_label": self.target_label,
            "entities_per_model_cell": float(self.entities_per_model_cell),
            "scale_from_model_cell": self.scale_from_model_cell,
            "formula_units_per_model_cell": self.formula_units_per_model_cell,
            "magnetic_reference_species": self.magnetic_reference_species,
            "magnetic_centers_per_model_cell": self.magnetic_centers_per_model_cell,
            "provenance": self.provenance,
        }


def configure_lindhard_experimental_coupling(
    component: Any,
    **updates: Any,
) -> dict[str, Any]:
    """Atomically configure normalization and a shared form-factor profile.

    This GUI-independent operation is the scripting counterpart of the
    Lindhard **Experimental coupling** panel. It returns the installed
    configuration values and leaves the component unchanged on validation
    failure.
    """

    if getattr(component, "type", None) != "lindhard":
        raise TypeError("experimental electronic coupling requires a lindhard component")
    unknown = set(updates) - LINDHARD_COUPLING_FIELDS
    if unknown:
        raise KeyError(f"unknown Lindhard coupling field {sorted(unknown)[0]!r}")
    current = getattr(component, "config", {})
    candidate = dict(current) if isinstance(current, Mapping) else {}
    candidate.update(updates)
    formula_mode = str(candidate.get("formula_units_mode", "manual")).strip().lower()
    if formula_mode not in {"auto", "manual"}:
        raise ValueError("formula_units_mode must be auto or manual")
    if formula_mode == "manual":
        formula_count = float(candidate.get("formula_units_per_cell", 1.0))
        if not np.isfinite(formula_count) or formula_count <= 0.0:
            raise ValueError("formula_units_per_cell must be finite and positive")
    magnetic_mode = str(
        candidate.get("magnetic_normalization_mode", "auto")
    ).strip().lower()
    if magnetic_mode not in {"auto", "manual"}:
        raise ValueError("magnetic_normalization_mode must be auto or manual")
    if magnetic_mode == "manual":
        magnetic_count = float(
            candidate.get("magnetic_centers_per_model_cell", 1.0)
        )
        if not np.isfinite(magnetic_count) or magnetic_count <= 0.0:
            raise ValueError(
                "magnetic_centers_per_model_cell must be finite and positive"
            )
    bulk_g = float(candidate.get("bulk_g_factor", 2.0))
    if not np.isfinite(bulk_g) or bulk_g <= 0.0:
        raise ValueError("bulk_g_factor must be finite and positive")
    from .form_factors import magnetic_form_factor_profile

    magnetic_form_factor_profile(np.asarray([0.0]), candidate)
    installed = {name: deepcopy(candidate[name]) for name in updates}
    component.config.update(installed)
    return installed


def _element_symbol(species: str) -> str:
    match = re.match(r"^([A-Z][a-z]?)", str(species).strip())
    return "" if match is None else match.group(1)


def _represented_sites_by_species(model: Any) -> dict[str, set[str]]:
    represented: dict[str, set[str]] = {}
    for index, basis in enumerate(model.basis):
        species = str(getattr(basis, "species", "") or "").strip()
        if not species:
            continue
        site = str(getattr(basis, "site", "") or "").strip()
        if not site:
            center = tuple(np.round(np.asarray(model.orbital_centers[index]), 12))
            site = repr(center)
        represented.setdefault(species, set()).add(site)
    return represented


def _resolve_formula_units(
    model: Any,
    config: Mapping[str, Any],
    crystal: Mapping[str, Any] | None,
) -> tuple[float, CrystalFormulaUnits | None, str]:
    mode = str(config.get("formula_units_mode", "manual")).strip().lower()
    if mode == "manual":
        value = float(config.get("formula_units_per_cell", 1.0))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("formula_units_per_cell must be finite and positive")
        return value, None, "manual formula-unit count"
    if mode != "auto":
        raise ValueError("formula_units_mode must be auto or manual")
    if not isinstance(crystal, Mapping):
        raise ValueError(
            "automatic formula-unit normalization requires a complete linked crystal"
        )
    result = infer_crystal_formula_units(
        crystal,
        model_lattice=model.direct_lattice,
    )
    return (
        float(result.formula_units_per_model_cell),
        result,
        "crystal composition and electronic model-cell volume",
    )


def _resolve_magnetic_centers(
    model: Any,
    config: Mapping[str, Any],
) -> tuple[str, float, str]:
    mode = str(config.get("magnetic_normalization_mode", "auto")).strip().lower()
    requested = str(config.get("magnetic_normalization_species", "") or "").strip()
    if mode == "manual":
        count = float(config.get("magnetic_centers_per_model_cell", 1.0))
        if not np.isfinite(count) or count <= 0.0:
            raise ValueError(
                "magnetic_centers_per_model_cell must be finite and positive"
            )
        return requested, count, "manual magnetic-center count"
    if mode != "auto":
        raise ValueError("magnetic_normalization_mode must be auto or manual")
    represented = _represented_sites_by_species(model)
    if not represented:
        raise ValueError(
            "automatic magnetic-center normalization requires species and site "
            "metadata on the tight-binding basis"
        )
    if requested:
        exact = [species for species in represented if species == requested]
        if not exact:
            requested_element = _element_symbol(requested)
            exact = [
                species
                for species in represented
                if requested_element and _element_symbol(species) == requested_element
            ]
        if len(exact) != 1:
            raise ValueError(
                f"magnetic normalization selection {requested!r} does not identify "
                "exactly one represented tight-binding species"
            )
        selected = exact[0]
    elif len(represented) == 1:
        selected = next(iter(represented))
    else:
        choices = ", ".join(sorted(represented))
        raise ValueError(
            "automatic magnetic-center normalization is ambiguous; select one "
            f"represented species ({choices}) or use a manual count"
        )
    return (
        selected,
        float(len(represented[selected])),
        "unique represented tight-binding sites",
    )


def resolve_electronic_response_normalization(
    model: Any,
    config: Mapping[str, Any],
    *,
    crystal: Mapping[str, Any] | None = None,
    target_basis: str = "per_model_cell",
    target_label: str = "",
) -> ElectronicResponseNormalization:
    """Resolve a dataset normalization without applying probe form factors.

    ``per_unit_cell`` means the electronic model cell. ``unknown`` preserves
    the model-cell ordinate and makes no absolute normalization claim.
    """

    basis = str(target_basis or "unknown").strip().lower()
    if basis not in ELECTRONIC_NORMALIZATION_BASES:
        raise ValueError(f"unknown electronic normalization basis {basis!r}")
    if basis in {"per_model_cell", "per_unit_cell", "unknown"}:
        return ElectronicResponseNormalization(
            target_basis=basis,
            target_label=str(target_label),
            entities_per_model_cell=1.0,
            provenance=(
                "electronic model cell"
                if basis != "unknown"
                else "normalization unspecified; model-cell response retained"
            ),
        )
    if basis == "per_formula_unit":
        count, _formula, provenance = _resolve_formula_units(model, config, crystal)
        return ElectronicResponseNormalization(
            target_basis=basis,
            target_label=str(target_label),
            entities_per_model_cell=count,
            formula_units_per_model_cell=count,
            provenance=provenance,
        )
    species, count, provenance = _resolve_magnetic_centers(model, config)
    return ElectronicResponseNormalization(
        target_basis=basis,
        target_label=str(target_label or species),
        entities_per_model_cell=count,
        magnetic_reference_species=species,
        magnetic_centers_per_model_cell=count,
        provenance=provenance,
    )
