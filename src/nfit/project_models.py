"""Project-level model configuration helpers.

This module contains model-state operations shared by project serialization,
workflow code, and the Qt project explorer.  Keeping them outside the GUI
module prevents persistence code from depending on Qt-facing implementation
details.
"""

from __future__ import annotations

from .fit_config import component_parameter_names
from .model_registry import model_definition
from .pipeline import ModelComponentSpec

# Default starting values for non-orbit dynamic parameters.  Parameters not
# listed here use zero.  Zeeman parameters must be nonzero to have an effect.
_MODEL_PARAMETER_DEFAULTS = {
    "g_factor": 2.0,
    "chi_perp_ratio": 1.0,
    "gamma_perp_ratio": 1.0,
    "m2_total": 1.0,
    "mode_coupling_u": 0.0,
    "total_amplitude": 1.0,
}


def reconcile_model_orbit_parameters(model: ModelComponentSpec) -> None:
    """Align a model's fit parameters with its configuration-derived terms.

    Tight binding delegates to its builder-state synchronizer. For Heisenberg
    RPA this covers exchange orbits plus enabled tensor, dipole, and Zeeman
    terms. Values of parameters whose names persist are kept; parameters of
    removed terms are dropped with their fit flags, limits, and sharing.
    """

    if model.type == "tight_binding":
        from .electronic_builder import (
            ensure_tight_binding_onsite_terms,
            reconcile_tight_binding_parameters,
        )

        ensure_tight_binding_onsite_terms(model)
        reconcile_tight_binding_parameters(model)
        return

    names = list(component_parameter_names(model))
    keep = set(names)
    for mapping in (
        model.parameters,
        model.fit_parameters,
        model.limits,
        model.sharing,
    ):
        if isinstance(mapping, dict):
            for key in [name for name in mapping if name not in keep]:
                mapping.pop(key)
    static = set(model_definition(model.type).parameters)
    for name in names:
        if name in static:
            continue
        model.parameters.setdefault(name, _MODEL_PARAMETER_DEFAULTS.get(name, 0.0))
        model.fit_parameters.setdefault(name, False)
        model.sharing.setdefault(name, {"mode": "global", "groups": {}})
