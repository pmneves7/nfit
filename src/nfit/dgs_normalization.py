"""Incident-energy conventions for direct-geometry trajectory normalization."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_TRAJECTORY_ENERGY_POLICY = "first_run"
TRAJECTORY_ENERGY_POLICIES = (
    ("first_run", "First run Ei (Mantid MDNorm)"),
    ("per_run", "Each run Ei"),
)


def validated_trajectory_energy_policy(value: str) -> str:
    """Validate the convention used only for normalization trajectories."""
    if value not in {key for key, _label in TRAJECTORY_ENERGY_POLICIES}:
        raise ValueError("trajectory_energy_policy must be 'first_run' or 'per_run'")
    return value


def trajectory_incident_energies(
    config: Mapping[str, Any], incident_energies: Iterable[float],
    *, reference_energy: float | None = None,
) -> tuple[float, ...]:
    """Resolve trajectory Ei in meV, retaining each run's original energy bounds.

    The default uses the first participating run's Ei for every trajectory,
    matching Mantid MDNorm's first-experiment convention. ``per_run`` uses
    each run's scalar Ei. A shared positive override takes precedence. These
    choices do not alter raw-event reconstruction or stored MDE coordinates.
    No pulse-resolved incident energy is inferred here.
    """
    policy = validated_trajectory_energy_policy(
        config.get("trajectory_energy_policy", DEFAULT_TRAJECTORY_ENERGY_POLICY)
    )
    energies = tuple(float(value) for value in incident_energies)
    override = config.get("incident_energy_override")
    if override is not None:
        override = float(override)
        if not math.isfinite(override) or override <= 0.0:
            raise ValueError("incident_energy_override must be positive and finite in meV")
        return (override,) * len(energies)
    if any(not math.isfinite(value) or value <= 0.0 for value in energies):
        raise ValueError("incident energies must be positive and finite in meV")
    if policy == "first_run" and energies:
        reference = energies[0] if reference_energy is None else float(reference_energy)
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError("trajectory reference energy must be positive and finite in meV")
        return (reference,) * len(energies)
    return energies


def set_dgs_trajectory_energy_policy(group, policy: str) -> None:
    """Set a saved raw-DGS or MDEvent group's normalization convention.

    Subsequent binnings use the chosen convention; existing histogram cache
    signatures become outdated. Reduced raw-event caches remain reusable
    because trajectory normalization does not change reconstructed events.
    """
    policy = validated_trajectory_energy_policy(policy)
    for key in ("mdevent", "raw_dgs"):
        config = group.metadata.get(key)
        if isinstance(config, dict) and config.get("format") in {
            "mantid-mdevent", "raw-direct-geometry-nexus"
        }:
            config["trajectory_energy_policy"] = policy
            return
    raise ValueError("trajectory energy policies require a raw DGS or MDEvent group")


def trajectory_normalization_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Include the effective default so older histogram caches are invalidated."""
    return {
        **config,
        "trajectory_energy_policy": validated_trajectory_energy_policy(
            config.get("trajectory_energy_policy", DEFAULT_TRAJECTORY_ENERGY_POLICY)
        ),
    }
