"""Saved and scriptable choices for native DGS and MDE event handling."""

from __future__ import annotations

import keyword

from .dgs_normalization import (
    DEFAULT_TRAJECTORY_ENERGY_POLICY,
    validated_trajectory_energy_policy,
)
from .dgs_reduction_policy import (
    DEFAULT_EVENT_PRECISION_POLICY,
    DEFAULT_MONITOR_VARIANCE_POLICY,
    DEFAULT_SYMMETRY_VARIANCE_POLICY,
    validated_event_precision_policy,
    validated_monitor_variance_policy,
    validated_symmetry_variance_policy,
)


def _reduction_config(group):
    for key, format_name in (
        ("raw_dgs", "raw-direct-geometry-nexus"),
        ("mdevent", "mantid-mdevent"),
    ):
        config = group.metadata.get(key)
        if isinstance(config, dict) and config.get("format") == format_name:
            return config, key == "raw_dgs"
    raise ValueError("DGS reduction policies require a raw DGS or MDEvent group")


def set_dgs_reduction_policies(
    group,
    *,
    monitor_variance_policy=None,
    event_precision_policy=None,
    symmetry_variance_policy=None,
) -> None:
    """Save validated choices used by subsequent reductions and histograms.

    Monitor fitting applies only to raw runs. Precision changes may invalidate
    reduced-event caches; histogram variance changes retain reusable events.
    Existing histogram signatures include the effective policy defaults.
    Validation is atomic: an invalid choice leaves all settings unchanged.
    """
    config, raw = _reduction_config(group)
    choices = {}
    if monitor_variance_policy is not None:
        if not raw:
            raise ValueError("monitor fitting cannot change coordinates stored in MDEvent files")
        choices["monitor_variance_policy"] = validated_monitor_variance_policy(monitor_variance_policy)
    if event_precision_policy is not None:
        choices["event_precision_policy"] = validated_event_precision_policy(event_precision_policy)
    if symmetry_variance_policy is not None:
        choices["symmetry_variance_policy"] = validated_symmetry_variance_policy(symmetry_variance_policy)
    config.update(choices)


def dgs_reduction_policy_script(group, *, group_variable="group") -> str:
    """Export the complete policy choices as an editable public-API snippet.

    ``group_variable`` names an already constructed raw-DGS or MDEvent group.
    This exports policies, not the group's source import and binning recipe.
    """
    if not group_variable.isidentifier() or keyword.iskeyword(group_variable):
        raise ValueError("group_variable must be a Python variable name")
    config, raw = _reduction_config(group)
    policies = {
        "event_precision_policy": validated_event_precision_policy(
            config.get("event_precision_policy", DEFAULT_EVENT_PRECISION_POLICY)
        ),
        "symmetry_variance_policy": validated_symmetry_variance_policy(
            config.get("symmetry_variance_policy", DEFAULT_SYMMETRY_VARIANCE_POLICY)
        ),
    }
    if raw:
        policies["monitor_variance_policy"] = validated_monitor_variance_policy(
            config.get("monitor_variance_policy", DEFAULT_MONITOR_VARIANCE_POLICY)
        )
    energy = validated_trajectory_energy_policy(
        config.get("trajectory_energy_policy", DEFAULT_TRAJECTORY_ENERGY_POLICY)
    )
    lines = [
        "from nfit import set_dgs_reduction_policies, set_dgs_trajectory_energy_policy",
        "",
        f"set_dgs_reduction_policies({group_variable},",
        *(f"    {key}={value!r}," for key, value in policies.items()),
        ")",
        f"set_dgs_trajectory_energy_policy({group_variable}, {energy!r})",
        "",
    ]
    return "\n".join(lines)
