"""Neutral resolution seams for optional electronic model builders."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def resolve_structure_first_model(component: Any) -> Any:
    """Resolve a structure-first model through the installed builder layer.

    Importing the implementation only when builder-backed state is encountered
    keeps the core electronic model independent of the editable builder while
    retaining the existing lazy resolution behavior.
    """

    builder = import_module(".electronic_builder", package=__package__)
    return builder.resolve_tight_binding_builder(component)


def resolve_manifold_symmetry_representation(
    manifold: Any,
    rotation_cartesian: Any,
) -> Any:
    """Resolve a manifold symmetry action through the public builder seam.

    The lazy import keeps :mod:`electronic_spin` independent of the builder at
    module-import time while preserving the builder wrapper's payload
    validation and runtime replacement behavior.
    """

    builder = import_module(".electronic_builder", package=__package__)
    return builder.manifold_symmetry_representation(manifold, rotation_cartesian)
