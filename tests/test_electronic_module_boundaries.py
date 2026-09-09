"""Equivalence and dependency checks for the electronic module boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from nfit import electronic_builder
from nfit.electronic_builder import (
    manifold_symmetry_representation,
    orbital_manifold_preset,
)
from nfit.electronic_resolvers import (
    resolve_manifold_symmetry_representation,
    resolve_structure_first_model,
)
from nfit.electronic_spin import (
    spinor_manifold_representation,
    spinor_rotation_representation,
)
from nfit.electronic_symmetry import (
    manifold_symmetry_representation as shared_manifold_representation,
)

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "nfit"
ELECTRONIC_LAYERS = {
    "electronic_builder",
    "electronic_spin",
    "electronic_structure",
}


def _electronic_dependencies(module_name: str) -> set[str]:
    tree = ast.parse((SOURCE_ROOT / f"{module_name}.py").read_text())
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module in ELECTRONIC_LAYERS
    }


def test_electronic_builder_spin_and_structure_dependencies_are_acyclic():
    dependencies = {
        module: _electronic_dependencies(module)
        for module in ELECTRONIC_LAYERS
    }

    assert dependencies == {
        "electronic_builder": {"electronic_spin", "electronic_structure"},
        "electronic_spin": {"electronic_structure"},
        "electronic_structure": set(),
    }


def test_shared_manifold_representation_preserves_builder_results_and_payloads():
    manifold = orbital_manifold_preset("M1", "p")
    rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )

    expected = manifold_symmetry_representation(manifold, rotation)
    np.testing.assert_allclose(
        shared_manifold_representation(manifold, rotation),
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        manifold_symmetry_representation(manifold.to_dict(), rotation),
        expected,
        atol=1e-12,
    )


def test_spinor_representation_composes_shared_orbital_and_spin_actions():
    manifold = orbital_manifold_preset("M1", "d")
    rotation = np.diag((-1.0, -1.0, 1.0))

    expected = np.kron(
        shared_manifold_representation(manifold, rotation),
        spinor_rotation_representation(rotation),
    )
    np.testing.assert_allclose(
        spinor_manifold_representation(manifold, rotation),
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        spinor_manifold_representation(manifold.to_dict(), rotation),
        expected,
        atol=1e-12,
    )


def test_neutral_resolver_delegates_to_the_current_builder_implementation(
    monkeypatch,
):
    component = object()
    sentinel = object()
    monkeypatch.setattr(
        electronic_builder,
        "resolve_tight_binding_builder",
        lambda candidate: sentinel if candidate is component else None,
    )

    assert resolve_structure_first_model(component) is sentinel


def test_manifold_resolver_delegates_to_current_public_builder_wrapper(monkeypatch):
    manifold = object()
    rotation = np.eye(3)
    sentinel = np.asarray(((2.0, 0.0), (0.0, 3.0)), dtype=np.complex128)
    calls = []

    def replacement(candidate, operation):
        calls.append((candidate, operation))
        return sentinel

    monkeypatch.setattr(
        electronic_builder,
        "manifold_symmetry_representation",
        replacement,
    )

    assert resolve_manifold_symmetry_representation(manifold, rotation) is sentinel
    assert calls == [(manifold, rotation)]


def test_spinor_representation_uses_public_builder_dispatch(monkeypatch):
    payload = {"legacy": "mapping"}
    rotation = np.eye(3)
    orbital = np.asarray(((1.0, 0.0), (0.0, -1.0)), dtype=np.complex128)
    calls = []

    def replacement(candidate, operation):
        calls.append((candidate, operation))
        return orbital

    monkeypatch.setattr(
        electronic_builder,
        "manifold_symmetry_representation",
        replacement,
    )

    expected = np.kron(orbital, spinor_rotation_representation(rotation))
    np.testing.assert_allclose(
        spinor_manifold_representation(payload, rotation),
        expected,
        atol=1e-12,
    )
    assert calls == [(payload, rotation)]


def test_spinor_representation_preserves_builder_mapping_validation():
    rotation = np.eye(3)

    with pytest.raises(KeyError, match="site_label"):
        spinor_manifold_representation({}, rotation)
