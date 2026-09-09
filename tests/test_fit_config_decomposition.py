from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from nfit import fit_config, model_registry
from nfit.dataset import PointData4D
from nfit.fit_config_heisenberg_rpa import _RpaComponentEvaluator
from nfit.pipeline import ModelComponentSpec


def _top_level_node(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )


def _imported_modules(path: Path) -> set[str]:
    imported = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
            else:
                imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_large_electronic_evaluators_live_outside_fit_config():
    source_dir = Path(fit_config.__file__).parent
    lindhard_wrapper = _top_level_node(
        source_dir / "fit_config.py", "_lindhard_factory"
    )
    fit_tree = ast.parse((source_dir / "fit_config.py").read_text())

    assert lindhard_wrapper.end_lineno - lindhard_wrapper.lineno < 25
    assert not any(
        isinstance(node, ast.ClassDef) and node.name == "_RpaComponentEvaluator"
        for node in fit_tree.body
    )
    assert _top_level_node(
        source_dir / "fit_config_electronic.py", "lindhard_factory"
    )
    assert _top_level_node(
        source_dir / "fit_config_heisenberg_rpa.py", "_RpaComponentEvaluator"
    )
    assert "fit_config" not in _imported_modules(
        source_dir / "fit_config_electronic.py"
    )
    assert "fit_config" not in _imported_modules(
        source_dir / "fit_config_heisenberg_rpa.py"
    )
    assert "model_registry" not in _imported_modules(
        source_dir / "model_registry_electronic.py"
    )


def test_heisenberg_rpa_compatibility_factory_matches_focused_evaluator():
    component = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"chi0": 0.3, "gamma0": 2.0, "J1": 0.1},
        fit_parameters={},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {
                    "label": "J1",
                    "bonds": [
                        {"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}
                    ],
                }
            ],
        },
    )
    data = PointData4D(
        H=np.linspace(0.0, 1.0, 12),
        K=np.zeros(12),
        L=np.zeros(12),
        E=np.linspace(0.5, 4.0, 12),
        intensity=np.zeros(12),
        sigma=np.ones(12),
        temperature=10.0,
    )
    params = {
        "M.chi0": 0.3,
        "M.gamma0": 2.0,
        "M.inverse_mode_energy_sq": 0.0,
        "M.J1": 0.1,
    }

    compatibility_value = fit_config._heisenberg_rpa_factory(component)(data, params)
    focused_value = _RpaComponentEvaluator(component).value(data, params)

    assert fit_config._RpaComponentEvaluator is _RpaComponentEvaluator
    np.testing.assert_array_equal(compatibility_value, focused_value)


def test_heisenberg_rpa_evaluator_uses_live_public_polarization(monkeypatch):
    component = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"chi0": 0.3, "gamma0": 2.0, "J1": 0.1},
        fit_parameters={},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {
                    "label": "J1",
                    "bonds": [
                        {"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}
                    ],
                }
            ],
        },
    )
    data = PointData4D(
        H=np.linspace(0.0, 1.0, 12),
        K=np.zeros(12),
        L=np.zeros(12),
        E=np.linspace(0.5, 4.0, 12),
        intensity=np.zeros(12),
        sigma=np.ones(12),
        temperature=10.0,
    )
    received = {}

    def fake_spectral_observable(data_arg, chipp, **kwargs):
        received.update(kwargs)
        return np.ones(data_arg.size, dtype=float)

    monkeypatch.setattr(fit_config, "ISOTROPIC_POLARIZATION", 1.25)
    monkeypatch.setattr(
        fit_config, "_spectral_model_observable", fake_spectral_observable
    )

    evaluator = _RpaComponentEvaluator(component)
    evaluator._intensity_per_chipp(data, 1.0)

    assert received["polarization"] == 1.25


def test_lindhard_compatibility_factory_forwards_complete_context(monkeypatch):
    from nfit import fit_config_electronic

    sentinel = object()
    component = object()
    components = {"band": object()}
    dressing_component = object()
    received = {}

    def fake_factory(component_arg, components_arg, **kwargs):
        received.update(
            component=component_arg,
            components=components_arg,
            **kwargs,
        )
        return sentinel

    monkeypatch.setattr(fit_config_electronic, "lindhard_factory", fake_factory)

    result = fit_config._lindhard_factory(
        component,
        components,
        dressing_component=dressing_component,
        dressing_kind="matrix",
    )

    assert result is sentinel
    assert received == {
        "component": component,
        "components": components,
        "dressing_component": dressing_component,
        "dressing_kind": "matrix",
    }


def test_electronic_registry_compatibility_wrappers_delegate(monkeypatch):
    from nfit import model_registry_electronic

    calls = []
    monkeypatch.setattr(
        model_registry_electronic,
        "register_lindhard_model",
        lambda _context: calls.append("lindhard"),
    )
    monkeypatch.setattr(
        model_registry_electronic,
        "register_electronic_rpa_models",
        lambda _context: calls.append("rpa"),
    )
    monkeypatch.setattr(
        model_registry_electronic,
        "register_tight_binding_model",
        lambda _context: calls.append("tight_binding"),
    )

    model_registry._register_electronic_structure_models()

    assert calls == ["lindhard", "rpa", "tight_binding"]


def test_electronic_registry_definitions_retain_historical_order():
    electronic_keys = tuple(
        key
        for key in model_registry.MODEL_TYPE_REGISTRY
        if model_registry.MODEL_TYPE_REGISTRY[key].category == "electronic_structure"
    )

    assert electronic_keys == (
        "lindhard",
        "stoner_rpa",
        "matrix_rpa",
        "hubbard_hund_rpa",
        "tight_binding",
    )
