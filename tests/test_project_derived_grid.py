from __future__ import annotations

import copy

import numpy as np
import pytest

from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.project_derived_grid import _loaded_data_bounds, plan_shared_derived_grid


def _auto_config() -> dict:
    return {
        "axes": [
            {
                "name": "H",
                "vector": [1.0],
                "lower": -1.0,
                "upper": 2.0,
                "step_size": 1.0,
                "num_bins": 4,
                "mode": "step",
                "auto_lower": True,
                "auto_lower_value": -1.0,
                "auto_upper": False,
            }
        ]
    }


def test_explicit_grid_does_not_resolve_or_load_sources():
    config = _auto_config()
    config["axes"][0]["auto_lower"] = False

    result = plan_shared_derived_grid(
        config,
        ["left", "right"],
        resolve_source=lambda _source: pytest.fail("explicit grids need no source traversal"),
        load_dataset=lambda _dataset: pytest.fail("explicit grids need no source loading"),
        event_bounds=lambda *_args: pytest.fail("explicit grids need no event scan"),
        symmetry_bounds=lambda bounds, _axes: bounds,
    )

    assert result == config
    assert result is not config


def test_hierarchical_event_components_are_scanned_without_dataset_loading():
    config = _auto_config()
    cold = object()
    warm = object()
    calls = []

    class EventNode:
        def __init__(self, name):
            self.name = name
            self.metadata = {"mdevent": {}}

    cold_node = EventNode("cold-node")
    warm_node = EventNode("warm-node")

    def event_bounds(node, datasets, _axes):
        calls.append((node, tuple(datasets)))
        return [(-2.0, 0.0)] if node is cold_node else [(0.0, 3.0)]

    result = plan_shared_derived_grid(
        config,
        ["hierarchy"],
        resolve_source=lambda _source: {
            "components": [
                {"node": cold_node, "datasets": [cold]},
                {"node": warm_node, "datasets": [warm]},
            ]
        },
        load_dataset=lambda _dataset: pytest.fail("MDEvent bounds must not load event data"),
        event_bounds=event_bounds,
        symmetry_bounds=lambda bounds, _axes: bounds,
    )

    assert calls == [(cold_node, (cold,)), (warm_node, (warm,))]
    assert result["axes"][0]["lower"] == -2.0
    assert result["axes"][0]["upper"] == 2.0


def test_partial_auto_resolution_preserves_explicit_limit_and_saved_recipe():
    config = _auto_config()
    saved = copy.deepcopy(config)

    result = plan_shared_derived_grid(
        config,
        ["left", "right"],
        resolve_source=lambda source: {"dataset": source},
        load_dataset=lambda source: MDHistoData(
            (MDHistoAxis("H", np.array([-2.5, -1.5]) if source == "left" else np.array([0.5, 1.5]), "rlu", "momentum"),),
            np.ones(1), np.ones(1), np.zeros(1, dtype=bool), np.ones(1),
        ),
        event_bounds=lambda *_args: pytest.fail("histograms do not use event bounds"),
        symmetry_bounds=lambda bounds, _axes: bounds,
    )

    assert config == saved
    assert result["axes"][0]["lower"] < saved["axes"][0]["lower"]
    assert result["axes"][0]["upper"] == 2.0
    assert result["axes"][0]["auto_lower"] is False


def test_symmetry_expansion_participates_in_shared_bounds():
    config = _auto_config()
    data = MDHistoData(
        (MDHistoAxis("H", np.array([0.5, 1.5]), "rlu", "momentum"),),
        np.ones(1), np.ones(1), np.zeros(1, dtype=bool), np.ones(1),
    )

    result = plan_shared_derived_grid(
        config,
        ["positive"],
        resolve_source=lambda _source: {"dataset": data},
        load_dataset=lambda dataset: dataset,
        event_bounds=lambda *_args: pytest.fail("histograms do not use event bounds"),
        symmetry_bounds=lambda bounds, _axes: [(-bounds[0][1], bounds[0][1])],
    )

    assert result["axes"][0]["lower"] < 0.0


def test_histogram_bounds_do_not_hide_invalid_output_basis():
    axes = [
        {"name": "H", "vector": [1.0, 0.0]},
        {"name": "K", "vector": [1.0, 0.0]},
    ]
    data = MDHistoData(
        (
            MDHistoAxis("H", np.array([0.0, 1.0]), "rlu", "momentum"),
            MDHistoAxis("K", np.array([0.0, 1.0]), "rlu", "momentum"),
        ),
        np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1), dtype=bool), np.ones((1, 1)),
    )

    with pytest.raises(ValueError, match="invertible basis"):
        _loaded_data_bounds(data, axes)


def test_automatic_dynamic_axis_fails_before_source_work():
    config = _auto_config()
    config["axes"][0]["mode"] = "discrete"

    with pytest.raises(ValueError, match="requires shared candidate centers"):
        plan_shared_derived_grid(
            config,
            ["left", "right"],
            resolve_source=lambda _source: pytest.fail("validation must precede traversal"),
            load_dataset=lambda _dataset: pytest.fail("validation must precede loading"),
            event_bounds=lambda *_args: pytest.fail("validation must precede scans"),
            symmetry_bounds=lambda bounds, _axes: bounds,
        )
