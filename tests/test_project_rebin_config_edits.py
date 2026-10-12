from __future__ import annotations

import ast
import copy
import inspect
import math
import textwrap

import numpy as np
import pytest

from nfit import project_gui, project_rebinning


class RecordingExplorer(project_gui.NfitProjectExplorer):
    """Exercise the existing setter hooks without constructing Qt widgets."""

    def __init__(self):
        self.events = []

    def _after_dataset_rebin_changed(self, dataset, group):
        self.events.append(("dataset", dataset, group))

    def _after_group_composite_changed(self, group):
        self.events.append(("composite", group))

    def _sync_details(self):
        self.events.append(("sync",))

    def _set_dataset_details_preserving_scroll(self, dataset, group):
        self.events.append(("details", dataset, group))


@pytest.fixture
def editing(monkeypatch):
    config = {
        "normalize": False,
        "resolution_mode": "step",
        "axes": [{"lower": 0.0, "upper": 4.0, "num_bins": 5,
                  "step_size": 1.0, "mode": "step", "fractional": True}],
    }
    monkeypatch.setattr(project_gui, "dataset_rebin_config", lambda _: config)
    monkeypatch.setattr(project_gui, "data_group_composite_config", lambda _: config)
    return RecordingExplorer(), config, object(), object()


def _edit_value(editing, composite, key, text, index=0):
    explorer, _, dataset, group = editing
    if composite:
        explorer._set_group_composite_axis_value(group, index, key, text)
    else:
        explorer._set_dataset_rebin_axis_value(dataset, group, index, key, text)


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("key,text,expected", [
    ("lower", "-1", {"lower": -1.0, "num_bins": 6, "step_size": 1.0,
                     "auto_lower": False}),
    ("upper", "5.9", {"upper": 5.9, "num_bins": 6, "step_size": 1.0,
                     "auto_upper": False}),
    ("num_bins", "3.9", {"num_bins": 3, "step_size": 2.0}),
    ("num_bins", "0", {"num_bins": 1, "step_size": 4.0}),
    ("step_size", "1.5", {"step_size": 1.5, "num_bins": 3,
                         "auto_step_size": False}),
])
def test_numeric_axis_edits_preserve_values_identity_and_change_hooks(
    editing, composite, key, text, expected
):
    explorer, config, dataset, group = editing
    old_axis = config["axes"][0]
    before = copy.deepcopy(old_axis)
    _edit_value(editing, composite, key, text)
    assert config["axes"][0] == {**before, **expected}
    assert (config["axes"][0] is old_axis) is not composite
    assert config["normalize"] is composite
    assert explorer.events == ([("composite", group)] if composite else
                               [("dataset", dataset, group)])


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("text", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_steps_retain_dataset_and_composite_behavior(editing, composite, text):
    explorer, config, dataset, group = editing
    axis = config["axes"][0]
    before = copy.deepcopy(config)
    _edit_value(editing, composite, "step_size", text)
    if not composite:
        assert config == before
        assert config["axes"][0] is axis
        assert explorer.events == []
    else:
        expected_bins = 1 if text == "inf" else 5
        expected_step = 4.0 if text == "inf" else 1.0
        assert config["axes"][0] == {
            **axis, "num_bins": expected_bins, "step_size": expected_step,
            "auto_step_size": False,
        }
        assert config["axes"][0] is not axis
        assert config["normalize"] is True
        assert explorer.events == [("composite", group)]


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("key,text", [("lower", "bad"), ("num_bins", "bad"),
                                     ("num_bins", "nan"), ("step_size", "bad")])
def test_malformed_numeric_text_preserves_refresh_difference(editing, composite, key, text):
    explorer, config, _, _ = editing
    before = copy.deepcopy(config)
    _edit_value(editing, composite, key, text)
    assert config == before
    assert explorer.events == ([("sync",)] if composite else [])


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("key", ["lower", "upper", "num_bins", "step_size"])
def test_unchanged_numeric_edits_do_not_invalidate(editing, composite, key):
    explorer, config, _, _ = editing
    before = copy.deepcopy(config)
    _edit_value(editing, composite, key, str(config["axes"][0][key]))
    assert config == before
    assert explorer.events == []


@pytest.mark.parametrize("composite", [False, True])
def test_blank_bound_uses_auto_value_without_normalizing_or_recalculating(editing, composite):
    explorer, config, dataset, group = editing
    axis = config["axes"][0]
    axis.update(auto_lower=False, auto_lower_value=-2.0)
    _edit_value(editing, composite, "lower", " ")
    assert config["axes"][0] == {**axis, "lower": -2.0, "auto_lower": True}
    assert config["axes"][0]["num_bins"] == 5
    assert config["normalize"] is False
    assert explorer.events == ([("composite", group)] if composite else
                               [("dataset", dataset, group)])
    explorer.events.clear()
    _edit_value(editing, composite, "lower", "")
    assert explorer.events == []


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("mode", ["step", "bins"])
def test_bounds_keep_existing_resolution_behavior(editing, composite, mode):
    _, config, _, _ = editing
    config["axes"][0]["mode"] = mode
    _edit_value(editing, composite, "upper", "6")
    assert config["axes"][0]["num_bins"] == (7 if mode == "step" else 5)
    assert config["axes"][0]["step_size"] == (1.0 if mode == "step" else 1.5)


@pytest.mark.parametrize("composite", [False, True])
def test_numeric_edit_invalid_index_is_ignored(editing, composite):
    explorer, config, _, _ = editing
    before = copy.deepcopy(config)
    _edit_value(editing, composite, "lower", "-1", index=7)
    assert config == before
    assert explorer.events == []


@pytest.mark.parametrize("composite", [False, True])
def test_invalid_old_step_preserves_existing_bound_edit_difference(editing, composite):
    _, config, _, _ = editing
    config["axes"][0].update(step_size=-1.0, tolerance=0.0)
    _edit_value(editing, composite, "upper", "6")
    assert config["axes"][0]["num_bins"] == 5
    assert config["axes"][0]["step_size"] == 1.5
    assert config["axes"][0]["tolerance"] == 1.5


@pytest.mark.parametrize("composite", [False, True])
def test_nonfinite_bounds_preserve_error_and_partial_edit_behavior(editing, composite):
    explorer, config, _, _ = editing
    before = copy.deepcopy(config)
    if composite:
        with pytest.raises(ValueError):
            _edit_value(editing, True, "upper", "nan")
        assert config == before
    else:
        _edit_value(editing, False, "upper", "nan")
        assert math.isnan(config["axes"][0]["upper"])
        assert config["axes"][0]["auto_upper"] is False
    assert explorer.events == []


def _call_setter(editing, composite, suffix, *args):
    explorer, _, dataset, group = editing
    if composite:
        getattr(explorer, "_set_group_composite_" + suffix)(group, *args)
    else:
        getattr(explorer, "_set_dataset_rebin_" + suffix)(dataset, group, *args)


def _change_events(editing, composite, *, details=False):
    _, _, dataset, group = editing
    events = [("composite", group)] if composite else [("dataset", dataset, group)]
    if details and not composite:
        events.append(("details", dataset, group))
    return events


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("mode", ["bins", "edges", "discrete", "tolerance"])
def test_axis_mode_retains_defaults_and_dataset_detail_refresh(editing, composite, mode):
    explorer, config, _, _ = editing
    axis = config["axes"][0]
    _call_setter(editing, composite, "axis_mode", 0, mode)
    assert config["axes"][0] is axis
    assert axis["mode"] == mode
    assert axis["fractional"] is (mode not in {"discrete", "tolerance"})
    if mode == "tolerance":
        assert axis["tolerance"] == 1.0
    assert explorer.events == _change_events(editing, composite, details=True)
    explorer.events.clear()
    _call_setter(editing, composite, "axis_mode", 0, mode)
    assert explorer.events == []


@pytest.mark.parametrize("composite", [False, True])
def test_unknown_axis_mode_falls_back_to_step_without_clearing_tolerance(editing, composite):
    _, config, _, _ = editing
    config["axes"][0].update(mode="tolerance", tolerance=0.2, fractional=False)
    _call_setter(editing, composite, "axis_mode", 0, "unknown")
    assert config["axes"][0]["mode"] == "step"
    assert config["axes"][0]["tolerance"] == 0.2
    assert config["axes"][0]["fractional"] is False


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("mode", ["step", "bins", "discrete", "tolerance"])
def test_fractional_assignment_retains_allowed_modes(editing, composite, mode):
    explorer, config, _, _ = editing
    axis = config["axes"][0]
    axis.update(mode=mode, fractional=False)
    _call_setter(editing, composite, "axis_fractional", 0, True)
    allowed = mode not in {"discrete", "tolerance"}
    assert axis["fractional"] is allowed
    assert explorer.events == (_change_events(editing, composite) if allowed else [])


@pytest.mark.parametrize("composite", [False, True])
def test_explicit_edges_are_in_place_and_clearing_always_notifies(editing, composite):
    explorer, config, _, _ = editing
    axis = config["axes"][0]
    _call_setter(editing, composite, "axis_edges", 0, "[0, 1, 3]")
    assert config["axes"][0] is axis
    assert axis["bin_edges"] == [0.0, 1.0, 3.0]
    assert (axis["lower"], axis["upper"], axis["num_bins"], axis["step_size"]) == (
        0.5, 2.0, 2, 1.5,
    )
    assert explorer.events == _change_events(editing, composite)
    for _ in range(2):
        explorer.events.clear()
        _call_setter(editing, composite, "axis_edges", 0, "")
        assert "bin_edges" not in axis
        assert explorer.events == _change_events(editing, composite)


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("text", ["[0, 0, 1]", "[1]", "bad", "[[0, 1]]"])
def test_invalid_explicit_edges_keep_configuration_and_restore_editor(editing, composite, text):
    explorer, config, dataset, group = editing
    before = copy.deepcopy(config)
    _call_setter(editing, composite, "axis_edges", 0, text)
    assert config == before
    assert explorer.events == ([("sync",)] if composite else
                               [("details", dataset, group)])


@pytest.mark.parametrize("composite", [False, True])
def test_legacy_resolution_switch_retains_axis_modes_and_refreshes_only_on_change(
    editing, composite
):
    explorer, config, _, _ = editing
    axis = config["axes"][0]
    axis["step_size"] = 2.0
    _call_setter(editing, composite, "resolution_mode", "bins")
    assert config["resolution_mode"] == "bins"
    assert axis["mode"] == "step"
    assert axis["step_size"] == 1.0
    assert explorer.events == _change_events(editing, composite)
    explorer.events.clear()
    _call_setter(editing, composite, "resolution_mode", "bins")
    assert explorer.events == []
    _call_setter(editing, composite, "resolution_mode", "unknown")
    assert config["resolution_mode"] == "step"
    assert explorer.events == _change_events(editing, composite)


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("key,text", [("lower", "-1"), ("lower", ""),
                                     ("upper", "5"), ("num_bins", "3"),
                                     ("step_size", "0"), ("step_size", "1.5"),
                                     ("step_size", "bad"), ("num_bins", "5")])
def test_numeric_service_and_gui_configuration_edits_agree(editing, composite, key, text):
    _, config, _, _ = editing
    direct = copy.deepcopy(config)
    service = (project_rebinning.edit_composite_rebin_axis_value if composite else
               project_rebinning.edit_dataset_rebin_axis_value)
    result = service(direct, 0, key, text)
    _edit_value(editing, composite, key, text)
    assert config == direct
    expected_events = {
        project_rebinning.RebinAxisEditResult.CHANGED: _change_events(editing, composite),
        project_rebinning.RebinAxisEditResult.UNCHANGED: [],
        project_rebinning.RebinAxisEditResult.INVALID: [("sync",)] if composite else [],
    }
    assert editing[0].events == expected_events[result]


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("suffix,args,function", [
    ("axis_mode", (0, "tolerance"), "set_rebin_axis_mode"),
    ("axis_fractional", (0, False), "set_rebin_axis_fractional"),
    ("axis_edges", (0, "[0, 1, 3]"), "set_rebin_axis_edges"),
    ("resolution_mode", ("bins",), "set_rebin_resolution_mode"),
])
def test_shared_service_and_gui_configuration_edits_agree(
    editing, composite, suffix, args, function
):
    _, config, _, _ = editing
    direct = copy.deepcopy(config)
    service_args = (0, [0, 1, 3]) if suffix == "axis_edges" else args
    assert getattr(project_rebinning, function)(direct, *service_args)
    _call_setter(editing, composite, suffix, *args)
    assert config == direct


@pytest.mark.parametrize("prefix,function", [
    ("_set_dataset_rebin", "edit_dataset_rebin_axis_value"),
    ("_set_group_composite", "edit_composite_rebin_axis_value"),
])
def test_gui_axis_setters_delegate_configuration_ownership(prefix, function):
    for suffix, service_name in [
        ("axis_value", function), ("axis_edges", "set_rebin_axis_edges"),
        ("axis_mode", "set_rebin_axis_mode"),
        ("axis_fractional", "set_rebin_axis_fractional"),
        ("resolution_mode", "set_rebin_resolution_mode"),
    ]:
        setter = getattr(project_gui.NfitProjectExplorer, prefix + "_" + suffix)
        tree = ast.parse(textwrap.dedent(inspect.getsource(setter)))
        service_calls = [
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_project_rebinning"
        ]
        assert service_name in service_calls
        assert not any(isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript) for target in node.targets
        ) for node in ast.walk(tree))


def test_configuration_service_remains_gui_independent():
    tree = ast.parse(inspect.getsource(project_rebinning))
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                   for alias in node.names)
    assert not any(module.startswith(("project_gui", "project_data", "PySide", "PyQt", "qt_"))
                   for module in modules)


def test_edge_configuration_service_accepts_numpy_arrays(editing):
    _, config, _, _ = editing
    assert project_rebinning.set_rebin_axis_edges(config, 0, np.array([0.0, 1.0, 3.0]))
    assert config["axes"][0]["bin_edges"] == [0.0, 1.0, 3.0]
