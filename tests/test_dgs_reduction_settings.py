"""Public, GUI and archive contracts for explicit native DGS policies."""

import copy
import json

import pytest

from nfit import (
    DataGroup,
    NfitProject,
    NfitProjectExplorer,
    bin_raw_dgs_group,
    dgs_reduction_policy_script,
    load_project,
    mdevent_dataset_group,
    raw_dgs,
    raw_dgs_dataset_group,
    save_project,
    set_dgs_reduction_policies,
    set_dgs_trajectory_energy_policy,
)
from nfit.project_composites import _composite_cache_signature, _composite_scope
from nfit.raw_dgs_cache import reduction_signature
from tests.test_mdevent import _write_mdevent
from tests.test_raw_dgs import _write_raw_dgs

OPTIONS = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[2]*4)


def _group(tmp_path, kind):
    source = tmp_path / f"{kind}.nxs"
    if kind == "raw_dgs":
        _write_raw_dgs(source, with_he3=True)
        return raw_dgs_dataset_group([source])
    _write_mdevent(source)
    return mdevent_dataset_group(source)


@pytest.mark.parametrize("kind", ["raw_dgs", "mdevent"])
def test_public_policy_script_replays_complete_choices_and_persists(tmp_path, kind):
    group = _group(tmp_path, kind)
    changes = dict(event_precision_policy="high_precision", symmetry_variance_policy="within_bin_covariance")
    if kind == "raw_dgs":
        changes["monitor_variance_policy"] = "stable"
    set_dgs_reduction_policies(group, **changes)
    set_dgs_trajectory_energy_policy(group, "per_run")
    script = dgs_reduction_policy_script(group, group_variable="imported_group")
    replay = copy.deepcopy(group)
    for key in changes:
        replay.metadata[kind].pop(key)
    replay.metadata[kind].pop("trajectory_energy_policy")
    exec(script, {"imported_group": replay})
    assert {key: replay.metadata[kind][key] for key in changes} == changes
    assert replay.metadata[kind]["trajectory_energy_policy"] == "per_run"
    if kind == "mdevent":
        assert "monitor_variance_policy" not in script
    project = NfitProject([DataGroup("Data", subgroups=[replay])])
    destination = tmp_path / "settings.nfit"
    save_project(project, destination)
    restored = load_project(destination).data_groups[0].subgroups[0]
    assert restored.metadata[kind] == json.loads(json.dumps(replay.metadata[kind]))
    assert dgs_reduction_policy_script(restored, group_variable="imported_group") == script


@pytest.mark.parametrize("kind", ["raw_dgs", "mdevent"])
def test_invalid_policy_update_is_atomic(tmp_path, kind):
    group = _group(tmp_path, kind)
    before = copy.deepcopy(group.metadata)
    with pytest.raises(ValueError, match="symmetry_variance_policy"):
        set_dgs_reduction_policies(group, event_precision_policy="high_precision", symmetry_variance_policy="unknown")
    assert group.metadata == before
    if kind == "mdevent":
        with pytest.raises(ValueError, match="monitor fitting"):
            set_dgs_reduction_policies(group, event_precision_policy="high_precision", monitor_variance_policy="stable")
        assert group.metadata == before


def test_policy_api_rejects_non_dgs_groups_and_unsafe_script_identifiers():
    group = DataGroup("ordinary data")
    with pytest.raises(ValueError, match="raw DGS or MDEvent"):
        set_dgs_reduction_policies(group, event_precision_policy="mantid")
    for name in ("class", "group.attr", "group);print('bad')", ""):
        with pytest.raises(ValueError, match="Python variable name"):
            dgs_reduction_policy_script(group, group_variable=name)


@pytest.mark.parametrize("kind", ["raw_dgs", "mdevent"])
def test_effective_policy_defaults_have_identical_composite_signatures(tmp_path, kind):
    group = _group(tmp_path, kind)
    root = DataGroup("Data", subgroups=[group])
    scope = _composite_scope(root, group)
    explicit = _composite_cache_signature(scope)
    keys = ["event_precision_policy", "symmetry_variance_policy"]
    if kind == "raw_dgs":
        keys.append("monitor_variance_policy")
    for key in keys:
        group.metadata[kind].pop(key)
    assert _composite_cache_signature(scope) == explicit
    for key, choice in (("event_precision_policy", "high_precision"),
                        ("symmetry_variance_policy", "within_bin_covariance")):
        set_dgs_reduction_policies(group, **{key: choice})
        changed = _composite_cache_signature(scope)
        assert changed != explicit
        explicit = changed
    if kind == "raw_dgs":
        set_dgs_reduction_policies(group, monitor_variance_policy="stable")
        assert _composite_cache_signature(scope) != explicit


@pytest.mark.parametrize("kind", ["raw_dgs", "mdevent"])
def test_policy_gui_controls_tooltips_clipboard_and_public_replay(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    group = _group(tmp_path, kind)
    replay = copy.deepcopy(group)
    root = DataGroup("Data", subgroups=[group])
    explorer = NfitProjectExplorer(NfitProject([root]))
    try:
        explorer._refresh_tree(select_dataset_group=group)
        choices = [("event_precision_policy", "mantid", "high_precision"),
                   ("symmetry_variance_policy", "independent_copies", "within_bin_covariance")]
        if kind == "raw_dgs":
            choices.insert(0, ("monitor_variance_policy", "mantid", "stable"))
        else:
            assert explorer.details_widget.findChild(QtWidgets.QComboBox, "mdevent_monitor_variance_policy") is None
        before = _composite_cache_signature(_composite_scope(root, group))
        for key, default, selected in choices:
            selector = explorer.details_widget.findChild(QtWidgets.QComboBox, f"{kind}_{key}")
            assert selector is not None and selector.toolTip()
            assert selector.currentData() == default
            selector.setCurrentIndex(selector.findData(selected))
            assert group.metadata[kind][key] == selected
        assert _composite_cache_signature(_composite_scope(root, group)) != before
        button = explorer.details_widget.findChild(QtWidgets.QPushButton, f"{kind}_copy_policy_script")
        assert button is not None and button.toolTip()
        button.click()
        script = QtWidgets.QApplication.clipboard().text()
        assert script == dgs_reduction_policy_script(group)
        exec(script, {"group": replay})
        assert replay.metadata[kind] == group.metadata[kind]
    finally:
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_histogram_only_policy_reuses_reduced_cache_but_monitor_and_precision_invalidate(tmp_path, monkeypatch):
    group = _group(tmp_path, "raw_dgs")
    first = bin_raw_dgs_group(group, **OPTIONS)
    assert first.metadata["reduced_event_cache"] == {"hits": 0, "misses": 1}
    config = group.metadata["raw_dgs"]
    original_signature = reduction_signature(group.datasets[0], config)
    set_dgs_reduction_policies(group, symmetry_variance_policy="within_bin_covariance")
    assert reduction_signature(group.datasets[0], config) == original_signature
    original = raw_dgs.inspect_raw_dgs_run
    calls = []

    def inspect(path, **kwargs):
        calls.append(kwargs)
        return original(path, **kwargs)

    monkeypatch.setattr(raw_dgs, "inspect_raw_dgs_run", inspect)
    cached = bin_raw_dgs_group(group, **OPTIONS)
    assert cached.metadata["reduced_event_cache"] == {"hits": 1, "misses": 0}
    assert calls == []
    set_dgs_reduction_policies(group, monitor_variance_policy="stable")
    assert reduction_signature(group.datasets[0], config) != original_signature
    calibrated = bin_raw_dgs_group(group, **OPTIONS)
    assert calibrated.metadata["reduced_event_cache"] == {"hits": 0, "misses": 1}
    assert calls == [{"monitor_variance_policy": "stable", "bad_pulse_threshold": 95.0}]
    monitor_signature = reduction_signature(group.datasets[0], config)
    calls.clear()
    set_dgs_reduction_policies(group, event_precision_policy="high_precision")
    assert reduction_signature(group.datasets[0], config) != monitor_signature
    precise = bin_raw_dgs_group(group, **OPTIONS)
    assert precise.metadata["reduced_event_cache"] == {"hits": 0, "misses": 1}
    assert calls == [{"monitor_variance_policy": "stable", "bad_pulse_threshold": 95.0}]
