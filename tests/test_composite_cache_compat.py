from __future__ import annotations

import copy
import json
import zipfile

import pytest

from nfit.project_cache_compat import (
    COMPOSITE_CACHE_SIGNATURE_TAG,
    LEGACY_COMPOSITE_CACHE_SIGNATURE_TAG,
    composite_cache_signatures_match,
)
from nfit.project_io import _binning_signatures_match


def _payload(*, tag=COMPOSITE_CACHE_SIGNATURE_TAG):
    event = {"dimensions": [{"frame": "QSample"} for _ in range(3)]}
    config = {"axes": [
        {"lower": -1.0, "upper": 1.0},
        {"lower": -1.0, "upper": 1.0},
        {"lower": -1.0, "upper": 1.0},
        {"lower": -5.0, "upper": 5.0},
    ]}
    dataset = ["run", ["source"], "single_crystal_inelastic", "mdevent", True,
               1.0, 1.0, {}, [], []]
    return [tag, event, None, {}, json.dumps(config, sort_keys=True), [], [], [],
            [dataset], [], []]


def _pair(payload):
    current = copy.deepcopy(payload)
    saved = copy.deepcopy(payload)
    current[0] = COMPOSITE_CACHE_SIGNATURE_TAG
    saved[0] = LEGACY_COMPOSITE_CACHE_SIGNATURE_TAG
    return json.dumps(saved), json.dumps(current)


def test_safe_legacy_event_signature_is_accepted_by_project_io():
    saved, current = _pair(_payload())
    assert composite_cache_signatures_match(saved, current)
    assert _binning_signatures_match(saved, current)


def test_legacy_event_signature_recurses_through_children_and_group_backgrounds():
    child_saved, child_current = _pair(_payload())
    parent = _payload()
    parent[7] = [["child", child_current]]
    parent[10] = [[None, "background-group", True, 0.5, "linear", "center", None,
                   child_current]]
    saved, current = _pair(parent)
    saved_payload = json.loads(saved)
    saved_payload[7][0][1] = child_saved
    saved_payload[10][0][7] = child_saved
    saved = json.dumps(saved_payload)
    assert composite_cache_signatures_match(saved, current)


@pytest.mark.parametrize("mutation", [
    "automatic_bound", "run_scale", "run_weight", "non_qsample",
    "direct_background", "measured_background", "scientific_change", "malformed",
])
def test_legacy_event_signature_rejects_unsafe_or_unknown_payloads(mutation):
    payload = _payload()
    if mutation == "automatic_bound":
        config = json.loads(payload[4])
        config["axes"][0]["auto_lower"] = True
        payload[4] = json.dumps(config, sort_keys=True)
    elif mutation == "run_scale":
        payload[8][0][6] = 2.0
    elif mutation == "run_weight":
        payload[8][0][5] = 0.5
    elif mutation == "non_qsample":
        payload[1]["dimensions"][0]["frame"] = "QLab"
    elif mutation == "direct_background":
        payload[10] = [["dataset", None, True, 1.0, "linear", "center", None, None]]
    elif mutation == "measured_background":
        _saved, nested = _pair(_payload())
        payload[10] = [[None, "group", True, 1.0, "linear", "measured_events", None,
                        nested]]
    elif mutation == "malformed":
        payload[8][0].pop()
    saved, current = _pair(payload)
    if mutation == "scientific_change":
        changed = json.loads(saved)
        changed[3] = {"a": 1}
        saved = json.dumps(changed)
    assert not composite_cache_signatures_match(saved, current)


def test_unknown_legacy_shape_and_non_event_signature_fail_closed():
    saved, current = _pair(_payload())
    short = json.loads(saved)[:-1]
    assert not composite_cache_signatures_match(json.dumps(short), current)
    non_event = json.loads(saved)
    non_event[1] = None
    assert not composite_cache_signatures_match(json.dumps(non_event), current)
    assert not composite_cache_signatures_match("not json", current)

    unknown = json.loads(saved)
    unknown[0] = "unknown-v0"
    assert not composite_cache_signatures_match(json.dumps(unknown), current)


@pytest.mark.parametrize("bad_bound", [None, float("inf")])
def test_legacy_event_signature_rejects_missing_or_nonfinite_bounds(bad_bound):
    payload = _payload()
    config = json.loads(payload[4])
    if bad_bound is None:
        config["axes"][0].pop("lower")
    else:
        config["axes"][0]["lower"] = bad_bound
    payload[4] = json.dumps(config)
    saved, current = _pair(payload)
    assert not composite_cache_signatures_match(saved, current)


@pytest.mark.parametrize("location", ["child", "background"])
def test_unsafe_nested_signature_invalidates_parent(location):
    nested = _payload()
    nested[8][0][6] = 2.0
    nested_saved, nested_current = _pair(nested)
    parent = _payload()
    if location == "child":
        parent[7] = [["child", nested_current]]
    else:
        parent[10] = [[None, "group", True, 1.0, "linear", "center", None,
                       nested_current]]
    saved, current = _pair(parent)
    saved_payload = json.loads(saved)
    if location == "child":
        saved_payload[7][0][1] = nested_saved
    else:
        saved_payload[10][0][7] = nested_saved
    assert not composite_cache_signatures_match(json.dumps(saved_payload), current)


@pytest.mark.parametrize("field", ["source", "mask", "grid"])
def test_legacy_signature_requires_exact_source_mask_and_grid(field):
    saved, current = _pair(_payload())
    changed = json.loads(saved)
    if field == "source":
        changed[8][0][1] = ["different-source"]
    elif field == "mask":
        changed[8][0][8] = [["different-mask"]]
    else:
        config = json.loads(changed[4])
        config["axes"][0]["upper"] = 2.0
        changed[4] = json.dumps(config, sort_keys=True)
    assert not composite_cache_signatures_match(json.dumps(changed), current)


def test_powder_event_signature_allows_qlab_source_dimensions():
    payload = _payload()
    payload[1]["dimensions"][0]["frame"] = "QLab"
    config = json.loads(payload[4])
    config["coordinate_mode"] = "powder"
    payload[4] = json.dumps(config, sort_keys=True)
    saved, current = _pair(payload)
    assert composite_cache_signatures_match(saved, current)


@pytest.mark.parametrize("named", [False, True])
def test_legacy_composite_archive_restores_lazily_under_current_signature(
    tmp_path, monkeypatch, named
):
    from nfit import project_gui
    from nfit.dataset import PointData4D
    from nfit.pipeline import DataGroup, DatasetEntry
    from nfit.project_io import NfitProject

    data = PointData4D(
        H=[0.0], K=[0.0], L=[0.0], E=[0.0], intensity=[1.0], sigma=[1.0]
    )
    group = DataGroup("events", [DatasetEntry("run", data, kind="mdevent")])
    config = project_gui.data_group_composite_config(group)
    config.update(enabled=True)
    binning_id = (
        project_gui.add_data_group_composite_binning(group, name="Alternate")
        if named
        else config["_binning_id"]
    )
    saved, current = _pair(_payload())
    member = "assets/binnings/composite/data.npz"
    project = NfitProject(
        [group],
        settings={
            project_gui.PROJECT_CACHE_BINNINGS_KEY: True,
            project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY: [{
                "type": "composite",
                "group_index": 0,
                "node_id": None,
                "binning_id": binning_id,
                "format_version": 6,
                "signature": saved,
                "member": member,
            }],
        },
    )
    path = tmp_path / "legacy-cache.nfit"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, b"sentinel artifact must stay lazy")

    monkeypatch.setattr(project_gui, "_composite_cache_signature", lambda *a, **k: current)
    monkeypatch.setattr(
        project_gui,
        "read_project_dataset_artifact",
        lambda *a, **k: pytest.fail("v6 artifact decoded during restore"),
    )
    monkeypatch.setattr(
        project_gui,
        "composite_dataset_data",
        lambda *a, **k: pytest.fail("legacy cache triggered a raw rebin"),
    )
    project_gui._COMPOSITE_DATA_CACHE.clear()
    project_gui._restore_project_binning_cache(project, path)

    key = project_gui._composite_cache_key(group, binning_id if named else None)
    backing = project_gui._COMPOSITE_DATA_CACHE.project_backing(key, current)
    assert backing is not None
    assert backing[0] == path and backing[1] == member
