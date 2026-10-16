"""Compatibility and lazy loading of saved fit-history display channels."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from nfit import fit_channels, project_io
from nfit.pipeline import DataGroup, FitTimelineEntry


def _channels():
    return {
        "selected": {
            "kind": "points",
            "fit_channel": "Intensity",
            "fit": np.array([1.123456789, 2.0]),
            "residual": np.array([0.25, -0.5]),
        },
        "other": {
            "kind": "mdhisto",
            "visualization_only": True,
            "fit": np.arange(6.0).reshape(2, 3),
            "residual": np.zeros((2, 3)),
        },
    }


def _no_decode(_payload):
    pytest.fail("fit-history display array decoded before viewer consumption")


def _saved_project():
    entry = FitTimelineEntry("Saved fit", channels=_channels())
    group = DataGroup("Workspace", fits=[entry])
    payload = project_io._project_to_dict(project_io.NfitProject([group]))
    return payload, entry.id


def _fit_by_id(project, identity):
    return next(entry for entry in project.data_groups[0].fits if entry.id == identity)


def test_channel_codec_preserves_schema_and_default_eager_api():
    source = _channels()
    original = source["selected"]["fit"].copy()
    encoded = fit_channels._fit_channels_to_dict(source)
    assert encoded["selected"]["fit_channel"] == "Intensity"
    assert encoded["other"]["visualization_only"] is True
    assert set(encoded["selected"]["fit"]) == {"__ndarray__", "dtype", "shape", "data"}
    assert encoded["selected"]["fit"]["dtype"] == "float32"
    assert encoded["selected"]["fit"]["shape"] == [2]

    restored = fit_channels._fit_channels_from_dict(encoded)
    for name, entry in source.items():
        for channel in fit_channels.FIT_CHANNEL_NAMES:
            assert restored[name][channel].dtype == np.float64
            np.testing.assert_array_equal(restored[name][channel], entry[channel].astype(np.float32))
    assert source["selected"]["fit"].dtype == np.float64
    np.testing.assert_array_equal(source["selected"]["fit"], original)


def test_project_load_and_resave_keep_nested_history_compressed(monkeypatch):
    payload, identity = _saved_project()
    saved_fit = next(item for item in payload["data_groups"][0]["fits"] if item["id"] == identity)
    saved_fit["children"] = [dict(saved_fit, id="nested-fit", children=[])]
    monkeypatch.setattr(fit_channels, "_decode_float_array", _no_decode)

    loaded = project_io._project_from_dict(payload)
    fit = _fit_by_id(loaded, identity)
    assert fit.channels == saved_fit["channels"]
    assert fit.children[0].channels == saved_fit["channels"]
    resaved = project_io._project_to_dict(loaded)
    resaved_fit = next(item for item in resaved["data_groups"][0]["fits"] if item["id"] == identity)
    assert resaved_fit == saved_fit


def test_decoding_one_channel_does_not_materialize_other_channels(monkeypatch):
    encoded = fit_channels._fit_channels_to_dict(_channels())
    monkeypatch.setattr(fit_channels, "_decode_float_array", _no_decode)
    lazy = fit_channels._fit_channels_from_dict(encoded, decode_arrays=False)
    monkeypatch.undo()
    original_decode = fit_channels._decode_float_array
    decoded_payloads = []

    def decode(payload):
        decoded_payloads.append(payload)
        return original_decode(payload)

    monkeypatch.setattr(fit_channels, "_decode_float_array", decode)
    array = fit_channels.fit_channel_array(lazy["selected"]["residual"])
    np.testing.assert_array_equal(array, [0.25, -0.5])
    assert decoded_payloads == [encoded["selected"]["residual"]]
    for name, entry in lazy.items():
        for channel in fit_channels.FIT_CHANNEL_NAMES:
            assert entry[channel] is encoded[name][channel]


@pytest.mark.parametrize("decode_arrays", [True, False])
@pytest.mark.parametrize("container", [list, np.asarray])
def test_legacy_list_and_ndarray_payloads_remain_supported(decode_arrays, container):
    values = container([1.25, 2.5])
    channels = fit_channels._fit_channels_from_dict(
        {1: {"fit": values, "residual": container([0.0, 1.0])}},
        decode_arrays=decode_arrays,
    )
    assert channels["1"]["kind"] == "points"
    np.testing.assert_array_equal(channels["1"]["fit"], values)
    np.testing.assert_array_equal(fit_channels._fit_channel_array(values), values)
    encoded = fit_channels._fit_channels_to_dict(channels)
    np.testing.assert_array_equal(fit_channels._fit_channel_array(encoded["1"]["fit"]), values)


@pytest.mark.parametrize("payload", [None, [], "bad", 1])
def test_malformed_outer_payloads_are_ignored(payload):
    assert fit_channels._fit_channels_from_dict(payload) == {}


@pytest.mark.parametrize("value", [
    None, {}, {"data": "a"}, {"data": "bm90IHpsaWI="},
    ["invalid"], [[1.0], [1.0, 2.0]],
])
def test_malformed_channel_payloads_are_skipped(value):
    assert fit_channels._fit_channel_array(value) is None
    restored = fit_channels._fit_channels_from_dict({
        "invalid-entry": [],
        "selected": {"fit_channel": "Intensity", "fit": value, "residual": [0.0]},
    })
    assert set(restored) == {"selected"}
    assert "fit" not in restored["selected"]
    assert restored["selected"]["fit_channel"] == "Intensity"
    np.testing.assert_array_equal(restored["selected"]["residual"], [0.0])


@pytest.mark.parametrize("updates", [
    {"dtype": "invalid"}, {"shape": [3]}, {"shape": [float("inf")]},
    {"shape": None}, {"data": "not-compressed"},
])
def test_malformed_encoded_arrays_remain_safe_at_consumption(updates):
    encoded = dict(fit_channels._encode_float_array([1.0, 2.0]), **updates)
    lazy = fit_channels._fit_channels_from_dict({"selected": {"fit": encoded}}, decode_arrays=False)
    assert lazy["selected"]["fit"] is encoded
    assert fit_channels._fit_channel_array(lazy["selected"]["fit"]) is None


@pytest.mark.parametrize("storage_mode", [None, "incremental"])
def test_archive_load_and_resave_do_not_decode_history(tmp_path, monkeypatch, storage_mode):
    from nfit.project_gui import load_project, save_project

    payload, identity = _saved_project()
    monkeypatch.setattr(fit_channels, "_decode_float_array", _no_decode)
    source = project_io._project_from_dict(payload)
    path = tmp_path / "history.nfit"
    save_project(source, path, storage_mode=storage_mode)
    loaded = load_project(path)
    channels = _fit_by_id(loaded, identity).channels
    assert channels == _fit_by_id(source, identity).channels
    save_project(loaded, path)
    assert _fit_by_id(load_project(path), identity).channels == channels


def test_clipboard_and_transfer_retain_encoded_history(tmp_path, monkeypatch):
    from nfit.project_clipboard import make_payload, paste_payload
    from nfit.project_transfer import (
        prepare_project_selection,
        read_project_selection,
        save_project_selection,
        write_project_selection,
    )

    payload, identity = _saved_project()
    monkeypatch.setattr(fit_channels, "_decode_float_array", _no_decode)
    source = project_io._project_from_dict(payload)
    entry = _fit_by_id(source, identity)
    copied = make_payload("fit", [entry])
    pasted = paste_payload(copied, target_role="fits", data_group=DataGroup("Target")).items[0]
    assert pasted.channels == entry.channels
    selection = prepare_project_selection(source, "fit", [entry], source_group=source.data_groups[0])
    path = write_project_selection(selection, tmp_path / "selected.nfit")
    loaded = read_project_selection(path)
    assert loaded.payload.items[0].channels == entry.channels
    resaved = save_project_selection(loaded, tmp_path / "resaved.nfit")
    assert read_project_selection(resaved).payload.items[0].channels == entry.channels


def test_gui_consumer_decodes_only_the_selected_dataset(monkeypatch):
    from nfit.dataset import PointListData
    from nfit.project_gui import attach_fit_channels_to_view

    payload, identity = _saved_project()
    project = project_io._project_from_dict(payload)
    fit = _fit_by_id(project, identity)
    view = PointListData(
        columns={"x": np.array([0.0, 1.0]), "y": np.array([2.0, 3.0])},
        coordinate_names=["x"],
        channels=[{"label": "Intensity", "value": "y", "error": None}],
    )
    decode = fit_channels._decode_float_array
    observed = []

    def record(value):
        observed.append(value)
        assert any(value is fit.channels["selected"][name] for name in fit_channels.FIT_CHANNEL_NAMES)
        return decode(value)

    monkeypatch.setattr(fit_channels, "_decode_float_array", record)
    attached = attach_fit_channels_to_view(project.data_groups[0], "selected", view)
    assert len(observed) == 2
    np.testing.assert_array_equal(attached.columns["fit"], _channels()["selected"]["fit"].astype(np.float32))
    np.testing.assert_array_equal(attached.columns["residual"], [0.25, -0.5])
    assert "fit" not in view.columns
    assert isinstance(fit.channels["other"]["fit"], dict)


def test_latest_fit_channels_preserves_runtime_metadata_and_live_arrays():
    from nfit.project_gui import latest_fit_channels

    channels = _channels()
    channels["selected"].update(
        likelihood="poisson", residual_label="Deviance residual", fit_data_scale=2.0,
    )
    group = DataGroup("Workspace", fits=[FitTimelineEntry("Saved fit", channels=channels)])
    latest = latest_fit_channels(group, "selected")
    assert latest["fit"] is channels["selected"]["fit"]
    assert latest["residual"] is channels["selected"]["residual"]
    assert latest["likelihood"] == "poisson"
    assert latest["residual_label"] == "Deviance residual"
    assert latest["fit_data_scale"] == 2.0
    assert latest_fit_channels(group, "missing") is None


def test_fit_channel_facades_reexport_authoritative_helpers():
    from nfit import project_gui

    for name in (
        "FIT_CHANNEL_NAMES", "_encode_float_array", "_fit_channel_array",
        "_fit_channels_from_dict", "_fit_channels_to_dict",
    ):
        assert getattr(project_io, name) is getattr(fit_channels, name)
        assert getattr(project_gui, name) is getattr(fit_channels, name)
    assert project_io._decode_float_array is fit_channels._decode_float_array
    assert fit_channels._fit_channel_array is fit_channels.fit_channel_array
    assert project_io.fit_channel_array is fit_channels.fit_channel_array


def test_fit_channel_owner_has_no_gui_or_facade_imports():
    path = Path(__file__).parents[1] / "src" / "nfit" / "fit_channels.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
            if node.module in (None, "nfit"):
                modules.extend(alias.name for alias in node.names)
    assert not any(
        module.startswith(("PySide", "PyQt"))
        or any(part in {"project_io", "project_gui"} or part.startswith("qt_") for part in module.split("."))
        for module in modules
    )
