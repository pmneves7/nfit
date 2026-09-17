"""Persisted live-recipe caches must survive a real process restart."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from nfit import project_data, project_gui
from nfit.analysis.core import AnalysisEntry
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup
from nfit.project_archive import read_project_manifest, write_project_manifest
from nfit.project_dataset_io import save_dataset_file
from nfit.project_io import NfitProject, _binning_signatures_match
from tests.project_gui_test_support import _tiny_mdhisto_data


@pytest.fixture
def saved_recipe(tmp_path, monkeypatch):
    monkeypatch.setenv("NFIT_PERFORMANCE_FILE", str(tmp_path / "performance.json"))
    project_data._VIEWER_VIEW_CACHE.clear()
    project_data._COMPOSITE_DATA_CACHE.clear()
    nodes = []
    for name, value in (("Low", 5.0), ("High", 2.0)):
        source = DatasetEntry(name, _tiny_mdhisto_data(value), kind="mdhisto")
        path = tmp_path / f"{name}.npz"
        save_dataset_file(source, path, use_view=False)
        source.metadata["source_file"] = str(path)
        source.replace_data(source.data, source_backed=True)
        nodes.append(DatasetGroup(name, datasets=[source]))
    group = DataGroup("Experiment", subgroups=nodes)
    for node in nodes:
        config = project_data.data_group_composite_config(
            project_data._composite_scope(group, node)
        )
        config.update(enabled=True, minimum_coverage=0.0)
    analysis = AnalysisEntry(
        "Low minus High", "histogram_arithmetic",
        [f"group-composite:{node.id}" for node in nodes],
        {"operation": "subtract", "right_scale": 1.0},
    )
    derived = project_data.create_derived_analysis_dataset(group, analysis)
    for name in ("Overview", "Detail"):
        project_data.add_dataset_rebin_binning(derived, name=name)
    project = NfitProject([group], settings={"cache_binnings": True})
    path = tmp_path / "derived.nfit"
    project_gui.save_project(project, path)
    expected = tmp_path / "expected.npz"
    result = project_data.dataset_for_slice_viewer(derived)
    np.savez(expected, **{
        field: getattr(result, field)
        for field in ("signal", "errors", "mask", "num_events")
    })
    yield path, derived.id, expected
    project_data._VIEWER_VIEW_CACHE.clear()
    project_data._COMPOSITE_DATA_CACHE.clear()


@pytest.mark.parametrize("legacy", [False, True])
def test_named_derived_caches_reopen_in_a_new_process_without_rebinning(saved_recipe, legacy):
    path, dataset_id, expected = saved_recipe
    manifest = read_project_manifest(path)
    derived_entries = [
        entry for entry in manifest["settings"]["binning_cache_entries"]
        if entry.get("dataset_id") == dataset_id
    ]
    assert len(derived_entries) == 3
    for entry in derived_entries:
        signature = json.loads(entry["signature"])
        assert signature[0] == ["derived_recipe", dataset_id]
        if legacy:
            signature[0] = ["memory", 7, 1234567]
            entry["signature"] = json.dumps(signature)
    if legacy:
        write_project_manifest(path, manifest)
    script = r'''
import sys
from pathlib import Path
import numpy as np
from nfit import project_data as data, project_gui as gui, project_composites as composites
from nfit.project_archive import read_project_manifest

def forbidden(*args, **kwargs):
    raise AssertionError("cache reuse must not load raw data, rebin, or recompress")

data.derived_analysis_dataset_data = forbidden
data._ensure_dataset_data_loaded = forbidden
composites.composite_dataset_data = forbidden
gui.write_dataset_artifact = forbidden
project = gui.load_project(sys.argv[1])
group = project.data_groups[0]
derived = next(item for item in group.iter_datasets() if item.id == sys.argv[2])
assert not gui.project_binnings_need_refresh(project)
with np.load(sys.argv[3]) as expected:
    aliases = gui._entries_with_visualization_binnings(
        group, [derived], force_rebin=True,
    )
    assert len(aliases) == 3
    for alias in aliases:
        result = data.dataset_for_slice_viewer(alias)
        for field in expected.files:
            np.testing.assert_equal(getattr(result, field), expected[field])
    for binning in data.dataset_rebin_binnings(derived):
        config = binning["config"]
        assert gui._saved_binning_compressed_size(
            project, kind="dataset", group=group, target=derived,
            binning_id=binning["id"], config=config,
        ) > 0
        result = data.dataset_for_slice_viewer(
            derived, rebin_config=config,
            cache_id=None if binning["fit"] else binning["id"],
        )
        for field in expected.files:
            np.testing.assert_equal(getattr(result, field), expected[field])
copied = Path(sys.argv[1]).with_name("saved-again.nfit")
gui.save_project(project, copied)
for entry in read_project_manifest(copied)["settings"]["binning_cache_entries"]:
    if entry.get("dataset_id") == derived.id:
        import json
        assert json.loads(entry["signature"])[0] == ["derived_recipe", derived.id]
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), dataset_id, str(expected)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_derived_compatibility_requires_all_scientific_fields_to_match(saved_recipe):
    path, dataset_id, _ = saved_recipe
    manifest = read_project_manifest(path)
    current = next(
        entry["signature"] for entry in manifest["settings"]["binning_cache_entries"]
        if entry.get("dataset_id") == dataset_id
    )
    legacy = json.loads(current)
    legacy[0] = ["memory", 0, 1234567]
    assert _binning_signatures_match(json.dumps(legacy), current)
    # Container semantics, grid, own/inherited masks, backgrounds and source
    # recipes must all retain exact validation; only the old token may differ.
    for index in range(1, 8):
        changed = copy.deepcopy(legacy)
        changed[index] = ["changed", changed[index]]
        assert not _binning_signatures_match(json.dumps(changed), current)
    ordinary = json.loads(current)
    ordinary[0] = ["memory", 0, 7654321]
    assert not _binning_signatures_match(json.dumps(legacy), json.dumps(ordinary))
    assert not _binning_signatures_match("not JSON", current)


@pytest.mark.parametrize("change", ["grid", "operation", "source"])
def test_saved_derived_cache_still_invalidates_when_inputs_change(saved_recipe, change):
    path, dataset_id, _ = saved_recipe
    project = project_gui.load_project(path)
    group = project.data_groups[0]
    derived = next(item for item in group.iter_datasets() if item.id == dataset_id)
    config = project_data.dataset_rebin_config(derived)
    assert not project_gui.project_binnings_need_refresh(project)
    if change == "grid":
        config["axes"][0]["step_size"] *= 2
    elif change == "operation":
        group.analyses[0].parameters["right_scale"] = 0.5
    else:
        group.subgroups[0].datasets[0].scale_factor = 2.0
    assert not project_gui._project_binning_is_current(
        "dataset", group, derived, "fit", config,
    )
