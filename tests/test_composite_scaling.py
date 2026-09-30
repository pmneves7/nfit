"""Scalar refresh preserves expensive base binning and saved lazy caches."""

import numpy as np
import pytest

import nfit
from nfit import project_composites as comp
from nfit.composite_scaling import composite_scaling
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import BackgroundSpec, DataGroup, DatasetEntry


def histogram(signal, errors=1):
    return MDHistoData(
        axes=(MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),),
        signal=np.full(2, signal, dtype=float),
        errors=np.full(2, errors, dtype=float),
        mask=np.zeros(2, dtype=bool),
        num_events=np.ones(2),
    )


def sample_group(directory=None):
    background = DatasetEntry("background", histogram(2, 3), kind="mdhisto")
    group = DataGroup(
        "sample", datasets=[DatasetEntry("sample", histogram(10, 2), kind="mdhisto"), background]
    )
    group.backgrounds = [
        BackgroundSpec("b", source_dataset_id=background.id, source_entry=background)
    ]
    config = comp.data_group_composite_config(group)
    config.update(
        enabled=True,
        auto_rebin=False,
        coordinate_mode="hkle",
        axes=[
            dict(
                name="H",
                lower=0.5,
                upper=1.5,
                step_size=1,
                num_bins=2,
                mode="step",
                auto_lower=False,
                auto_upper=False,
            )
        ],
    )
    if directory is not None:
        from nfit import save_dataset_file

        for entry in group.datasets:
            source = directory / f"{entry.name}.npz"
            save_dataset_file(entry, source, use_view=False)
            entry.metadata["source_file"] = str(source)
            entry.replace_data(entry.data, source_backed=True)
    return group


def test_scalar_edits_skip_binning_with_manual_rebin(monkeypatch):
    group = sample_group()
    first = nfit.refresh_composite_dataset(group)
    np.testing.assert_allclose(first.signal, 8)
    monkeypatch.setattr(comp, "composite_dataset_data", lambda *a, **k: pytest.fail("base rebin"))
    nfit.configure_composite_scaling(group, data_scale=2, result_scale=3, fit_weight=7)
    group.backgrounds[0].scale = 0.5
    result = comp._cached_composite_dataset_data(group, force_rebin=False)
    np.testing.assert_allclose(result.signal, 57)
    np.testing.assert_allclose(result.errors, 3 * np.sqrt(16 + 2.25))
    assert comp.composite_dataset_entry(group).fit_weight == 7
    group.backgrounds[0].scale = 0
    np.testing.assert_allclose(nfit.refresh_composite_dataset(group).signal, 60)


def test_saved_base_reopens_lazily_and_rescales_without_binning(tmp_path, monkeypatch):
    group = sample_group(tmp_path)
    nfit.configure_composite_scaling(group, data_scale=2, result_scale=3, fit_weight=4)
    project = nfit.NfitProject([group], settings={"cache_binnings": True})
    path = tmp_path / "sample.nfit"
    nfit.save_project(project, path)
    entries = project.settings["binning_cache_entries"]
    assert len(entries) == 1
    assert entries[0]["stage"] == "unsubtracted"
    comp._COMPOSITE_DATA_CACHE.clear()
    original = np.load
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("eager numerical loading"))
    restored = nfit.load_project(path)
    monkeypatch.setattr(np, "load", original)
    group = restored.data_groups[0]
    monkeypatch.setattr(comp, "composite_dataset_data", lambda *a, **k: pytest.fail("base rebin"))
    group.backgrounds[0].scale = 0.25
    np.testing.assert_allclose(nfit.refresh_composite_dataset(group).signal, 58.5)
    assert comp.composite_dataset_entry(group).fit_weight == 4


def test_scaling_validation_and_export(tmp_path):
    group = sample_group(tmp_path)
    project = nfit.NfitProject([group])
    path = tmp_path / "sample.nfit"
    nfit.save_project(project, path)
    nfit.configure_composite_scaling(group, data_scale=2, result_scale=3, fit_weight=4)
    script = nfit.composite_workflow_script(project, group.name)
    namespace = {"__name__": "export_test"}
    exec(script, namespace)
    np.testing.assert_allclose(namespace["run"]().signal, 54)
    with pytest.raises(ValueError):
        nfit.configure_composite_scaling(group, data_scale=-1)
    assert composite_scaling(group)["data_scale"] == 2


def test_zero_scale_and_fit_weight_do_not_rebin(monkeypatch):
    group = sample_group()
    first = nfit.refresh_composite_dataset(group)
    nfit.configure_composite_scaling(group, fit_weight=0)
    assert nfit.refresh_composite_dataset(group) is first
    monkeypatch.setattr(comp, "composite_dataset_data", lambda *a, **k: pytest.fail("base rebin"))
    nfit.configure_composite_scaling(group, data_scale=0)
    result = nfit.refresh_composite_dataset(group)
    np.testing.assert_allclose(result.signal, -2)
    np.testing.assert_allclose(result.errors, 3)
    nfit.configure_composite_scaling(group, result_scale=0)
    result = nfit.refresh_composite_dataset(group)
    np.testing.assert_allclose(result.signal, 0)
    np.testing.assert_allclose(result.errors, 0)


def test_background_projection_is_reused_for_link_scale(monkeypatch):
    group = sample_group()
    nfit.refresh_composite_dataset(group)
    monkeypatch.setattr(
        comp, "_composite_background_data", lambda *a, **k: pytest.fail("background projection")
    )
    group.backgrounds[0].scale = 2
    np.testing.assert_allclose(nfit.refresh_composite_dataset(group).signal, 6)


def test_existing_group_controls_scale_composite_without_editing_runs(monkeypatch):
    from PySide6 import QtWidgets

    from nfit.pipeline import DatasetGroup
    source = sample_group()
    node = DatasetGroup("sample", datasets=source.datasets, metadata=source.metadata, backgrounds=source.backgrounds)
    root = DataGroup("workspace", subgroups=[node])
    scope = comp._composite_scope(root, node)
    first = nfit.refresh_composite_dataset(root, node=node)
    explorer = nfit.NfitProjectExplorer(nfit.NfitProject([root]))
    item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(item)
    assert item.text(0) == "sample"
    assert explorer.window.findChild(QtWidgets.QDoubleSpinBox, "composite_data_scale") is None
    assert explorer.window.findChild(QtWidgets.QDoubleSpinBox, "composite_result_scale") is None
    assert explorer.window.findChild(QtWidgets.QDoubleSpinBox, "composite_fit_weight") is None
    monkeypatch.setattr(comp, "composite_dataset_data", lambda *a, **k: pytest.fail("event rebin"))
    explorer._set_group_bulk_value("scale_factor", "2")
    np.testing.assert_allclose(nfit.refresh_composite_dataset(root, node=node).signal, first.signal * 2)
    assert all(run.scale_factor == 1 for run in node.datasets)
    explorer._set_group_bulk_value("fit_weight", "7")
    assert comp.composite_dataset_entry(scope).fit_weight == 7
    assert all(run.fit_weight == 1 for run in node.datasets)
    assert explorer.group_scale_edit.text() == "2"
    assert explorer.group_fit_weight_edit.text() == "7"


def test_direct_background_calibration_reuses_projection(monkeypatch):
    group = sample_group()
    nfit.refresh_composite_dataset(group)
    monkeypatch.setattr(comp, "_composite_background_data", lambda *a, **k: pytest.fail("background rebin"))
    group.backgrounds[0].source_entry.scale_factor = 2
    result = nfit.refresh_composite_dataset(group)
    np.testing.assert_allclose(result.signal, 6)
    np.testing.assert_allclose(result.errors, np.sqrt(4 + 36))


def test_zero_background_restores_sample_coverage():
    group = sample_group()
    background = group.backgrounds[0].source_entry
    background.replace_data(background.data.with_updates(mask=np.array([True, False])))
    first = nfit.refresh_composite_dataset(group)
    assert first.mask[0]
    group.backgrounds[0].scale = 0
    result = nfit.refresh_composite_dataset(group)
    assert not result.mask.any()
    np.testing.assert_allclose(result.signal, 10)
    np.testing.assert_allclose(result.errors, 2)


def test_zero_composite_fit_weight_is_visualization_only():
    from nfit.project_gui import fit_dataset_inputs
    group = sample_group()
    nfit.configure_composite_scaling(group, fit_weight=0)
    inputs, _ = fit_dataset_inputs(group, purpose="fit")
    assert inputs == []


def test_refresh_prepares_scaling_base_for_legacy_corrected_cache(monkeypatch):
    group = sample_group()
    first = nfit.refresh_composite_dataset(group)
    comp._COMPOSITE_DATA_CACHE.pop(comp._composite_base_cache_key(group), None)
    calls = []
    original = comp.composite_dataset_data
    def count(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(comp, "composite_dataset_data", count)
    np.testing.assert_allclose(nfit.refresh_composite_dataset(group).signal, first.signal)
    assert len(calls) == 1
    group.backgrounds[0].scale = .25
    np.testing.assert_allclose(nfit.refresh_composite_dataset(group).signal, 9.5)
    assert len(calls) == 1


def test_identical_named_grid_reuses_unsubtracted_histogram(monkeypatch):
    group = sample_group()
    first = nfit.refresh_composite_dataset(group)
    named_id = nfit.add_data_group_composite_binning(group, name="same grid", duplicate_from="fit")
    monkeypatch.setattr(comp, "composite_dataset_data", lambda *a, **k: pytest.fail("duplicate grid rebin"))
    np.testing.assert_allclose(nfit.refresh_composite_dataset(group, binning_id=named_id).signal, first.signal)
    group.backgrounds[0].scale = .25
    np.testing.assert_allclose(nfit.refresh_composite_dataset(group, binning_id=named_id).signal, 9.5)


def test_scalar_progress_identifies_cached_arithmetic(monkeypatch):
    group = sample_group()
    nfit.refresh_composite_dataset(group)
    group.backgrounds[0].scale = .25
    monkeypatch.setattr(comp, "composite_dataset_data", lambda *a, **k: pytest.fail("event rebin"))
    progress = []
    nfit.refresh_composite_dataset(group, progress_callback=progress.append)
    assert any(event["stage"] == "composite_scaling" for event in progress)
    assert all(event["stage"] not in {"raw_dgs_events", "raw_dgs_normalization"} for event in progress)
