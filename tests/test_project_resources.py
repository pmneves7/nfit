from __future__ import annotations

import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from nfit.analysis.artifacts import write_dataset_artifact
from nfit.mapped_archive import array_storage_nbytes
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DataGroup, DatasetEntry, FitTimelineEntry, ModelComponentSpec
from nfit.project_io import NfitProject
from nfit.project_resources import CACHE_EXCLUSIONS_KEY, ProjectResources, excluded_project_cache
from nfit.rebin_cache import CompressedBinning, RebinCache


def histogram(value=3.0):
    return MDHistoData(
        axes=(MDHistoAxis("H", np.arange(5.0), "rlu", "momentum"),),
        signal=np.full(4, value), errors=np.ones(4),
        mask=np.zeros(4, dtype=bool), num_events=np.ones(4),
    )


@pytest.fixture
def resources(monkeypatch):
    from nfit import (
        electronic_structure,
        project_caches,
        project_composites,
        project_data,
        project_resources,
    )
    from nfit.resource_budget import configure_memory_providers

    viewer_cache, composite_cache = RebinCache(), RebinCache()
    monkeypatch.setattr(project_data, "_VIEWER_VIEW_CACHE", viewer_cache)
    monkeypatch.setattr(project_composites, "_COMPOSITE_DATA_CACHE", composite_cache)
    monkeypatch.setattr(project_data, "_PREPARED_POINT_LIST_CACHE", OrderedDict())
    monkeypatch.setattr(project_caches, "MODEL_OVERLAY_CACHE", OrderedDict())
    monkeypatch.setattr(electronic_structure, "_FOURIER_COEFFICIENT_CACHE", OrderedDict())
    monkeypatch.setattr(electronic_structure, "_HAMILTONIAN_COMPONENT_CACHE", OrderedDict())
    monkeypatch.setattr(project_resources, "load_resource_limits", lambda: {"cpu_limit": 2, "ram_limit_mb": 1024})
    monkeypatch.setattr(project_resources, "temporary_storage_usage", lambda: ())
    configure_memory_providers(rss=lambda: 2_000_000, limit=lambda: 100_000_000, available=lambda: 200_000_000)
    dataset = DatasetEntry("Sample", None)
    group = DataGroup("Workspace", datasets=[dataset])
    project = NfitProject([group])
    state = SimpleNamespace(
        project=project, group=group, dataset=dataset,
        viewer_cache=viewer_cache, composite_cache=composite_cache,
        viewer_payloads=[], released=[], changed=[], busy=False,
    )

    def release(payloads):
        state.released.append(payloads)
        state.viewer_payloads.clear()

    state.manager = ProjectResources(
        project, viewer_payloads=lambda: tuple(state.viewer_payloads), release_viewers=release,
        busy=lambda: state.busy, changed=lambda: state.changed.append(True),
    )
    yield state
    viewer_cache.clear()
    composite_cache.clear()
    configure_memory_providers()


def saved_histogram(resources, tmp_path, *, loaded=False):
    payload = histogram()
    artifact = tmp_path / "cube.npz"
    write_dataset_artifact(payload, artifact)
    member = "assets/binnings/sample/data.npz"
    project_path = tmp_path / "project.nfit"
    with ZipFile(project_path, "w") as archive:
        archive.writestr("project.json", "{}")
        archive.write(artifact, member)
    resources.project._project_path = project_path
    if loaded:
        resources.viewer_cache[resources.dataset.id] = ("signature", payload)
    resources.viewer_cache.set_project_backing(
        resources.dataset.id, signature="signature", project_path=project_path,
        member=member, lazy=not loaded,
    )
    return payload, member, artifact.stat().st_size


def row_of(snapshot, *, kind=None, key=None):
    return next(row for row in snapshot.rows if (kind is None or row.kind == kind) and (key is None or row.key == key))


def test_inventory_lists_cold_cache_without_decoding(resources, tmp_path, monkeypatch):
    _payload, _member, compressed_size = saved_histogram(resources, tmp_path)
    monkeypatch.setattr(resources.viewer_cache, "get", lambda *_args: pytest.fail("inventory decoded a cube"))
    monkeypatch.setattr(resources.manager, "_load_source", lambda *_args: pytest.fail("inventory loaded source data"))
    snapshot = resources.manager.snapshot()
    row = row_of(snapshot, kind="Histogram")
    assert row.state == "Saved in project"
    assert row.project_disk_bytes == compressed_size
    assert row.ram_bytes == 0
    assert row.can_load and not row.can_unload and row.can_delete
    assert snapshot.used_bytes == 0
    assert snapshot.process_bytes == 2_000_000
    assert snapshot.limit_bytes == 100_000_000


def test_many_cold_recipes_read_each_cache_catalog_once(resources, monkeypatch):
    calls = []
    original = resources.viewer_cache.resource_records
    monkeypatch.setattr(resources.viewer_cache, "resource_records", lambda: (calls.append(True), original())[1])
    recipes = tuple((resources.viewer_cache, f"{resources.dataset.id}:bin-{index}",
                     f"Sample · Binning {index}", f"assets/binnings/{index}/data.npz",
                     lambda: pytest.fail("inventory computed a histogram"))
                    for index in range(2775))
    resources.manager.recipes = lambda: recipes
    snapshot = resources.manager.snapshot()
    assert len(snapshot.rows) == 2776
    assert len(calls) == 1
    assert all(row.ram_bytes == 0 for row in snapshot.rows)


@pytest.mark.parametrize("missing", [False, True])
def test_unavailable_cached_binning_can_recompute_from_its_recipe(resources, tmp_path, missing):
    payload, member, _size = saved_histogram(resources, tmp_path)
    if missing:
        resources.project._project_path.unlink()
    else:
        resources.project._project_path.write_bytes(b"corrupt archive")
    loaded = []

    def recompute():
        loaded.append(True)
        resources.viewer_cache[resources.dataset.id] = ("new signature", payload)
        return payload

    resources.manager.recipes = lambda: ((resources.viewer_cache, resources.dataset.id,
        "Sample · Default", member, recompute),)
    row = row_of(resources.manager.snapshot(), kind="Histogram")
    assert row.can_load
    resources.manager.load((row.key,))
    assert loaded == [True]
    assert row_of(resources.manager.snapshot(), kind="Histogram").state == "In RAM"


def test_inventory_deduplicates_shared_storage_and_reports_users(resources):
    payload = histogram()
    second = DatasetEntry("Second", None)
    resources.group.datasets.append(second)
    resources.viewer_cache[resources.dataset.id] = ("first", payload)
    resources.viewer_cache[second.id] = ("second", payload.with_updates(metadata={"alias": True}))
    resources.viewer_payloads.extend((
        ("Viewer A", payload.with_updates(metadata={"viewer": "A"})),
        ("Viewer B", payload.with_updates(metadata={"viewer": "B"})),
    ))
    snapshot = resources.manager.snapshot()
    histograms = [row for row in snapshot.rows if row.kind == "Histogram"]
    assert len(histograms) == 2
    assert snapshot.used_bytes == array_storage_nbytes(payload).total
    assert all(row.ram_bytes == snapshot.used_bytes for row in histograms)
    assert all(row.reclaimable_bytes == 0 for row in histograms)
    assert all(row.users == ("Viewer A", "Viewer B") for row in histograms)


def test_compressed_histogram_bytes_are_in_total_and_reclaimable_memory(resources):
    artifact = CompressedBinning.from_data(histogram(), max_bytes=10_000)
    assert artifact is not None
    key = resources.dataset.id
    resources.viewer_cache._compressed[key] = ("signature", artifact)
    resources.viewer_cache._compressed_bytes = artifact.nbytes
    snapshot = resources.manager.snapshot()
    row = row_of(snapshot, kind="Histogram")
    assert row.ram_bytes == artifact.nbytes
    assert snapshot.used_bytes == artifact.nbytes
    assert row.reclaimable_bytes == artifact.nbytes
    resources.manager.unload((row.key,))
    assert resources.manager.snapshot().used_bytes == 0


def test_unload_detaches_users_first_and_preserves_saved_backing(resources, tmp_path, monkeypatch):
    payload, member, _size = saved_histogram(resources, tmp_path, loaded=True)
    resources.viewer_payloads.append(("Viewer", payload.with_updates(metadata={"viewer": True})))
    row = row_of(resources.manager.snapshot(), kind="Histogram")
    assert row.reclaimable_bytes == array_storage_nbytes(payload).total
    original_unload = resources.viewer_cache.unload

    def unload_after_release(key):
        assert resources.released and not resources.viewer_payloads
        return original_unload(key)

    monkeypatch.setattr(resources.viewer_cache, "unload", unload_after_release)
    resources.manager.unload((row.key,))
    assert resources.viewer_cache.peek_resident(resources.dataset.id) is None
    assert resources.viewer_cache.project_backing(resources.dataset.id, "signature")[1] == member
    assert resources.released[0][0] is payload
    cold = row_of(resources.manager.snapshot(), kind="Histogram")
    assert cold.can_load and cold.project_disk_bytes > 0


def test_loading_saved_cache_uses_public_loader_and_reenables_saving(resources, tmp_path):
    expected, member, _size = saved_histogram(resources, tmp_path)
    resources.project.settings[CACHE_EXCLUSIONS_KEY] = [member]
    row = row_of(resources.manager.snapshot(), kind="Histogram")
    resources.manager.load((row.key,))
    actual = resources.viewer_cache.peek_resident(resources.dataset.id)[1]
    np.testing.assert_array_equal(actual.signal, expected.signal)
    assert not excluded_project_cache(resources.project, member)
    assert resources.changed == [True]
    assert row_of(resources.manager.snapshot(), kind="Histogram").can_unload


def test_unsaved_source_and_mixed_selection_are_protected_atomically(resources):
    resources.dataset.replace_data(histogram())
    other = DatasetEntry("Reloadable", histogram(), metadata={"source_file": "/source.dat"})
    resources.group.datasets.append(other)
    snapshot = resources.manager.snapshot()
    unsaved = row_of(snapshot, key=f"source:{resources.dataset.id}")
    assert not unsaved.can_unload and not unsaved.can_delete
    assert "Unsaved" in unsaved.reason
    with pytest.raises(ValueError, match="do not all support"):
        resources.manager.unload((f"source:{other.id}", unsaved.key))
    assert other.data is not None and resources.dataset.data is not None
    assert not resources.released


def test_source_memory_round_trip_uses_real_portable_dataset_loader(resources, tmp_path):
    from nfit.project_dataset_io import save_dataset_file
    payload = histogram()
    resources.dataset.replace_data(payload)
    source = tmp_path / "source.npz"
    save_dataset_file(resources.dataset, source, use_view=False)
    resources.dataset.metadata["source_file"] = str(source)
    resources.dataset.replace_data(payload, source_backed=True)
    key = f"source:{resources.dataset.id}"
    assert row_of(resources.manager.snapshot(), key=key).can_unload
    resources.manager.unload((key,))
    assert resources.dataset.data is None
    assert row_of(resources.manager.snapshot(), key=key).can_load
    resources.manager.load((key,))
    np.testing.assert_array_equal(resources.dataset.data.signal, payload.signal)
    assert resources.dataset.data_matches_source


def test_disappeared_source_protects_its_loaded_copy(resources, tmp_path):
    source = tmp_path / "removed.npz"
    source.write_bytes(b"source")
    resources.dataset.metadata["source_file"] = str(source)
    resources.dataset.replace_data(histogram(), source_backed=True)
    key = f"source:{resources.dataset.id}"
    assert row_of(resources.manager.snapshot(), key=key).can_unload
    source.unlink()
    row = row_of(resources.manager.snapshot(), key=key)
    assert not row.can_unload
    assert "unavailable" in row.reason
    with pytest.raises(ValueError):
        resources.manager.unload((key,))
    assert resources.dataset.data is not None


@pytest.mark.parametrize("action", ["load", "unload", "delete"])
def test_active_operation_protects_resource_actions(resources, tmp_path, action):
    saved_histogram(resources, tmp_path, loaded=action == "unload")
    resources.busy = True
    row = row_of(resources.manager.snapshot(), kind="Histogram")
    assert not row.can_load and not row.can_unload and not row.can_delete
    assert "operation" in row.reason.lower()
    with pytest.raises(RuntimeError, match="active project operation"):
        getattr(resources.manager, action)((row.key,))
    assert not resources.released and not resources.changed


def test_delete_forgets_cache_but_retains_resident_data_and_records_exclusion(resources, tmp_path):
    payload, member, _size = saved_histogram(resources, tmp_path, loaded=True)
    row = row_of(resources.manager.snapshot(), kind="Histogram")
    resources.manager.delete((row.key,))
    assert resources.viewer_cache.peek_resident(resources.dataset.id)[1] is payload
    assert resources.viewer_cache.project_backing(resources.dataset.id, "signature") is None
    assert excluded_project_cache(resources.project, member)
    assert resources.changed == [True]
    assert not row_of(resources.manager.snapshot(), kind="Histogram").can_delete


def test_models_analysis_outputs_and_fit_channels_are_in_inventory(resources):
    from nfit.project_caches import MODEL_OVERLAY_CACHE

    model = ModelComponentSpec("Mode")
    model._nfit_electronic_model_cache = {"evaluated": np.arange(6.0)}
    resources.group.models[model.name] = model
    overlay = {"signal": np.arange(7.0)}
    MODEL_OVERLAY_CACHE[id(resources.group)] = overlay
    analysis = DatasetEntry("Analysis", histogram(), metadata={"derived_from_analysis": "analysis-id"})
    resources.group.datasets.append(analysis)
    fit = FitTimelineEntry("Fit", channels={"Sample": {"fit": np.arange(8.0)}})
    resources.group.fits.append(fit)
    snapshot = resources.manager.snapshot()
    assert row_of(snapshot, kind="Analysis output").ram_bytes == array_storage_nbytes(analysis.data).total
    assert row_of(snapshot, kind="Evaluated model").ram_bytes == 6 * 8
    assert row_of(snapshot, kind="Model overlays").ram_bytes == 7 * 8
    fit_row = row_of(snapshot, kind="Fit results")
    assert fit_row.ram_bytes == 8 * 8 and not fit_row.can_unload and not fit_row.can_delete
    expected = array_storage_nbytes((analysis.data, model._nfit_electronic_model_cache, overlay, resources.group.fits)).total
    assert snapshot.used_bytes == expected


def test_unload_model_cache_releases_viewers_using_model_payload(resources):
    model = ModelComponentSpec("Mode")
    model._nfit_electronic_model_cache = {"evaluated": np.arange(6.0)}
    resources.group.models[model.name] = model
    resources.viewer_payloads.append(("Model viewer", model._nfit_electronic_model_cache))
    row = row_of(resources.manager.snapshot(), kind="Evaluated model")
    resources.manager.unload((row.key,))
    assert resources.released and resources.released[0]
    assert not resources.viewer_payloads
    assert not hasattr(model, "_nfit_electronic_model_cache")


def test_shared_model_intermediates_are_counted_once_and_explicitly_unloadable(resources):
    from nfit import electronic_structure

    payload = np.arange(5.0)
    electronic_structure._FOURIER_COEFFICIENT_CACHE["coefficients"] = payload
    electronic_structure._HAMILTONIAN_COMPONENT_CACHE["components"] = payload
    snapshot = resources.manager.snapshot()
    rows = [row for row in snapshot.rows if row.kind == "Shared model cache"]
    assert len(rows) == 2
    assert snapshot.used_bytes == payload.nbytes
    assert all(row.ram_bytes == payload.nbytes and row.reclaimable_bytes == 0 for row in rows)
    resources.manager.unload((rows[0].key,))
    assert not electronic_structure._FOURIER_COEFFICIENT_CACHE
    assert electronic_structure._HAMILTONIAN_COMPONENT_CACHE
    assert resources.manager.snapshot().used_bytes == payload.nbytes


def test_analysis_records_include_numerical_diagnostics_without_deleting_results(resources):
    from nfit.analysis.core import AnalysisEntry

    analysis = AnalysisEntry("Analysis record", "integrate", [], {}, metadata={"diagnostic": np.arange(3.0)})
    resources.group.analyses.append(analysis)
    row = row_of(resources.manager.snapshot(), kind="Analysis results")
    assert row.ram_bytes == 3 * 8
    assert not row.can_delete and not row.can_unload


def test_reduced_events_are_disk_only_and_deletable_without_loading(resources, tmp_path):
    from nfit.raw_dgs_cache import _ReducedEventCache

    path = tmp_path / "events.npz"
    path.write_bytes(b"not decoded by the resource inventory")
    resources.dataset._raw_dgs_reduction_cache = _ReducedEventCache("signature", path)
    row = row_of(resources.manager.snapshot(), kind="Reduced events")
    assert row.temporary_disk_bytes == path.stat().st_size
    assert row.ram_bytes == 0 and not row.can_load and not row.can_unload
    assert row.can_delete
    resources.manager.delete((row.key,))
    assert resources.dataset._raw_dgs_reduction_cache is None


def test_recipe_load_is_explicit_and_removes_cache_exclusion(resources):
    member = "assets/binnings/recompute/data.npz"
    called = []
    resources.project.settings[CACHE_EXCLUSIONS_KEY] = [member]
    resources.manager.recipes = lambda: (
        (resources.viewer_cache, resources.dataset.id, "Sample · Overview", member, lambda: called.append(True)),
    )
    row = row_of(resources.manager.snapshot(), kind="Histogram")
    assert not called and row.state == "Needs recompute"
    resources.manager.load((row.key,))
    assert called == [True]
    assert not excluded_project_cache(resources.project, member)


def test_deleted_binning_is_not_recreated_on_next_save(resources, tmp_path, monkeypatch):
    from nfit import project_gui
    from nfit.project_archive import read_project_manifest

    monkeypatch.setattr(project_gui, "_VIEWER_VIEW_CACHE", resources.viewer_cache)
    monkeypatch.setattr(project_gui, "_COMPOSITE_DATA_CACHE", resources.composite_cache)
    payload = histogram()
    source = tmp_path / "sample.npz"
    write_dataset_artifact(payload, source)
    resources.dataset.metadata["source_file"] = str(source)
    resources.dataset.kind = "mdhisto"
    resources.dataset.replace_data(payload, source_backed=True)
    config = project_gui.dataset_rebin_config(resources.dataset)
    config.update(enabled=True, minimum_coverage=0.0)
    resources.project.settings[project_gui.PROJECT_CACHE_BINNINGS_KEY] = True
    path = tmp_path / "saved-project.nfit"
    project_gui.save_project(resources.project, path)
    row = row_of(resources.manager.snapshot(), kind="Histogram")
    member = resources.project.settings[project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY][0]["member"]
    resources.manager.delete((row.key,))
    assert excluded_project_cache(resources.project, member)
    monkeypatch.setattr(project_gui, "write_dataset_artifact", lambda *_args, **_kwargs: pytest.fail("deleted cache was recreated"))
    project_gui.save_project(resources.project, path)
    with ZipFile(path) as archive:
        assert member not in archive.namelist()
    assert not read_project_manifest(path)["settings"].get(project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY)
    assert resources.dataset.data is not None
    assert config["enabled"]


def test_resource_service_has_no_qt_or_gui_coordinator_imports():
    from nfit import project_resources

    source = Path(project_resources.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or "", *(alias.name for alias in node.names)]
        else:
            continue
        assert not any(module.startswith(("PySide", "PyQt")) or module in {"project_gui", "qt_resource_manager"} for module in modules)
