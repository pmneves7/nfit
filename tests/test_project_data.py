# ruff: noqa: F401, F403, F405
import copy

import nfit.project_data as project_data
from nfit.analysis.core import AnalysisEntry
from nfit.mdhisto import MDHistoChannel
from nfit.project_archive import read_project_manifest
from tests.project_gui_test_support import *
from tests.project_gui_test_support import (
    _explorer_with_fit_result,
    _grid_mdhisto_data,
    _he_mdhisto_data,
    _points_for_dynamic_writeback,
    _rpa_overlay_group,
    _standard_shortcut_text,
    _tiny_mdhisto_data,
    _tree_items_with_children,
)


def test_saved_hierarchical_composite_grid_does_not_materialize_reference(
    monkeypatch,
):
    from nfit import project_composites

    leaf = DatasetGroup(
        "Leaf",
        datasets=[DatasetEntry("scan", _tiny_mdhisto_data(1.0), kind="mdhisto")],
    )
    parent = DatasetGroup("Parent", subgroups=[leaf])
    root = DataGroup("Workspace1", subgroups=[parent])
    leaf_config = project_gui.data_group_composite_config(
        project_gui._composite_scope(root, leaf)
    )
    leaf_config["enabled"] = True
    parent.metadata[project_gui.GROUP_COMPOSITE_KEY] = {
        "enabled": True,
        "axes": copy.deepcopy(leaf_config["axes"]),
    }
    monkeypatch.setattr(
        project_composites,
        "_composite_reference_data",
        lambda _scope: pytest.fail("saved axes must not materialize a reference"),
    )

    config = project_gui.data_group_composite_config(
        project_gui._composite_scope(root, parent)
    )

    assert config["axes"] == parent.metadata[project_gui.GROUP_COMPOSITE_KEY]["axes"]

    explorer = NfitProjectExplorer(NfitProject([root]))
    monkeypatch.setattr(
        project_gui,
        "_ensure_dataset_data_loaded",
        lambda _dataset: pytest.fail("selection must not load source data"),
    )
    explorer._set_dataset_collection_details(root, parent)


def test_corelli_composite_migrates_to_fractional_momentum_and_discrete_energy():
    from PySide6 import QtWidgets

    raw_dgs = {
        "format": "corelli-correlation-nexus",
        "dimensions": [
            {"lower": -5.0, "upper": 5.0},
            {"lower": -5.0, "upper": 5.0},
            {"lower": -5.0, "upper": 5.0},
            {"lower": -30.0, "upper": 30.0},
        ],
    }
    batch = DatasetGroup("CORELLI", metadata={"raw_dgs": raw_dgs})
    batch.metadata[project_gui.GROUP_COMPOSITE_KEY] = {
        "auto_rebin": False,
        "stale": False,
        "axes": [
            {
                "name": name,
                "mode": "bins",
                "fractional": False,
                "lower": -5.0,
                "upper": 5.0,
                "num_bins": 11,
                "step_size": 1.0,
            }
            for name in ("H", "K", "L", "DeltaE")
        ],
    }
    root = DataGroup("Workspace1", subgroups=[batch])

    config = project_gui.data_group_composite_config(
        project_gui._composite_scope(root, batch)
    )

    assert [axis["fractional"] for axis in config["axes"]] == [
        True,
        True,
        True,
        False,
    ]
    assert config["corelli_assignment_version"] == 1
    assert config["stale"] is True

    explorer = NfitProjectExplorer(NfitProject([root]))
    explorer._set_dataset_collection_details(root, batch)
    energy_assignment = explorer.details_widget.findChild(
        QtWidgets.QComboBox,
        "group_composite_axis_assignment_3",
    )
    assert energy_assignment is not None
    assert not energy_assignment.isEnabled()
    assert "separate" in energy_assignment.toolTip()


def test_live_derived_dataset_keeps_owner_in_viewer_binning_aliases():
    source = DatasetEntry("source", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Workspace1", datasets=[source])
    analysis = AnalysisEntry("comparison", "dataset_clone", [source.id], {})
    derived = project_gui.create_derived_analysis_dataset(group, analysis)

    datasets, names = project_gui.slice_viewer_datasets(
        group,
        use_composite=False,
    )

    assert derived.name in names
    assert len(datasets) == 2


def test_composite_preserves_nonidentity_histogram_coordinates():
    basis = [[1, 1, 0, 0], [0, 0, 1, 0], [1, -1, 0, 0], [0, 0, 0, 1]]
    axes = tuple(
        MDHistoAxis(name, edges, "meV" if i == 3 else "rlu", "energy" if i == 3 else "momentum")
        for i, (name, edges) in enumerate(zip(
            ["parallel", "vertical", "transverse", "E"],
            [[0.5, 1.5, 2.5], [1.5, 2.5, 3.5], [-0.1, 0.1], [2.75, 3.25]],
            strict=True,
        ))
    )
    signal = np.arange(1., 5.).reshape(2, 2, 1, 1)
    data = MDHistoData(axes, signal, np.ones_like(signal), np.zeros_like(signal, dtype=bool), np.ones_like(signal), metadata={"rebin": {"vectors": basis}})
    group = DataGroup("HHL", datasets=[DatasetEntry("source", data, kind="mdhisto")])
    config = dict(enabled=True, mean_weighting="uniform", minimum_coverage=0., axes=[
        dict(name=axis.name, vector=vector, lower=float(axis.centers[0]), upper=float(axis.centers[-1]), step_size=float(np.diff(axis.values)[0]), mode="step", fractional=False)
        for axis, vector in zip(axes, basis, strict=True)
    ])
    result = project_gui._composite_mdhisto_data(group, config)
    np.testing.assert_allclose(result.signal, signal)
    np.testing.assert_allclose(result.errors, data.errors)
    np.testing.assert_allclose(result.auxiliary_channels["coverage_fraction"].values, 1.)


def test_save_reuses_binnings_evicted_from_memory(tmp_path, monkeypatch):
    project_gui._COMPOSITE_DATA_CACHE.clear()
    monkeypatch.setattr(project_gui, "_COMPOSITE_DATA_CACHE_MAX_BYTES", 100_000)
    # The service owns the budget; the GUI alias is retained for compatibility.
    from nfit import project_composites
    monkeypatch.setattr(project_composites, "_COMPOSITE_DATA_CACHE_MAX_BYTES", 100_000)
    monkeypatch.setattr(project_composites, "_COMPOSITE_DATA_CACHE_LIMIT", 0)
    subgroups = []
    for i in range(3):
        source = tmp_path / f"source-{i}.nxs"
        source.write_bytes(b"placeholder")
        node = DatasetGroup(str(i), datasets=[DatasetEntry(str(i), _tiny_mdhisto_data(i + 1.), kind="mdhisto", metadata={"source_file": str(source)})])
        config = project_gui.data_group_composite_config(project_gui._composite_scope(DataGroup("temporary"), node))
        config.update(enabled=True, minimum_coverage=0.)
        subgroups.append(node)
    root = DataGroup("root", subgroups=subgroups)
    project = NfitProject([root], settings={"cache_binnings": True})
    expected, _ = project_gui.slice_viewer_datasets(root)
    assert not project_gui._COMPOSITE_DATA_CACHE
    assert len(project_gui._COMPOSITE_DATA_CACHE._compressed) == 3
    with monkeypatch.context() as checks:
        checks.setattr("nfit.rebin_cache.CompressedBinning.restore", lambda *a: pytest.fail("freshness checks must not read arrays"))
        assert not project_gui.project_binnings_need_refresh(project)
    monkeypatch.setattr(project_composites, "composite_dataset_data", lambda *a, **k: pytest.fail("saving current binnings must not rebin"))
    path = tmp_path / "overflow.nfit"
    save_project(project, path)
    assert len(read_project_manifest(path)["settings"][project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY]) == 3
    restored = load_project(path)
    actual, _ = project_gui.slice_viewer_datasets(restored.data_groups[0])
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_allclose(left.signal, right.signal, equal_nan=True)
        np.testing.assert_allclose(left.errors, right.errors, equal_nan=True)
        np.testing.assert_array_equal(left.mask, right.mask)
        np.testing.assert_array_equal(left.num_events, right.num_events)
    restored.data_groups[0].subgroups[0].metadata[project_gui.GROUP_COMPOSITE_KEY]["minimum_coverage"] = 0.5
    assert project_gui.project_binnings_need_refresh(restored)
    project_gui._COMPOSITE_DATA_CACHE.clear()


def test_tree_cache_badges_follow_current_dataset_and_composite_bins():
    from nfit import project_composites

    project_gui._VIEWER_VIEW_CACHE.clear()
    project_gui._COMPOSITE_DATA_CACHE.clear()

    direct = DatasetEntry("direct", _tiny_mdhisto_data(1.0), kind="mdhisto")
    direct_config = project_data.dataset_rebin_config(direct)
    direct_config.update(enabled=True, minimum_coverage=0.0)
    nested = DatasetGroup(
        "nested",
        datasets=[DatasetEntry("run", _tiny_mdhisto_data(2.0), kind="mdhisto")],
    )
    root = DataGroup("Workspace1", datasets=[direct], subgroups=[nested])
    scope = project_gui._composite_scope(root, nested)
    composite_config = project_gui.data_group_composite_config(scope)
    composite_config.update(enabled=True, minimum_coverage=0.0)

    assert project_data.dataset_for_slice_viewer(direct, force_rebin=True) is not None
    assert project_composites._cached_composite_dataset_data(
        scope, force_rebin=True
    ) is not None
    explorer = NfitProjectExplorer(NfitProject([root]))
    datasets_item = explorer.tree.topLevelItem(0).child(0)
    dataset_item = datasets_item.child(0)
    subgroup_item = datasets_item.child(1)

    assert project_gui._dataset_cached_binnings_current(root, direct)
    assert project_gui._composite_cached_binnings_current(root, nested)
    assert dataset_item.icon(0).cacheKey() == project_gui._tree_item_icon(
        "dataset", cached=True
    ).cacheKey()
    assert subgroup_item.icon(0).cacheKey() == project_gui._tree_item_icon(
        "folder", cached=True
    ).cacheKey()
    assert "cached and up to date" in dataset_item.toolTip(0)
    assert "cached and up to date" in subgroup_item.toolTip(0)

    explorer.tree.setCurrentItem(dataset_item)
    direct_config["auto_rebin"] = False
    direct_config["minimum_coverage"] = 0.5
    explorer._after_dataset_rebin_changed(direct, root)
    composite_config["minimum_coverage"] = 0.5
    explorer._refresh_cache_badges()
    assert dataset_item.icon(0).cacheKey() == project_gui._tree_item_icon(
        "dataset"
    ).cacheKey()
    assert subgroup_item.icon(0).cacheKey() == project_gui._tree_item_icon(
        "folder"
    ).cacheKey()
    assert not dataset_item.toolTip(0)
    assert not subgroup_item.toolTip(0)

    project_gui._VIEWER_VIEW_CACHE.clear()
    project_gui._COMPOSITE_DATA_CACHE.clear()


def test_open_composite_viewer_refreshes_every_cache_badge(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    project_gui._VIEWER_VIEW_CACHE.clear()
    project_gui._COMPOSITE_DATA_CACHE.clear()

    subgroups = []
    for index in range(2):
        subgroup = DatasetGroup(
            f"background {index + 1}",
            datasets=[
                DatasetEntry(
                    f"run {index + 1}",
                    _tiny_mdhisto_data(index + 1.0),
                    kind="mdhisto",
                )
            ],
            enabled=False,
        )
        subgroups.append(subgroup)
    root = DataGroup("Powder background averages", subgroups=subgroups)
    for subgroup in subgroups:
        config = project_gui.data_group_composite_config(
            project_gui._composite_scope(root, subgroup)
        )
        config.update(enabled=True, minimum_coverage=0.0)

    explorer = NfitProjectExplorer(NfitProject([root]))
    datasets_item = explorer.tree.topLevelItem(0).child(0)
    subgroup_items = [datasets_item.child(index) for index in range(2)]
    plain_icon = project_gui._tree_item_icon("folder").cacheKey()
    cached_icon = project_gui._tree_item_icon("folder", cached=True).cacheKey()
    assert [item.icon(0).cacheKey() for item in subgroup_items] == [
        plain_icon,
        plain_icon,
    ]

    viewer = explorer.open_slice_viewer(root, use_composite=True)

    assert viewer is not None
    assert all(
        project_gui._composite_cached_binnings_current(root, subgroup)
        for subgroup in subgroups
    )
    assert [item.icon(0).cacheKey() for item in subgroup_items] == [
        cached_icon,
        cached_icon,
    ]
    assert all("cached and up to date" in item.toolTip(0) for item in subgroup_items)

    viewer.window.close()
    explorer.window.close()
    project_gui._VIEWER_VIEW_CACHE.clear()
    project_gui._COMPOSITE_DATA_CACHE.clear()


def test_hierarchical_rebin_retains_matching_child_caches(monkeypatch):
    from nfit import project_composites
    project_gui._COMPOSITE_DATA_CACHE.clear()
    child = DatasetGroup("child", datasets=[DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")])
    parent = DatasetGroup("parent", subgroups=[child])
    root = DataGroup("root", subgroups=[parent])
    child_scope = project_gui._composite_scope(root, child)
    config = project_gui.data_group_composite_config(child_scope)
    config.update(enabled=True, minimum_coverage=0., mean_weighting="uniform")
    parent.metadata[project_gui.GROUP_COMPOSITE_KEY] = copy.deepcopy(config)
    project_gui._cached_composite_dataset_data(project_gui._composite_scope(root, parent))
    assert project_gui._peek_cached_composite_dataset_data(child_scope) is not None
    monkeypatch.setattr(project_composites, "composite_dataset_data", lambda *a, **k: pytest.fail("child already computed"))
    assert project_gui.prepare_project_binning_cache(NfitProject([root])) == 0


def test_composite_progress_reports_dataset_count_and_global_point_work():
    events = []
    report = project_gui._composite_progress_callback(events.append, 6)

    report({"stage": "rebin", "iteration": 75, "total": 100})

    assert events == [
        {
            "stage": "rebin",
            "iteration": 75,
            "total": 100,
            "datasets_total": 6,
            "datasets_completed": 0,
            "message": "rebinning 6 datasets: 75/100 point contributions",
        }
    ]


def test_composite_progress_counts_prepared_source_datasets():
    events = []
    group = DataGroup(
        "Datagroup1",
        datasets=[
            DatasetEntry("first", _tiny_mdhisto_data(1.0), kind="mdhisto"),
            DatasetEntry("second", _tiny_mdhisto_data(2.0), kind="mdhisto"),
        ],
    )
    config = project_gui.data_group_composite_config(group)
    config.update(enabled=True, minimum_coverage=0.0)

    project_gui.composite_dataset_data(group, progress_callback=events.append)

    source_events = [
        event
        for event in events
        if event.get("stage") == "rebin_sources"
        and event.get("datasets_completed") is not None
    ]
    assert [event["datasets_completed"] for event in source_events] == [0, 1, 2]
    assert all(event["datasets_total"] == 2 for event in source_events)


def test_viewer_batch_progress_and_cache_cover_more_than_four_composites():
    project_gui._COMPOSITE_DATA_CACHE.clear()
    subgroups = []
    for index in range(6):
        subgroup = DatasetGroup(
            f"Group {index + 1}",
            datasets=[
                DatasetEntry(
                    f"scan {index + 1}",
                    _tiny_mdhisto_data(float(index + 1)),
                    kind="mdhisto",
                )
            ],
        )
        config = project_gui.data_group_composite_config(
            project_gui._composite_scope(DataGroup("temporary"), subgroup)
        )
        config.update(enabled=True, minimum_coverage=0.0)
        subgroups.append(subgroup)
    group = DataGroup("Workspace1", subgroups=subgroups)

    first_events = []
    _datasets, names = project_gui.slice_viewer_datasets(
        group,
        progress_callback=first_events.append,
    )
    assert len(names) == 6
    batch_events = [
        event for event in first_events if event.get("stage") == "rebin_batch"
    ]
    assert [event["batch_completed"] for event in batch_events] == [
        0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6
    ]
    assert [event["batch_name"] for event in batch_events[::2]] == [
        f"Group {index}" for index in range(1, 7)
    ]
    assert all(event["batch_total"] == 6 for event in batch_events)
    assert len(project_gui._COMPOSITE_DATA_CACHE) == 6

    second_events = []
    project_gui.slice_viewer_datasets(group, progress_callback=second_events.append)
    assert not any(
        event.get("stage") == "rebin_sources" for event in second_events
    )
    assert len(
        [event for event in second_events if event.get("stage") == "rebin_batch"]
    ) == 12


def test_project_can_embed_and_restore_current_composite_binning(tmp_path, monkeypatch):
    import zipfile

    from nfit.project_rebin_panels import _format_bytes as format_rebin_bytes

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    project_gui._COMPOSITE_DATA_CACHE.clear()
    source = tmp_path / "large-source.nxs"
    source.write_bytes(b"source placeholder")
    dataset = DatasetEntry(
        "scan",
        _tiny_mdhisto_data(4.0),
        kind="mdhisto",
        metadata={"source_file": str(source)},
    )
    group = DataGroup("Workspace1", datasets=[dataset])
    dataset_config = project_gui.dataset_rebin_config(dataset)
    dataset_config.update(enabled=True, minimum_coverage=0.0)
    config = project_gui.data_group_composite_config(group)
    config.update(enabled=True, minimum_coverage=0.0)
    project = NfitProject(
        [group],
        settings={project_gui.PROJECT_CACHE_BINNINGS_KEY: True},
    )
    cached_dataset = project_gui.dataset_for_slice_viewer(dataset)
    cached, _names = project_gui.slice_viewer_datasets(group)
    path = tmp_path / "cached.nfit"

    assert not project_gui.project_binnings_need_refresh(project)
    save_events = []
    save_project(project, path, progress_callback=save_events.append)
    assert save_events == []

    dataset_disk_size = project_gui._saved_binning_compressed_size(
        project,
        kind="dataset",
        group=group,
        target=dataset,
        binning_id=dataset_config["_binning_id"],
        config=dataset_config,
    )
    composite_disk_size = project_gui._saved_binning_compressed_size(
        project,
        kind="dataset group",
        group=group,
        target=group,
        binning_id=config["_binning_id"],
        config=config,
    )
    with zipfile.ZipFile(path) as archive:
        saved_sizes = {
            entry["type"]: archive.getinfo(entry["member"]).compress_size
            for entry in project.settings[
                project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY
            ]
        }
    assert dataset_disk_size == saved_sizes["dataset"]
    assert composite_disk_size == saved_sizes["composite"]

    explorer = NfitProjectExplorer(project)
    explorer._set_dataset_details(dataset, group)
    dataset_size_label = explorer.details_widget.findChild(
        QtWidgets.QLabel,
        "dataset_rebin_memory_estimate",
    )
    assert (
        f"Compressed disk size: {format_rebin_bytes(dataset_disk_size)}"
        in dataset_size_label.text()
    )
    explorer._set_dataset_collection_details(group, group)
    composite_size_label = explorer.details_widget.findChild(
        QtWidgets.QLabel,
        "group_composite_memory_estimate",
    )
    assert (
        f"Compressed disk size: {format_rebin_bytes(composite_disk_size)}"
        in composite_size_label.text()
    )
    explorer.window.close()

    config["minimum_coverage"] = 0.25
    assert (
        project_gui._saved_binning_compressed_size(
            project,
            kind="dataset group",
            group=group,
            target=group,
            binning_id=config["_binning_id"],
            config=config,
        )
        is None
    )
    assert project_gui.project_binnings_need_refresh(project)
    refresh_events = []
    save_project(project, path, progress_callback=refresh_events.append)
    assert any(
        event.get("stage") == "rebin_batch" for event in refresh_events
    )
    assert not project_gui.project_binnings_need_refresh(project)

    manifest = read_project_manifest(path)
    entries = manifest["settings"][project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY]
    assert len(entries) == 2
    assert {
        entry["format_version"] for entry in entries
    } == {project_gui.PROJECT_BINNING_CACHE_FORMAT_VERSION}
    assert all(isinstance(entry.get("signature"), str) for entry in entries)
    with zipfile.ZipFile(path) as archive:
        assert entries[0]["member"] in archive.namelist()

    project_gui._COMPOSITE_DATA_CACHE.clear()
    restored = load_project(path)
    restored_group = restored.data_groups[0]
    restored_dataset = next(restored_group.iter_datasets())

    def ensure_without_raw_source(entry):
        if entry is restored_dataset:
            pytest.fail("restored binning should avoid the raw source")
        return entry.data

    monkeypatch.setattr(
        project_gui,
        "_ensure_dataset_data_loaded",
        ensure_without_raw_source,
    )
    restored_dataset_view = project_gui.dataset_for_slice_viewer(restored_dataset)
    np.testing.assert_allclose(
        restored_dataset_view.signal,
        cached_dataset.signal,
        equal_nan=True,
    )
    views, names = project_gui.slice_viewer_datasets(restored_group)
    assert names == ["Workspace1 Composite"]
    np.testing.assert_allclose(views[0].signal, cached[0].signal, equal_nan=True)

    restored.settings[project_gui.PROJECT_CACHE_BINNINGS_KEY] = False
    save_project(restored, path)
    manifest = read_project_manifest(path)
    assert project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY not in manifest["settings"]
    with zipfile.ZipFile(path) as archive:
        assert not any(name.startswith("assets/binnings/") for name in archive.namelist())


def test_project_save_embeds_session_disk_cache_and_removes_spill(tmp_path):
    source = tmp_path / "source.nxs"
    source.write_bytes(b"source placeholder")
    dataset = DatasetEntry(
        "scan",
        _tiny_mdhisto_data(4.0),
        kind="mdhisto",
        metadata={"source_file": str(source)},
    )
    dataset.replace_data(dataset.data, source_backed=True)
    group = DataGroup("Workspace1", datasets=[dataset])
    config = project_gui.dataset_rebin_config(dataset)
    config.update(enabled=True, minimum_coverage=0.0)
    project = NfitProject(
        [group], settings={project_gui.PROJECT_CACHE_BINNINGS_KEY: True}
    )
    cache = project_gui._VIEWER_VIEW_CACHE
    cache.clear()
    project_gui.dataset_for_slice_viewer(dataset)
    spill = tmp_path / "session-spill.npz"

    def spill_to_disk(_label, artifact):
        artifact.write_npz(spill)
        return spill

    cache.before_discard = spill_to_disk
    cache._compress_resident(dataset.id, max_bytes=100_000)
    cache._discard_compressed(dataset.id)
    assert spill.exists()
    assert project_gui._project_binning_is_current(
        "dataset", group, dataset, config["_binning_id"], config
    )

    path = tmp_path / "disk-backed.nfit"
    try:
        save_project(project, path)
        assert not spill.exists()
        assert project_gui._project_binning_is_current(
            "dataset", group, dataset, config["_binning_id"], config
        )
        restored = project_gui._peek_cached_dataset_view(dataset)
        np.testing.assert_allclose(restored.signal, 4.0)
    finally:
        cache.before_discard = None
        cache.clear()


def test_persisted_binning_restore_primes_dependencies_before_caching(tmp_path, monkeypatch):
    source = tmp_path / "source.npz"
    source.write_bytes(b"source placeholder")
    dataset = DatasetEntry(
        "scan",
        _tiny_mdhisto_data(2.0),
        kind="mdhisto",
        metadata={"source_file": str(source)},
    )
    dataset.replace_data(dataset.data, source_backed=True)
    group = DataGroup("Workspace1", datasets=[dataset])
    project_gui.dataset_rebin_config(dataset).update(enabled=True, minimum_coverage=0.0)
    project_gui.data_group_composite_config(group).update(
        enabled=True,
        minimum_coverage=0.0,
    )
    project = NfitProject(
        [group],
        settings={project_gui.PROJECT_CACHE_BINNINGS_KEY: True},
    )
    path = tmp_path / "cached.nfit"
    save_project(project, path)

    restored = project_gui._project_from_dict(read_project_manifest(path))
    restored_dataset = next(restored.data_groups[0].iter_datasets())
    original_signature = project_gui._composite_cache_signature
    mutated = False

    def mutating_signature(scope):
        nonlocal mutated
        if not mutated:
            restored_dataset.replace_data(restored_dataset.data, source_backed=True)
            mutated = True
        return original_signature(scope)

    monkeypatch.setattr(project_gui, "_composite_cache_signature", mutating_signature)
    project_gui._VIEWER_VIEW_CACHE.clear()
    project_gui._COMPOSITE_DATA_CACHE.clear()
    project_gui._restore_project_binning_cache(restored, path)

    assert not project_gui.project_binnings_need_refresh(restored)

    legacy = project_gui._project_from_dict(read_project_manifest(path))
    for entry in legacy.settings[project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY]:
        entry["format_version"] = 4
        entry.pop("signature", None)
    project_gui._VIEWER_VIEW_CACHE.clear()
    project_gui._COMPOSITE_DATA_CACHE.clear()
    project_gui._restore_project_binning_cache(legacy, path)
    assert not project_gui.project_binnings_need_refresh(legacy)

    incompatible = project_gui._project_from_dict(read_project_manifest(path))
    for entry in incompatible.settings[project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY]:
        entry["format_version"] = 3
    project_gui._VIEWER_VIEW_CACHE.clear()
    project_gui._COMPOSITE_DATA_CACHE.clear()
    project_gui._restore_project_binning_cache(incompatible, path)
    assert project_gui.project_binnings_need_refresh(incompatible)

    changed = project_gui._project_from_dict(read_project_manifest(path))
    changed_dataset = next(changed.data_groups[0].iter_datasets())
    project_gui.dataset_rebin_config(changed_dataset)["minimum_coverage"] = 0.5
    project_gui._VIEWER_VIEW_CACHE.clear()
    project_gui._COMPOSITE_DATA_CACHE.clear()
    project_gui._restore_project_binning_cache(changed, path)
    assert project_gui.project_binnings_need_refresh(changed)


def test_project_cache_persists_every_named_dataset_binning(tmp_path):
    source = tmp_path / "source.nxs"
    source.write_bytes(b"source placeholder")
    dataset = DatasetEntry(
        "scan",
        _grid_mdhisto_data(),
        kind="mdhisto",
        metadata={"source_file": str(source)},
    )
    fit = project_data.dataset_rebin_config(dataset)
    fit.update(enabled=True, minimum_coverage=0.0)
    auxiliary_id = project_data.add_dataset_rebin_binning(dataset, name="Overview")
    auxiliary = project_data.dataset_rebin_config_by_id(dataset, auxiliary_id)
    auxiliary.update(enabled=True, minimum_coverage=0.0)
    auxiliary["axes"][0].update(mode="bins", num_bins=1)
    project = NfitProject(
        [DataGroup("Workspace1", datasets=[dataset])],
        settings={project_gui.PROJECT_CACHE_BINNINGS_KEY: True},
    )
    path = tmp_path / "all-binnings.nfit"

    save_project(project, path)
    manifest = read_project_manifest(path)
    entries = manifest["settings"][project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY]

    assert {entry["binning_id"] for entry in entries} == {
        fit["_binning_id"],
        auxiliary_id,
    }
    project_gui._VIEWER_VIEW_CACHE.clear()
    restored = load_project(path)
    restored_dataset = next(restored.data_groups[0].iter_datasets())
    restored_binnings = project_data.dataset_rebin_binnings(restored_dataset)
    assert [(item["name"], item["fit"]) for item in restored_binnings] == [
        ("Default", True),
        ("Overview", False),
    ]
    assert all(
        project_gui._peek_cached_dataset_view(
            restored_dataset,
            rebin_config=item["config"],
            cache_id=(None if item["fit"] else item["id"]),
        )
        is not None
        for item in restored_binnings
    )


def test_cache_survives_source_load_and_refreshes_only_changed_named_binning(tmp_path):
    source = tmp_path / "source.nxs"
    source.write_bytes(b"source placeholder")
    dataset = DatasetEntry(
        "scan",
        _tiny_mdhisto_data(3.0),
        kind="mdhisto",
        metadata={"source_file": str(source)},
    )
    fit = project_data.dataset_rebin_config(dataset)
    fit.update(enabled=True, minimum_coverage=0.0)
    auxiliary_id = project_data.add_dataset_rebin_binning(dataset, name="Overview")
    auxiliary = project_data.dataset_rebin_config_by_id(dataset, auxiliary_id)
    auxiliary.update(enabled=True, minimum_coverage=0.0)
    group = DataGroup("Workspace1", datasets=[dataset])
    project = NfitProject(
        [group], settings={project_gui.PROJECT_CACHE_BINNINGS_KEY: True}
    )
    progress_events = []
    assert (
        project_gui.prepare_project_binning_cache(
            project, progress_callback=progress_events.append
        )
        == 2
    )
    boundary_events = [
        event for event in progress_events if event.get("stage") == "rebin_batch"
    ]
    assert [
        (
            event["rebin_name"],
            event["rebin_completed"],
            event["rebin_total"],
            bool(event.get("batch_item_complete")),
        )
        for event in boundary_events
    ] == [
        ("Default", 0, 2, False),
        ("Default", 1, 2, True),
        ("Overview", 1, 2, False),
        ("Overview", 2, 2, True),
    ]

    dataset.replace_data(dataset.data, source_backed=True)
    assert not project_gui.project_binnings_need_refresh(project)

    auxiliary["minimum_coverage"] = 0.5
    assert project_gui._project_binning_is_current(
        "dataset", group, dataset, fit["_binning_id"], fit
    )
    assert not project_gui._project_binning_is_current(
        "dataset", group, dataset, auxiliary_id, auxiliary
    )
    refresh_events = []
    assert (
        project_gui.prepare_project_binning_cache(
            project, progress_callback=refresh_events.append
        )
        == 1
    )
    refresh_boundary_events = [
        event for event in refresh_events if event.get("stage") == "rebin_batch"
    ]
    assert [
        (
            event["rebin_name"],
            event["rebin_completed"],
            event["rebin_total"],
        )
        for event in refresh_boundary_events
    ] == [("Overview", 1, 2), ("Overview", 2, 2)]
    assert not project_gui.project_binnings_need_refresh(project)

    dataset.replace_data(dataset.data)
    assert project_gui.project_binnings_need_refresh(project)


def test_rebin_progress_cancel_is_cooperative():
    from PySide6 import QtWidgets

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    explorer = NfitProjectExplorer(NfitProject([DataGroup("Workspace1")]))
    callback = explorer._make_rebin_progress_callback("Rebinning test...")
    controller = callback._nfit_progress_controller
    assert controller.cancel_button.toolTip()
    controller.cancel_button.click()
    with pytest.raises(project_gui.RebinCancellationRequested):
        callback({"stage": "rebin", "iteration": 1, "total": 10})
    explorer._close_rebin_progress(callback)
    application.processEvents()


def test_project_cache_persists_every_named_composite_binning(tmp_path):
    source = tmp_path / "source.nxs"
    source.write_bytes(b"source placeholder")
    group = DataGroup(
        "Workspace1",
        datasets=[
            DatasetEntry(
                "scan",
                _tiny_mdhisto_data(3.0),
                kind="mdhisto",
                metadata={"source_file": str(source)},
            )
        ],
    )
    fit = project_gui._fit_data_group_composite_config(group)
    fit.update(enabled=True, minimum_coverage=0.0)
    auxiliary_id = project_data.add_data_group_composite_binning(
        group, name="Coarse overview"
    )
    auxiliary = project_data.data_group_composite_config_by_id(group, auxiliary_id)
    auxiliary.update(enabled=True, minimum_coverage=0.0)
    project = NfitProject(
        [group], settings={project_gui.PROJECT_CACHE_BINNINGS_KEY: True}
    )
    path = tmp_path / "composite-binnings.nfit"

    save_project(project, path)
    entries = read_project_manifest(path)["settings"][
        project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY
    ]

    assert {entry["binning_id"] for entry in entries} == {
        fit["_binning_id"],
        auxiliary_id,
    }
    assert {entry["type"] for entry in entries} == {"composite"}

    auxiliary["minimum_coverage"] = 0.5
    assert project_gui._project_binning_is_current(
        "composite", group, group, fit["_binning_id"], fit
    )
    assert not project_gui._project_binning_is_current(
        "composite", group, group, auxiliary_id, auxiliary
    )
    assert project_gui.prepare_project_binning_cache(project) == 1
    assert not project_gui.project_binnings_need_refresh(project)


def test_dataset_details_text_summarizes_axes_source_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    source = tmp_path / "scan.nxs"
    source.write_bytes(b"12345")
    data = _tiny_mdhisto_data(1.0)
    data.metadata.update(
        {
            "temperature": 12.5,
            "field": "7 T",
            "sample_environment": {"temperature": 12.5, "field": "7 T", "log": list(range(12))},
            "oriented_lattice": {"orientation_matrix": np.eye(3).tolist()},
        }
    )
    dataset = DatasetEntry(
        "scan",
        data,
        kind="nxs",
        metadata={"source_file": str(source), "timestamp": "2026-01-02T03:04:05"},
        parameters={"sample": "NiO"},
    )
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        lattice_parameters={
            "a": 4.17,
            "b": 4.17,
            "c": 4.17,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
    )
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    text = explorer.details_label.text()
    panel_titles = [
        box.title() for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)
    ]

    tabs = explorer.details_widget.findChild(
        QtWidgets.QTabWidget, "dataset_details_tabs"
    )
    assert tabs is not None
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Overview",
        "Physics",
        "Binning & channels",
        "Metadata",
    ]
    assert set(panel_titles) == {
        "Dataset",
        "Conditions",
        "Signal convention",
        "Axes",
        "Rebin",
        "Crystal",
        "Data",
        "Source",
        "Metadata",
    }
    assert "Axes\nDimensions: 2" in text
    assert "Crystal\nLattice parameters" in text
    assert "a: 4.17" in text
    assert "Oriented lattice orientation matrix" in text
    assert "0: H (rlu) - 1 bins" in text
    assert "1: E (meV) - 1 bins" in text
    assert "File size on disk: 5 B" in text
    assert text.index("Axes") < text.index("Metadata")
    assert "temperature: 12.5" in text
    assert "field: 7 T" in text
    assert "timestamp: 2026-01-02T03:04:05" in text
    assert "sample: NiO" in text

    metadata_tree = explorer.details_widget.findChild(
        QtWidgets.QTreeWidget, "dataset_metadata_tree"
    )
    assert metadata_tree is not None
    assert metadata_tree.maximumHeight() == 16777215
    assert metadata_tree.headerItem().text(0) == "Field"
    assert metadata_tree.headerItem().text(1) == "Value"
    top_level = {
        metadata_tree.topLevelItem(index).text(0): metadata_tree.topLevelItem(index)
        for index in range(metadata_tree.topLevelItemCount())
    }
    assert top_level["sample_environment"].text(1) == "3 field(s)"
    child_names = {
        top_level["sample_environment"].child(index).text(0)
        for index in range(top_level["sample_environment"].childCount())
    }
    assert {"field", "temperature", "log"}.issubset(child_names)


def test_dataset_details_fit_bins_include_file_dataset_and_group_masks():
    data = _grid_mdhisto_data()
    editable = data.mutable_copy()
    editable.mask[0, 0] = True
    data = editable.immutable_copy()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    mask = create_mask(dataset, type="box")
    mask.parameters["axes"] = ["H", "E"]
    mask.parameters["center"] = [1.5, 5.0]
    mask.parameters["width"] = [0.1, 0.1]
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        masks=[
            MaskSpec(
                "SharedMask",
                type="box",
                parameters={
                    "axes": ["H", "E"],
                    "center": [0.5, 15.0],
                    "width": [0.1, 0.1],
                },
            )
        ],
    )

    text = dataset_details_text(dataset, group=group)

    assert "Total bins: 4" in text
    assert "Unmasked bins: 3" in text
    assert "Fit bins: 1 of 4" in text
    viewed = dataset_for_slice_viewer(
        dataset, extra_masks=project_gui.effective_dataset_masks(group, dataset)
    )
    assert int(np.count_nonzero(~viewed.mask)) == 1


def test_dataset_rebin_config_updates_slice_viewer_materializes_and_saves(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto", parameters={"temperature": 12.5})
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    config = dataset_rebin_config(dataset)
    assert config["axes"][0]["num_bins"] == 2
    assert config["axes"][1]["num_bins"] == 2
    assert [axis["mode"] for axis in config["axes"]] == ["step", "step"]
    assert config["max_batch_mb"] == 192
    assert config["auto_rebin"] is True
    assert config["minimum_coverage"] == pytest.approx(0.9)

    enable_check = explorer.details_widget.findChild(QtWidgets.QCheckBox, "dataset_rebin_enabled")
    assert enable_check is not None
    rebin_panel = next(
        box
        for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)
        if box.title() == "Rebin"
    )
    assert rebin_panel.findChild(QtWidgets.QLabel, "dataset_rebin_axis_variable_0").text() == "H"
    assert rebin_panel.findChild(QtWidgets.QLabel, "dataset_rebin_axis_variable_1").text() == "E"
    assert (
        "Scalar variable"
        in rebin_panel.findChild(QtWidgets.QLabel, "dataset_rebin_axis_variable_0").toolTip()
    )
    assert rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_fractional") is None
    tabs = rebin_panel.findChild(QtWidgets.QTabWidget, "dataset_rebin_controls")
    assert tabs is not None
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Rebin settings",
        "Bin information",
    ]
    memory_estimate = rebin_panel.findChild(
        QtWidgets.QLabel, "dataset_rebin_memory_estimate"
    )
    assert memory_estimate is not None
    assert "result memory" in memory_estimate.text()
    assert "Compressed disk size: -" in memory_estimate.text()
    assert memory_estimate.toolTip()
    assignment = rebin_panel.findChild(
        QtWidgets.QComboBox, "dataset_rebin_axis_assignment_0"
    )
    assert assignment is not None and assignment.currentData() is True
    assert assignment.toolTip()
    bin_tree = rebin_panel.findChild(QtWidgets.QTreeWidget, "dataset_rebin_bin_tree")
    assert bin_tree is not None and bin_tree.topLevelItemCount() == 2
    auto_check = rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_auto")
    assert auto_check is not None
    assert auto_check.isChecked()
    assert rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_fit_enabled") is None
    mean_combo = rebin_panel.findChild(QtWidgets.QComboBox, "dataset_rebin_mean_weighting")
    assert mean_combo is not None
    assert mean_combo.currentData() == "uniform"
    coverage_edit = rebin_panel.findChild(
        QtWidgets.QLineEdit, "dataset_rebin_minimum_coverage"
    )
    assert coverage_edit is not None
    assert float(coverage_edit.text()) == pytest.approx(0.9)
    assert coverage_edit.toolTip()
    coverage_edit.setText("0.8")
    coverage_edit.editingFinished.emit()
    assert dataset_rebin_config(dataset)["minimum_coverage"] == pytest.approx(0.8)
    batch_spin = rebin_panel.findChild(QtWidgets.QSpinBox, "dataset_rebin_max_batch_mb")
    assert batch_spin is not None
    assert batch_spin.value() == 192
    assert "not a cap on total rebinner memory use" in batch_spin.toolTip()
    batch_spin.setValue(64)
    assert dataset_rebin_config(dataset)["max_batch_mb"] == 64
    create_button = rebin_panel.findChild(QtWidgets.QPushButton, "dataset_rebin_create")
    rebin_now_button = rebin_panel.findChild(QtWidgets.QPushButton, "dataset_rebin_now")
    rebin_all_button = rebin_panel.findChild(
        QtWidgets.QPushButton, "dataset_rebin_all_now"
    )
    save_rebin_button = rebin_panel.findChild(QtWidgets.QPushButton, "dataset_rebin_save")
    assert create_button is not None
    assert rebin_now_button is not None
    assert rebin_now_button.text() == "Rebin now"
    assert rebin_all_button is not None and rebin_all_button.isHidden()
    assert rebin_all_button.toolTip()
    assert create_button.text() == "Create dataset from rebin"
    assert save_rebin_button is not None
    assert save_rebin_button.text() == "Save rebin to disk"
    enable_check.setChecked(True)
    explorer._set_dataset_rebin_axis_value(dataset, group, 0, "num_bins", "1")
    explorer._set_dataset_rebin_axis_value(dataset, group, 1, "num_bins", "1")
    assert dataset_rebin_config(dataset)["axes"][0]["step_size"] == 1.0
    explorer._set_dataset_rebin_axis_value(dataset, group, 0, "step_size", "0.75")
    axis_config = dataset_rebin_config(dataset)["axes"][0]
    assert axis_config["num_bins"] == 2
    assert axis_config["step_size"] == pytest.approx(0.75)


    viewed = dataset_for_slice_viewer(dataset)

    assert viewed is not None
    assert viewed.shape[0] >= 1
    energy_step = viewed.metadata["rebin"]["step_size"][1]
    np.testing.assert_allclose(
        viewed.axes[1].centers / energy_step,
        np.rint(viewed.axes[1].centers / energy_step),
    )
    assert viewed.metadata["rebin"]["normalize"] is True
    assert viewed.metadata["combined_mask_count"] == int(np.count_nonzero(viewed.mask))

    dialog_save_path = tmp_path / "dialog-rebinned.npz"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(dialog_save_path), "NumPy archives (*.npz)"),
    )
    assert explorer.save_rebin_for_selection()
    dialog_saved = np.load(dialog_save_path)
    assert dialog_saved["signal"].shape == viewed.shape
    assert int(dialog_saved["axis_count"]) == 2

    rebinned = explorer.materialize_rebin_for_selection()

    assert rebinned is not None
    assert group.dataset_names == ["scan", "scan rebinned"]
    assert isinstance(rebinned.data, MDHistoData)
    assert rebinned.data.shape == viewed.shape
    assert rebinned.parameters["temperature"] == pytest.approx(12.5)

    save_path = tmp_path / "rebinned.npz"
    save_dataset_file(dataset, save_path)

    saved = np.load(save_path)
    assert saved["signal"].shape == viewed.shape
    assert int(saved["axis_count"]) == 2


def test_named_dataset_binnings_are_independent_and_one_is_designated_for_fitting():
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    fit_config = project_data.dataset_rebin_config(dataset)
    fit_config["axes"][0]["step_size"] = 0.5

    auxiliary_id = project_data.add_dataset_rebin_binning(
        dataset,
        name="Fine HK",
        duplicate_from=fit_config["_binning_id"],
    )
    auxiliary = project_data.dataset_rebin_config_by_id(dataset, auxiliary_id)
    auxiliary["axes"][0]["step_size"] = 0.1

    assert fit_config["axes"][0]["step_size"] == pytest.approx(0.5)
    assert [item["name"] for item in project_data.dataset_rebin_binnings(dataset)] == [
        "Default",
        "Fine HK",
    ]
    assert [item["fit"] for item in project_data.dataset_rebin_binnings(dataset)] == [
        True,
        False,
    ]

    project_data.make_dataset_fit_binning(dataset, auxiliary_id)
    binnings = project_data.dataset_rebin_binnings(dataset)
    assert [(item["name"], item["fit"]) for item in binnings] == [
        ("Fine HK", True),
        ("Default", False),
    ]
    assert binnings[0]["config"]["axes"][0]["step_size"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="fit binning cannot be removed"):
        project_data.remove_dataset_rebin_binning(dataset, auxiliary_id)


def test_dataset_rebin_panel_selects_and_edits_named_binning(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    fit = project_data.dataset_rebin_config(dataset)
    auxiliary_id = project_data.add_dataset_rebin_binning(dataset, name="Wide view")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    combo = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "dataset_rebin_binning"
    )
    assert combo is not None and combo.toolTip()
    assert [combo.itemText(index) for index in range(combo.count())] == [
        "Default (fit)",
        "Wide view",
    ]
    rebin_all_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "dataset_rebin_all_now"
    )
    assert rebin_all_button is not None and not rebin_all_button.isHidden()
    explorer._select_dataset_binning(dataset, auxiliary_id)
    project_gui.dataset_rebin_config(dataset)["axes"][0]["step_size"] = 0.125

    assert fit["axes"][0]["step_size"] != pytest.approx(0.125)
    assert project_data.dataset_rebin_config_by_id(dataset, auxiliary_id)["axes"][0][
        "step_size"
    ] == pytest.approx(0.125)
    assert explorer.details_widget.findChild(
        QtWidgets.QCheckBox, "dataset_rebin_fit_binning"
    ).toolTip()
    explorer.window.close()


def test_rebin_all_dataset_binnings_populates_viewer_entries_and_refreshes(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    project_gui._VIEWER_VIEW_CACHE.clear()
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    fit = project_data.dataset_rebin_config(dataset)
    fit.update(enabled=True, minimum_coverage=0.0, stale=True)
    auxiliary_id = project_data.add_dataset_rebin_binning(dataset, name="Overview")
    auxiliary = project_data.dataset_rebin_config_by_id(dataset, auxiliary_id)
    auxiliary.update(enabled=True, minimum_coverage=0.0, stale=True)
    auxiliary["axes"][0].update(mode="bins", num_bins=1)
    explorer = NfitProjectExplorer(NfitProject([group]))
    refreshes = []
    monkeypatch.setattr(explorer, "_make_rebin_progress_callback", lambda *a, **k: None)
    monkeypatch.setattr(explorer, "_close_rebin_progress", lambda _progress: None)
    monkeypatch.setattr(explorer, "_set_dataset_details", lambda *_args: None)
    monkeypatch.setattr(explorer, "refresh_slice_viewer", refreshes.append)

    assert explorer.rebin_all_dataset_binnings_now(dataset, group)
    assert fit["stale"] is False
    assert auxiliary["stale"] is False
    assert refreshes == [group]
    assert project_gui._peek_cached_dataset_view(dataset) is not None
    assert project_gui._peek_cached_dataset_view(
        dataset, rebin_config=auxiliary, cache_id=auxiliary_id
    ) is not None
    _datasets, names = project_gui.slice_viewer_datasets(group, force_rebin=False)
    assert names == ["scan", "scan · Overview"]


def test_named_visualization_binning_is_zero_weight_and_viewer_selectable(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    fit = project_data.dataset_rebin_config(dataset)
    fit.update(enabled=True, minimum_coverage=0.0)
    auxiliary_id = project_data.add_dataset_rebin_binning(dataset, name="Overview")
    auxiliary = project_data.dataset_rebin_config_by_id(dataset, auxiliary_id)
    auxiliary.update(enabled=True, minimum_coverage=0.0)
    auxiliary["axes"][0].update(mode="bins", num_bins=1)

    fit_inputs, _fit_bundles = project_gui.fit_dataset_inputs(group, purpose="fit")
    view_inputs, _view_bundles = project_gui.fit_dataset_inputs(
        group, purpose="visualization"
    )
    datasets, names = project_gui.slice_viewer_datasets(group)

    assert [(item.name, item.weight) for item in fit_inputs] == [("scan", 1.0)]
    assert [(item.name, item.weight) for item in view_inputs] == [
        ("scan · Overview", 0.0)
    ]
    component = ModelComponentSpec(
        "background",
        "constant_background",
        applies_to=["scan"],
        parameters={"constant": 1.0},
        fit_parameters={"constant": True},
        sharing={"constant": {"mode": "per_dataset"}},
        metadata={"fitted_values": {"constant": {"scan": 7.5}}},
    )
    group.models[component.name] = component
    overlay_components = project_gui._components_with_binning_aliases(
        [component], view_inputs
    )
    compiled = project_gui.compile_fit_problem(overlay_components, view_inputs)
    overlay_params = project_gui._overlay_current_params(group, compiled)
    assert next(
        value
        for name, value in overlay_params.items()
        if name.startswith("background.constant")
    ) == pytest.approx(7.5)
    assert names == ["scan", "scan · Overview"]
    assert [item.metadata["binning_name"] for item in datasets] == [
        "Default",
        "Overview",
    ]

    viewer = QtMDHistoSliceViewer(datasets, dataset_names=names)
    assert viewer.dataset_combo.count() == 1
    assert viewer.dataset_combo.currentText() == "scan"
    assert [
        viewer.binning_combo.itemText(index)
        for index in range(viewer.binning_combo.count())
    ] == ["Default", "Overview"]
    viewer.binning_combo.setCurrentIndex(1)
    assert viewer.dataset_index == 1
    viewer.window.close()


def test_materialized_composite_round_trips_as_project_owned_dataset(tmp_path):
    first = DatasetEntry(
        "first",
        _tiny_mdhisto_data(2.0),
        kind="mdhisto",
        metadata={"source_file": str(tmp_path / "first.npz")},
    )
    second = DatasetEntry(
        "second",
        _tiny_mdhisto_data(4.0),
        kind="mdhisto",
        metadata={"source_file": str(tmp_path / "second.npz")},
    )
    group = DataGroup("Workspace", datasets=[first, second])
    project = NfitProject([group])
    path = tmp_path / "materialized.nfit"
    save_project(project, path)
    config = project_gui.data_group_composite_config(group)
    config.update({"enabled": True, "mean_weighting": "uniform"})
    for axis in config["axes"]:
        axis["mode"] = "discrete"

    entry = project_gui.materialize_composite_dataset(path, group)
    save_project(project, path)
    restored = load_project(path)
    restored_entry = next(
        dataset
        for dataset in restored.data_groups[0].iter_datasets()
        if dataset.id == entry.id
    )

    assert restored.data_groups[0].subgroups[-1].name == "Materialized data"
    assert not restored.data_groups[0].subgroups[-1].enabled
    assert restored_entry.metadata["project_artifact_path"].startswith(
        f"assets/datasets/{entry.id}/"
    )
    assert restored_entry.data is None
    restored_data = dataset_for_slice_viewer(restored_entry)
    assert isinstance(restored_data, MDHistoData)
    np.testing.assert_allclose(restored_data.signal[~restored_data.mask], 3.0)


def test_rebin_settings_copy_and_paste_between_datasets(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    source = DatasetEntry("source", _grid_mdhisto_data(), kind="mdhisto")
    target = DatasetEntry("target", _grid_mdhisto_data(), kind="mdhisto")
    source_config = dataset_rebin_config(source)
    source_config.update(
        {
            "enabled": True,
            "auto_rebin": False,
            "mean_weighting": "uniform",
            "max_batch_mb": 64,
            "stale": False,
        }
    )
    source_config["axes"][0].update({"lower": -3.0, "upper": 4.0, "num_bins": 14, "step_size": 0.5})
    source_config["symmetry"] = {
        "mode": "point_group",
        "expression": "-1",
    }
    group = DataGroup("Datagroup1", datasets=[source, target])
    explorer = NfitProjectExplorer(NfitProject([group]))
    datasets_item = explorer.tree.topLevelItem(0).child(0)

    explorer.tree.setCurrentItem(datasets_item.child(0))
    copy_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "dataset_rebin_copy_settings"
    )
    assert copy_button is not None
    assert copy_button.toolTip()
    copy_button.click()
    clipboard_payload = json.loads(QtWidgets.QApplication.clipboard().text())
    assert clipboard_payload["schema"] == "nfit.rebin-settings"
    assert "stale" not in clipboard_payload["settings"]

    explorer.tree.setCurrentItem(datasets_item.child(1))
    paste_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "dataset_rebin_paste_settings"
    )
    assert paste_button is not None
    assert paste_button.toolTip()
    paste_button.click()

    pasted = dataset_rebin_config(target)
    for key in project_gui.REBIN_SETTINGS_KEYS:
        if key in source_config:
            assert pasted[key] == source_config[key]
    assert pasted["stale"] is True
    assert explorer.details_widget.findChild(
        QtWidgets.QCheckBox, "dataset_rebin_enabled"
    ).isChecked()
    assert explorer.details_widget.findChild(
        QtWidgets.QComboBox, "dataset_rebin_axis_mode_0"
    ).currentData() == source_config["axes"][0]["mode"]


def test_rebin_settings_paste_rejects_incompatible_axis_count():
    source = {
        "enabled": True,
        "axes": [
            {"name": "H", "lower": -1.0, "upper": 1.0, "num_bins": 2},
            {"name": "K", "lower": -1.0, "upper": 1.0, "num_bins": 2},
        ],
    }
    target = {
        "enabled": False,
        "axes": [{"name": "q", "lower": 0.0, "upper": 2.0, "num_bins": 2}],
    }

    with pytest.raises(ValueError, match="2 axes.*1 axes"):
        project_gui._rebin_config_from_clipboard_text(
            project_gui._rebin_settings_clipboard_text(source),
            target,
        )

    assert target["enabled"] is False


def test_saved_nfit_npz_import_restores_mdhisto_axes_data_and_context(tmp_path):
    data = _grid_mdhisto_data()
    editable = data.mutable_copy()
    editable.mask[0, 0] = True
    editable.signal[0, 0] = np.nan
    editable.errors[0, 0] = np.nan
    editable.num_events[0, 0] = 0.0
    data = editable.immutable_copy().with_updates(
        coordinate_system=2,
        visual_normalization=1,
        metadata={**data.metadata, "signal_semantics": "density"},
    )
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    dataset.parameters["temperature"] = 1.8
    dataset.parameters["magnetic_field"] = {
        "magnitude_T": 4.0,
        "direction": [0.0, 0.0, 1.0],
        "frame": "uvw",
    }
    path = tmp_path / "scan.npz"

    save_dataset_file(dataset, path)
    imported = dataset_entry_from_path(path)

    assert isinstance(imported.data, MDHistoData)
    np.testing.assert_allclose(imported.data.signal, data.signal)
    assert [axis.name for axis in imported.data.axes] == [axis.name for axis in data.axes]
    assert [axis.units for axis in imported.data.axes] == [axis.units for axis in data.axes]
    assert imported.parameters["temperature"] == pytest.approx(1.8)
    assert imported.parameters["magnetic_field"] == dataset.parameters["magnetic_field"]
    assert imported.data.metadata["signal_semantics"] == "density"
    assert imported.data.coordinate_system == 2
    assert imported.data.visual_normalization == 1
    reloaded_view = dataset_for_slice_viewer(imported)
    np.testing.assert_array_equal(reloaded_view.mask, data.mask)
    np.testing.assert_allclose(reloaded_view.signal, data.signal, equal_nan=True)
    np.testing.assert_allclose(reloaded_view.errors, data.errors, equal_nan=True)
    np.testing.assert_allclose(reloaded_view.num_events, data.num_events)


def test_nfit_npz_import_requires_signal_semantics(tmp_path):
    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    modern_path = tmp_path / "modern.npz"
    incomplete_path = tmp_path / "incomplete.npz"
    save_dataset_file(dataset, modern_path)

    with np.load(modern_path) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    metadata.pop("signal_semantics", None)
    payload["metadata_json"] = json.dumps(metadata)
    np.savez_compressed(incomplete_path, **payload)

    with pytest.raises(ValueError, match="missing signal_semantics metadata"):
        dataset_entry_from_path(incomplete_path)


def test_dataset_signal_semantics_control_is_documented_and_updates_data(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    data = _grid_mdhisto_data()
    data.metadata["signal_semantics"] = "bin_integral"
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer._set_dataset_details(dataset, group)
    combo = explorer.details_widget.findChild(QtWidgets.QComboBox, "dataset_signal_semantics")
    assert combo is not None
    assert "Bin-integral" in combo.toolTip()
    combo.setCurrentIndex(combo.findData("density"))

    assert dataset.data.metadata["signal_semantics"] == "density"
    assert dataset.data.metadata["signal_semantics_source"] == "user_selected"


def test_dataset_rebin_edits_preserve_details_scroll_position(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    data = _grid_mdhisto_data()
    data.metadata["notes"] = {f"entry_{index}": f"value {index}" for index in range(30)}
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.window.resize(760, 280)
    explorer.window.show()
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))
    explorer.details_scroll.setFixedHeight(180)
    explorer.details_widget.setMinimumHeight(1000)
    QtWidgets.QApplication.processEvents()

    project_gui.dataset_rebin_config(dataset)["auto_rebin"] = False
    enable_rebin = explorer.details_widget.findChild(QtWidgets.QCheckBox, "dataset_rebin_enabled")
    assert enable_rebin is not None
    enable_rebin.setChecked(True)
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()

    scrollbar = explorer.details_scroll.verticalScrollBar()
    assert scrollbar.maximum() > 0
    position = max(1, scrollbar.maximum() // 2)
    scrollbar.setValue(position)
    assert scrollbar.value() == position

    editor = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "dataset_rebin_resolution_value_0"
    )
    assert editor is not None
    editor.setFocus()
    editor.setText("0.75")
    editor.editingFinished.emit()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()

    assert scrollbar.value() == min(position, scrollbar.maximum())
    assert (
        explorer.details_widget.findChild(QtWidgets.QLineEdit, "dataset_rebin_resolution_value_0")
        is editor
    )
    assert explorer.window.focusWidget().objectName() == "dataset_rebin_resolution_value_0"


def test_large_dataset_rebin_defaults_manual_and_defers_refresh(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    monkeypatch.setattr(project_data, "REBIN_AUTO_MAX_CONTRIBUTIONS", 1)
    project_gui._VIEWER_VIEW_CACHE.clear()

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    config = dataset_rebin_config(dataset)
    assert config["auto_rebin"] is False
    rebin_panel = next(
        box
        for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)
        if box.title() == "Rebin"
    )
    auto_check = rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_auto")
    status = rebin_panel.findChild(QtWidgets.QLabel, "dataset_rebin_status")
    assert auto_check is not None and not auto_check.isChecked()
    assert status is not None
    assert "manual" in status.text().lower() or "disabled" in status.text().lower()

    enable_check = rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_enabled")
    assert enable_check is not None
    enable_check.setChecked(True)
    refreshes = []
    monkeypatch.setattr(explorer, "refresh_slice_viewer", lambda group: refreshes.append(group))
    explorer._set_dataset_rebin_axis_value(dataset, group, 0, "num_bins", "1")

    assert refreshes == []
    assert dataset_rebin_config(dataset)["stale"] is True
    deferred = project_gui._viewer_data_before_scale(dataset, force_rebin=False)
    assert deferred is not None
    assert deferred.shape == data.shape
    assert dataset_rebin_config(dataset)["stale"] is True

    rebin_now_button = rebin_panel.findChild(QtWidgets.QPushButton, "dataset_rebin_now")
    assert rebin_now_button is not None and rebin_now_button.isEnabled()
    rebin_now_button.click()
    QtWidgets.QApplication.processEvents()
    assert dataset_rebin_config(dataset)["stale"] is False
    assert refreshes == [group]
    forced = dataset_for_slice_viewer(dataset)
    assert forced is not None
    assert forced.shape[0] >= 1
    assert np.any(np.isclose(forced.axes[0].centers, 0.0))
    explorer._set_dataset_details(dataset, group)
    bin_summary = explorer.details_widget.findChild(
        QtWidgets.QLabel, "dataset_rebin_bin_shape"
    )
    assert bin_summary is not None
    assert "Resolved cached grid" in bin_summary.text()
    assert str(forced.shape) in bin_summary.text()


def test_dataset_rebin_axis_mode_switches_between_step_bins_and_tolerance(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))
    config = dataset_rebin_config(dataset)
    config["auto_rebin"] = False
    config["enabled"] = True

    mode_combo = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "dataset_rebin_axis_mode_0"
    )
    value_edit = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "dataset_rebin_resolution_value_0"
    )
    assert mode_combo is not None and mode_combo.currentData() == "step"
    assignment_combo = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "dataset_rebin_axis_assignment_0"
    )
    assert assignment_combo is not None and assignment_combo.currentData() is True
    assignment_combo.setCurrentIndex(assignment_combo.findData(False))
    assert config["axes"][0]["mode"] == "step"
    assert config["axes"][0]["fractional"] is False
    assert value_edit is not None and float(value_edit.text()) == pytest.approx(1.0)

    mode_combo.setCurrentIndex(mode_combo.findData("bins"))
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()
    assert config["axes"][0]["mode"] == "bins"
    value_edit = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "dataset_rebin_resolution_value_0"
    )
    assert value_edit is not None and value_edit.text() == "2"

    mode_combo = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "dataset_rebin_axis_mode_0"
    )
    mode_combo.setCurrentIndex(mode_combo.findData("tolerance"))
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()
    assignment_combo = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "dataset_rebin_axis_assignment_0"
    )
    assert assignment_combo.currentData() is False
    assert not assignment_combo.isEnabled()

    mode_combo = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "dataset_rebin_axis_mode_0"
    )
    mode_combo.setCurrentIndex(mode_combo.findData("bins"))

    explorer._set_dataset_rebin_axis_value(dataset, group, 0, "lower", "-1")
    assert config["axes"][0]["num_bins"] == 2
    assert config["axes"][0]["step_size"] == pytest.approx(2.5)

    mode_combo = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "dataset_rebin_axis_mode_0"
    )
    mode_combo.setCurrentIndex(mode_combo.findData("step"))
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()
    assert config["axes"][0]["mode"] == "step"
    explorer._set_dataset_rebin_axis_value(dataset, group, 0, "step_size", "0.75")
    explorer._set_dataset_rebin_axis_value(dataset, group, 0, "lower", "-0.8")
    explorer._set_dataset_rebin_axis_value(dataset, group, 0, "upper", "2")
    assert config["axes"][0]["num_bins"] == 4
    assert config["axes"][0]["step_size"] == pytest.approx(0.75)
    rebinned = dataset_for_slice_viewer(dataset)
    assert isinstance(rebinned, MDHistoData)
    assert rebinned.axes[0].centers == pytest.approx([-0.8, -0.05, 0.7, 1.45])
    assert rebinned.axes[0].values == pytest.approx([-1.175, -0.425, 0.325, 1.075, 1.825])


def test_data_group_composite_uses_scale_fit_weight_and_rebinning():
    first = DatasetEntry(
        "first", _tiny_mdhisto_data(1.0), kind="mdhisto", data_type="single_crystal_inelastic"
    )
    second = DatasetEntry(
        "second",
        _tiny_mdhisto_data(5.0),
        kind="mdhisto",
        data_type="single_crystal_inelastic",
        scale_factor=-1.0,
        fit_weight=3.0,
    )
    group = DataGroup("Datagroup1", datasets=[first, second])
    config = project_gui.data_group_composite_config(group)
    config["enabled"] = True
    for axis in config["axes"]:
        axis["mode"] = "discrete"

    composite = project_gui.composite_dataset_data(group)

    assert isinstance(composite, MDHistoData)
    # Averaging weights are fit_weight / sigma^2 = [1, 3] with unit sigma, so
    # the composite is 0.25 * 1 + 0.75 * (-5) and its uncertainty is the
    # weighted-mean propagation sqrt(1 + 9) / 4.
    np.testing.assert_allclose(composite.signal[~composite.mask], -3.5)
    np.testing.assert_allclose(composite.errors[~composite.mask], np.sqrt(10.0) / 4.0)
    np.testing.assert_allclose(composite.num_events[~composite.mask], 2.0)
    assert composite.metadata["rebin"]["weighted_by_fit_weight"] is True

    datasets, names = project_gui.slice_viewer_datasets(group)
    assert names == ["Datagroup1 Composite"]
    np.testing.assert_allclose(datasets[0].signal[~datasets[0].mask], -3.5)

    constituent_datasets, constituent_names = project_gui.slice_viewer_datasets(
        group, use_composite=False
    )
    assert constituent_names == ["first", "second"]
    np.testing.assert_allclose(
        constituent_datasets[1].signal[~constituent_datasets[1].mask], -5.0
    )

    inputs, bundles = project_gui.fit_dataset_inputs(group)
    assert [item.name for item in inputs] == ["Datagroup1 Composite"]
    assert list(bundles) == ["Datagroup1 Composite"]


def test_point_data_rebin_automatically_uses_normalization_denominator():
    data = PointData4D(
        H=[0.0, 0.0],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[0.0, 0.0],
        intensity=[0.0, 10.0],
        sigma=[1.0, 1.0],
        normalization_denominator=[1.0, 3.0],
    )
    dataset = DatasetEntry("points", data, data_type="single_crystal_inelastic")
    config = dataset_rebin_config(dataset)
    config.update(
        {
            "enabled": True,
            "mean_weighting": "uniform",
            "minimum_coverage": 0.0,
        }
    )
    for axis in config["axes"]:
        axis.update(
            lower=-0.5,
            upper=0.5,
            auto_lower=False,
            auto_upper=False,
            num_bins=1,
            mode="bins",
        )

    rebinned = project_gui.rebinned_dataset_data(dataset)

    np.testing.assert_allclose(rebinned.signal[~rebinned.mask], [7.5])
    np.testing.assert_allclose(rebinned.errors[~rebinned.mask], [np.sqrt(10.0) / 4.0])
    assert rebinned.metadata["rebin"][
        "weighted_by_normalization_denominator"
    ] is True


def test_mdhisto_rebin_automatically_uses_normalization_denominator():
    data = MDHistoData(
        axes=(
            MDHistoAxis("|Q|", [0.5, 1.5, 2.5], "1/angstrom", "momentum"),
            MDHistoAxis("DeltaE", [-1.5, -0.5, 0.5], "meV", "energy"),
        ),
        signal=np.asarray([[0.0, 10.0], [0.0, 10.0]]),
        errors=np.ones((2, 2)),
        mask=np.zeros((2, 2), dtype=bool),
        num_events=np.ones((2, 2)),
        auxiliary_channels={
            "normalization_denominator": MDHistoChannel(
                np.asarray([[1.0, 3.0], [1.0, 3.0]])
            )
        },
    )
    dataset = DatasetEntry("histogram", data, data_type="powder_inelastic")
    config = dataset_rebin_config(dataset)
    config.update(
        {
            "enabled": True,
            "mean_weighting": "uniform",
            "minimum_coverage": 0.0,
        }
    )
    for axis, lower, upper in zip(
        config["axes"], (0.5, -1.5), (2.5, 0.5), strict=True
    ):
        axis.update(
            lower=lower,
            upper=upper,
            auto_lower=False,
            auto_upper=False,
            num_bins=1,
            mode="bins",
        )

    rebinned = project_gui.rebinned_dataset_data(dataset)

    np.testing.assert_allclose(rebinned.signal[~rebinned.mask], [7.5])
    np.testing.assert_allclose(rebinned.errors[~rebinned.mask], [np.sqrt(20.0) / 8.0])
    assert rebinned.metadata["rebin"][
        "weighted_by_normalization_denominator"
    ] is True
    np.testing.assert_allclose(
        rebinned.auxiliary_channels["normalization_denominator"].values,
        [[8.0]],
    )


def test_point_data_tolerance_axis_derives_nominal_energy_bins_without_leakage():
    energies = np.asarray(
        [-0.402, -0.398, -0.003, 0.004, 0.398, 0.403, 1.198, 1.203, 2.0, 2.8, 3.6]
    )
    data = PointData4D(
        H=np.zeros(energies.size),
        K=np.zeros(energies.size),
        L=np.zeros(energies.size),
        E=energies,
        intensity=np.arange(energies.size, dtype=float),
        sigma=np.ones(energies.size),
    )
    dataset = DatasetEntry("points", data, data_type="single_crystal_inelastic")
    config = dataset_rebin_config(dataset)
    config.update(enabled=True, minimum_coverage=0.0)
    for axis in config["axes"][:3]:
        axis["mode"] = "discrete"
    config["axes"][3].update(mode="tolerance", tolerance=0.1)

    rebinned = project_gui.rebinned_dataset_data(dataset)

    np.testing.assert_allclose(
        rebinned.axes[3].centers,
        [-0.4, 0.0005, 0.4005, 1.2005, 2.0, 2.8, 3.6],
    )
    assert rebinned.shape == (1, 1, 1, 7)
    assert rebinned.metadata["rebin"]["fractional_axes"] == [False] * 4


def test_data_group_composite_controls_show_summary_and_update_config(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    first = DatasetEntry(
        "first", _tiny_mdhisto_data(1.0), kind="mdhisto", data_type="single_crystal_inelastic"
    )
    second = DatasetEntry(
        "second", _tiny_mdhisto_data(2.0), kind="mdhisto", data_type="single_crystal_inelastic"
    )
    second.scale_factor = -1.0
    second.fit_weight = 2.0
    group = DataGroup("Datagroup1", datasets=[first, second])
    initial = project_gui.data_group_composite_config(group)
    initial["axes"][1].update(
        lower=1.3,
        upper=1.3,
        auto_lower=True,
        auto_upper=True,
        auto_lower_value=0.0,
        auto_upper_value=2.0,
    )
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0))

    text = explorer.details_label.text()
    assert "Datasets (incl. nested): 2" in text
    checkbox = explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_enabled")
    assert checkbox is not None
    assert checkbox.toolTip()
    batch_spin = explorer.details_widget.findChild(
        QtWidgets.QSpinBox, "group_composite_max_batch_mb"
    )
    assert batch_spin is not None
    assert batch_spin.value() == 192
    auto_check = explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_auto")
    assert auto_check is not None
    assert auto_check.isChecked()
    tabs = explorer.details_widget.findChild(QtWidgets.QTabWidget, "group_composite_tabs")
    assert tabs is not None
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Rebin settings",
        "Metadata dimensions",
        "Physics",
        "Bin information",
    ]
    memory_estimate = explorer.details_widget.findChild(
        QtWidgets.QLabel, "group_composite_memory_estimate"
    )
    assert memory_estimate is not None
    assert "result memory" in memory_estimate.text()
    assert "Compressed disk size: -" in memory_estimate.text()
    binning_combo = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "group_composite_binning"
    )
    assert binning_combo is not None and binning_combo.currentText() == "Default (fit)"
    for object_name in (
        "group_composite_add_binning",
        "group_composite_duplicate_binning",
        "group_composite_rename_binning",
        "group_composite_remove_binning",
    ):
        button = explorer.details_widget.findChild(QtWidgets.QPushButton, object_name)
        assert button is not None and button.toolTip()
    fit_binning = explorer.details_widget.findChild(
        QtWidgets.QCheckBox, "group_composite_fit_binning"
    )
    assert fit_binning is not None and fit_binning.isChecked() and fit_binning.toolTip()
    assert explorer.details_widget.findChild(
        QtWidgets.QTreeWidget, "group_composite_bin_tree"
    ) is not None
    rebin_now_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "group_composite_rebin_now"
    )
    assert rebin_now_button is not None
    assert rebin_now_button.toolTip()
    rebin_all_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "group_composite_rebin_all_now"
    )
    assert rebin_all_button is not None and rebin_all_button.isHidden()
    assert rebin_all_button.toolTip()
    materialize_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "group_composite_create"
    )
    controls_layout = materialize_button.parentWidget().layout()
    action_layout = next(
        controls_layout.itemAt(index).layout()
        for index in range(controls_layout.count())
        if controls_layout.itemAt(index).layout() is not None
        and controls_layout.itemAt(index).layout().indexOf(materialize_button) >= 0
    )
    assert action_layout.indexOf(rebin_now_button) < action_layout.indexOf(materialize_button)
    assert action_layout.itemAt(action_layout.indexOf(materialize_button) - 1).spacerItem()
    copy_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "group_composite_copy_settings"
    )
    paste_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "group_composite_paste_settings"
    )
    assert copy_button is not None and copy_button.toolTip()
    assert paste_button is not None and paste_button.toolTip()

    lower_edit = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "group_composite_axis_lower_1"
    )
    upper_edit = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "group_composite_axis_upper_1"
    )
    assert lower_edit.text() == upper_edit.text() == "1.3"
    assert not project_gui._rebin_axis_bound_is_auto(initial["axes"][1], "lower")

    final_axis_label = explorer.details_widget.findChild(
        QtWidgets.QLabel, "group_composite_axis_label_1"
    )
    assert final_axis_label is not None
    controls_grid = final_axis_label.parentWidget().layout()
    final_axis_row = controls_grid.getItemPosition(
        controls_grid.indexOf(final_axis_label)
    )[0]
    footer_rows = [
        controls_grid.getItemPosition(index)[0]
        for index in range(controls_grid.count())
        if controls_grid.itemAt(index).layout() is not None
        and controls_grid.itemAt(index).layout().indexOf(auto_check) >= 0
    ]
    assert footer_rows == [final_axis_row + 1]

    checkbox.setChecked(True)

    config = project_gui.data_group_composite_config(group)
    assert config["enabled"] is True
    assert config["mean_weighting"] == "uniform"
    coverage_edit = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "group_composite_minimum_coverage"
    )
    assert coverage_edit is not None
    assert float(coverage_edit.text()) == pytest.approx(0.9)
    coverage_edit.setText("0.85")
    coverage_edit.editingFinished.emit()
    assert config["minimum_coverage"] == pytest.approx(0.85)
    resolution_mode = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "group_composite_axis_mode_0"
    )
    assert resolution_mode is not None and resolution_mode.currentData() == "step"
    resolution_mode.setCurrentIndex(resolution_mode.findData("bins"))
    assert config["axes"][0]["mode"] == "bins"
    copy_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "group_composite_copy_settings"
    )
    paste_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "group_composite_paste_settings"
    )
    copied_lower = config["axes"][0]["lower"]
    copy_button.click()
    config["axes"][0]["lower"] = copied_lower - 10.0
    paste_button.click()
    pasted = project_gui.data_group_composite_config(group)
    assert pasted["axes"][0]["lower"] == copied_lower
    assert pasted["stale"] is True
    assignment = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "group_composite_axis_assignment_1"
    )
    assert assignment is not None and assignment.currentData() is True
    assignment.setCurrentIndex(assignment.findData(False))
    updated_axis = project_gui.data_group_composite_config(group)["axes"][1]
    assert updated_axis["mode"] == "step"
    assert updated_axis["fractional"] is False


def test_large_data_group_composite_defaults_manual_and_defers_refresh(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    monkeypatch.setattr(project_data, "REBIN_AUTO_MAX_CONTRIBUTIONS", 1)
    project_gui._COMPOSITE_DATA_CACHE.clear()

    first = DatasetEntry(
        "first", _tiny_mdhisto_data(1.0), kind="mdhisto", data_type="single_crystal_inelastic"
    )
    second = DatasetEntry(
        "second", _tiny_mdhisto_data(2.0), kind="mdhisto", data_type="single_crystal_inelastic"
    )
    group = DataGroup("Datagroup1", datasets=[first, second])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0))

    config = project_gui.data_group_composite_config(group)
    assert config["auto_rebin"] is False
    auto_check = explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_auto")
    assert auto_check is not None
    assert not auto_check.isChecked()
    status_label = explorer.details_widget.findChild(QtWidgets.QLabel, "group_composite_status")
    assert status_label is not None

    refresh_calls = []
    monkeypatch.setattr(explorer, "refresh_slice_viewer", lambda group: refresh_calls.append(group))
    checkbox = explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_enabled")
    assert checkbox is not None
    checkbox.setChecked(True)

    assert refresh_calls == []
    assert config["stale"] is True
    datasets, names = project_gui.slice_viewer_datasets(group, force_rebin=False)
    assert datasets == []
    assert names == []

    assert explorer.rebin_composite_now(group) is True
    assert config["stale"] is False
    assert refresh_calls == [group]
    datasets, names = project_gui.slice_viewer_datasets(group, force_rebin=False)
    assert names == ["Datagroup1 Composite"]
    np.testing.assert_allclose(datasets[0].signal[~datasets[0].mask], 1.5)


def test_rebin_all_composite_binnings_populates_viewer_entries_and_refreshes(
    monkeypatch,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    project_gui._COMPOSITE_DATA_CACHE.clear()
    group = DataGroup(
        "Datagroup1",
        datasets=[
            DatasetEntry("first", _tiny_mdhisto_data(1.0), kind="mdhisto"),
            DatasetEntry("second", _tiny_mdhisto_data(2.0), kind="mdhisto"),
        ],
    )
    fit = project_gui.data_group_composite_config(group)
    fit.update(enabled=True, minimum_coverage=0.0, stale=True)
    auxiliary_id = project_data.add_data_group_composite_binning(
        group, name="Overview"
    )
    auxiliary = project_data.data_group_composite_config_by_id(group, auxiliary_id)
    auxiliary.update(enabled=True, minimum_coverage=0.0, stale=True)
    auxiliary["axes"][0].update(mode="bins", num_bins=1)
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0))
    rebin_all_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "group_composite_rebin_all_now"
    )
    assert rebin_all_button is not None and not rebin_all_button.isHidden()
    refreshes = []
    monkeypatch.setattr(explorer, "_make_rebin_progress_callback", lambda *a, **k: None)
    monkeypatch.setattr(explorer, "_close_rebin_progress", lambda _progress: None)
    monkeypatch.setattr(explorer, "_sync_details", lambda: None)
    monkeypatch.setattr(explorer, "refresh_slice_viewer", refreshes.append)

    assert explorer.rebin_all_composite_binnings_now(group)
    assert fit["stale"] is False
    assert auxiliary["stale"] is False
    assert refreshes == [group]
    assert project_gui._peek_cached_composite_dataset_data(group) is not None
    assert project_gui._peek_cached_composite_dataset_data(
        group, config_override=auxiliary, binning_id=auxiliary_id
    ) is not None
    _datasets, names = project_gui.slice_viewer_datasets(group, force_rebin=False)
    assert names == ["Datagroup1 Composite", "Datagroup1 Composite · Overview"]


def test_crossing_rebin_size_threshold_disables_auto_until_user_reenables(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    config = dataset_rebin_config(dataset)
    config.update(enabled=True, auto_rebin=True, stale=False)
    config.pop("auto_rebin_user_override", None)
    monkeypatch.setattr(project_data, "REBIN_AUTO_MAX_CONTRIBUTIONS", 1)
    monkeypatch.setattr(explorer, "refresh_slice_viewer", lambda _group: None)

    explorer._after_dataset_rebin_changed(dataset, group)

    assert config["auto_rebin"] is False
    explorer._set_dataset_rebin_auto(dataset, group, True)
    assert config["auto_rebin"] is True
    assert config["auto_rebin_user_override"] is True

    explorer._after_dataset_rebin_changed(dataset, group)
    assert config["auto_rebin"] is True


def test_composite_controls_live_on_dataset_collections_not_workspace(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    first = DatasetEntry("first", _tiny_mdhisto_data(1.0), kind="mdhisto")
    second = DatasetEntry("second", _tiny_mdhisto_data(2.0), kind="mdhisto")
    subgroup = DatasetGroup("Group1", datasets=[second])
    group = DataGroup("Datagroup1", datasets=[first], subgroups=[subgroup])
    explorer = NfitProjectExplorer(NfitProject([group]))
    workspace_item = explorer.tree.topLevelItem(0)
    datasets_item = workspace_item.child(0)
    subgroup_item = next(
        datasets_item.child(index)
        for index in range(datasets_item.childCount())
        if datasets_item.child(index).text(0) == "Group1"
    )

    def flush_deletes():
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    explorer.tree.setCurrentItem(workspace_item)
    flush_deletes()
    assert explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_enabled") is None

    explorer.tree.setCurrentItem(datasets_item)
    flush_deletes()
    top_check = explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_enabled")
    assert top_check is not None
    top_check.setChecked(True)
    assert project_gui.data_group_composite_config(group)["enabled"] is True

    explorer.tree.setCurrentItem(subgroup_item)
    flush_deletes()
    nested_check = explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_enabled")
    assert nested_check is not None
    assert not nested_check.isChecked()
    nested_check.setChecked(True)
    nested_scope = project_gui._composite_scope(group, subgroup)
    assert project_gui.data_group_composite_config(nested_scope)["enabled"] is True


def test_nested_composite_replaces_only_its_dataset_group_descendants():
    root_dataset = DatasetEntry("root", _tiny_mdhisto_data(10.0), kind="mdhisto")
    first = DatasetEntry("first", _tiny_mdhisto_data(2.0), kind="mdhisto")
    second = DatasetEntry("second", _tiny_mdhisto_data(4.0), kind="mdhisto")
    subgroup = DatasetGroup("Group1", datasets=[first, second])
    group = DataGroup("Datagroup1", datasets=[root_dataset], subgroups=[subgroup])
    nested_scope = project_gui._composite_scope(group, subgroup)
    project_gui.data_group_composite_config(nested_scope)["enabled"] = True

    datasets, names = project_gui.slice_viewer_datasets(group)

    assert names == ["root", "Group1 Composite"]
    np.testing.assert_allclose(datasets[0].signal, [[10.0]])
    np.testing.assert_allclose(datasets[1].signal[~datasets[1].mask], 3.0)
    inputs, _bundles = project_gui.fit_dataset_inputs(group)
    assert [item.name for item in inputs] == ["root", "Group1 Composite"]


def test_mdhisto_rebin_applies_enabled_masks_before_binning():
    axis = MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum")
    data = MDHistoData(
        axes=(axis,),
        signal=np.array([1.0, 100.0]),
        errors=np.array([1.0, 1.0]),
        mask=np.array([False, False]),
        num_events=np.array([1.0, 1.0]),
    )
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    mask = create_mask(dataset)
    mask.parameters["H"] = [0.0, 1.0]
    mask.enabled = False
    config = dataset_rebin_config(dataset)
    config["enabled"] = True
    for axis in config["axes"]:
        axis["mode"] = "discrete"
    config["resolution_mode"] = "bins"
    config["axes"][0].update(
        {
            "lower": 0.0,
            "upper": 2.0,
            "auto_lower": False,
            "auto_upper": False,
            "num_bins": 1,
            "mode": "bins",
        }
    )

    viewed = dataset_for_slice_viewer(dataset)

    assert viewed is not None
    np.testing.assert_allclose(viewed.signal, [50.5])
    assert viewed.metadata["rebin"]["source_nfit_mask_count"] == 0

    mask.enabled = True
    viewed = dataset_for_slice_viewer(dataset)

    assert viewed is not None
    np.testing.assert_allclose(viewed.signal, [100.0])
    np.testing.assert_allclose(
        viewed.auxiliary_channels["coverage_fraction"].values,
        [0.5],
    )
    assert viewed.mask[0]
    assert viewed.metadata["rebin"]["source_nfit_mask_count"] == 1


def test_mdhisto_composite_masks_output_below_geometric_coverage_cutoff():
    axis = MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum")

    def source(value):
        return MDHistoData(
            axes=(axis,),
            signal=np.array([value, value]),
            errors=np.ones(2),
            mask=np.array([True, False]),
            num_events=np.ones(2),
        )

    group = DataGroup(
        "Datagroup1",
        datasets=[
            DatasetEntry("first", source(1.0), kind="mdhisto"),
            DatasetEntry("second", source(2.0), kind="mdhisto"),
        ],
    )
    config = project_gui.data_group_composite_config(group)
    config["enabled"] = True
    config["resolution_mode"] = "bins"
    config["axes"][0].update(
        {"lower": 0.0, "upper": 2.0, "num_bins": 1, "step_size": 2.0, "mode": "bins"}
    )
    config["axes"][0].update({"auto_lower": False, "auto_upper": False})

    strict = project_gui.composite_dataset_data(group)

    np.testing.assert_allclose(
        strict.auxiliary_channels["coverage_fraction"].values,
        [0.5],
    )
    assert strict.mask[0]
    assert project_gui._mdhisto_fit_bin_count(strict) == 0

    config["minimum_coverage"] = 0.5
    permissive = project_gui.composite_dataset_data(group)

    assert not permissive.mask[0]
    assert project_gui._mdhisto_fit_bin_count(permissive) == 1


def test_point_data_rebin_applies_enabled_masks_before_binning():
    data = PointData4D(
        H=[0.25, 0.75],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[0.0, 0.0],
        intensity=[1.0, 100.0],
        sigma=[1.0, 1.0],
    )
    dataset = DatasetEntry("points", data, kind="point")
    mask = create_mask(dataset)
    mask.parameters["H"] = [0.0, 0.5]
    mask.enabled = False
    config = dataset_rebin_config(dataset)
    config["enabled"] = True
    for axis in config["axes"]:
        axis.update(
            {
                "lower": 0.0,
                "upper": 1.0,
                "auto_lower": False,
                "auto_upper": False,
                "num_bins": 1,
                "mode": "bins",
            }
        )

    rebinned = project_gui.rebinned_dataset_data(dataset)

    assert isinstance(rebinned, MDHistoData)
    np.testing.assert_allclose(rebinned.signal, [[[[50.5]]]])

    mask.enabled = True
    rebinned = project_gui.rebinned_dataset_data(dataset)

    np.testing.assert_allclose(rebinned.signal, [[[[100.0]]]])


def test_point_data_auto_limits_center_zero_and_momentum_matrix_updates_labels(
    monkeypatch,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    data = PointData4D(
        H=[-0.2, 0.8],
        K=[0.0, 0.4],
        L=[-0.6, 0.2],
        E=[0.0, 1.0],
        intensity=[1.0, 2.0],
        sigma=[1.0, 1.0],
    )
    dataset = DatasetEntry("points", data, data_type="single_crystal_inelastic")
    group = DataGroup("Data", datasets=[dataset])
    config = dataset_rebin_config(dataset)

    assert all(axis["auto_lower"] and axis["auto_upper"] for axis in config["axes"])
    output = project_gui._rebin_point_data(data, config)
    for axis in output.axes:
        assert np.any(np.isclose(axis.centers, 0.0))

    project_gui._update_rebin_momentum_matrix(
        config["axes"],
        [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, 0.0]],
        data=data,
    )
    assert [axis["name"] for axis in config["axes"][:3]] == [
        "[H,H,0]",
        "[0,0,L]",
        "[K,-K,0]",
    ]
    with pytest.raises(ValueError, match="rank 2 rather than 3"):
        project_gui._update_rebin_momentum_matrix(
            config["axes"],
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, -2.0]],
            data=data,
        )

    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_dataset=dataset)
    lower = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "dataset_rebin_axis_lower_0"
    )
    matrix = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "dataset_rebin_momentum_matrix"
    )
    energy_axis_label = explorer.details_widget.findChild(
        QtWidgets.QLabel, "dataset_rebin_axis_variable_3"
    )
    assert lower is not None and lower.text() == "" and lower.placeholderText() == "auto"
    assert matrix is not None and matrix.toolTip()
    assert energy_axis_label is not None
    assignment = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "dataset_rebin_axis_mode_3"
    )
    assert assignment is not None and assignment.currentData() == "step"
    assert "Discrete" in assignment.toolTip()
    assignment.setCurrentIndex(assignment.findData("tolerance"))
    assert dataset_rebin_config(dataset)["axes"][3]["mode"] == "tolerance"
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_rebin_symmetry_toggle_preserves_expression_and_notation(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry(
        "scan",
        _tiny_mdhisto_data(1.0),
        kind="mdhisto",
        data_type="single_crystal_inelastic",
    )
    group = DataGroup("Data", datasets=[dataset])
    dataset_config = dataset_rebin_config(dataset)
    dataset_config["auto_rebin"] = False
    dataset_config["symmetry"].update(
        {"mode": "point_group", "expression": "m-3m"}
    )
    composite_config = project_gui.data_group_composite_config(group)
    composite_config["auto_rebin"] = False
    composite_config["symmetry"].update(
        {"mode": "operations", "expression": "x,y,z;-x,-y,-z"}
    )

    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._set_dataset_rebin_symmetry_enabled(dataset, group, False)
    assert dataset_config["symmetry"]["mode"] == "none"
    assert dataset_config["symmetry"]["expression"] == "m-3m"
    assert dataset_config["symmetry"]["last_mode"] == "point_group"
    explorer._set_dataset_rebin_symmetry_enabled(dataset, group, True)
    assert dataset_config["symmetry"]["mode"] == "point_group"
    assert dataset_config["symmetry"]["expression"] == "m-3m"

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0))
    symmetry_check = explorer.details_widget.findChild(
        QtWidgets.QCheckBox, "group_composite_symmetry_enabled"
    )
    assert symmetry_check is not None and symmetry_check.isChecked()
    explorer._set_group_composite_symmetry_enabled(group, False)
    assert composite_config["symmetry"]["mode"] == "none"
    assert composite_config["symmetry"]["expression"] == "x,y,z;-x,-y,-z"
    symmetry_mode = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "group_composite_symmetry_mode"
    )
    symmetry_expression = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "group_composite_symmetry_expression"
    )
    assert symmetry_mode.currentData() == "operations"
    assert symmetry_expression.text() == "x,y,z;-x,-y,-z"
    explorer._set_group_composite_symmetry_enabled(group, True)
    assert composite_config["symmetry"]["mode"] == "operations"
    assert composite_config["symmetry"]["expression"] == "x,y,z;-x,-y,-z"
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_import_dataset_paths_dispatches_by_data_type_and_round_trips(
    tmp_path, mpms_file, hb2a_file
):
    group = DataGroup("Datagroup1")

    mpms = import_dataset_paths(group, [mpms_file], data_type="magnetization")[0]
    assert mpms.data_type == "magnetization"
    assert isinstance(mpms.data, PointListData)
    assert mpms.metadata["importer"] == "mpms_dat"
    assert mpms.data.coordinate_names == ["Temperature", "Magnetic Field"]

    powder = import_dataset_paths(group, [hb2a_file], data_type="powder_elastic")[0]
    assert isinstance(powder.data, PointListData)

    nxs = import_dataset_paths(group, ["/fake/scan.nxs"], data_type="single_crystal_inelastic")[0]
    assert nxs.data is None  # lazy MDHisto placeholder
    assert nxs.data_type == "single_crystal_inelastic"

    project = NfitProject([group])
    path = tmp_path / "proj.nfit"
    save_project(project, path)
    reloaded = load_project(path)
    types = [ds.data_type for ds in reloaded.data_groups[0].datasets]
    assert types == ["magnetization", "powder_elastic", "single_crystal_inelastic"]
    assert reloaded.data_groups[0].datasets[0].metadata["importer"] == "mpms_dat"


def test_powder_ins_csv_import_round_trips_conditions_and_paired_channels(tmp_path):
    path = tmp_path / "constant_e.csv"
    path.write_text("0.4,1.0,0.1\n0.6,2.0,0.2\n", encoding="utf-8")
    options = {
        str(path): {
            "layout": "cut",
            "cut_type": "constant_energy",
            "fixed_value": 0.5,
            "temperature_K": 6.0,
            "source_representation": "cross_section",
            "source_unit": "1/meV/V",
            "fit_representation": "cross_section",
            "normalization_basis": "per_magnetic_ion",
            "normalization_label": "V",
            "kf_ki_state": "removed",
        }
    }
    group = DataGroup("Powder")
    dataset = import_dataset_paths(
        group,
        [path],
        data_type="powder_inelastic",
        importer_name="powder_ins_csv",
        importer_options=options,
    )[0]
    assert isinstance(dataset.data, MDHistoData)
    assert dataset.parameters["temperature"] == 6.0
    assert dataset.parameters["constant_energy_meV"] == 0.5
    assert dataset.metadata["import_options"]["normalization_label"] == "V"
    view = dataset_for_slice_viewer(dataset)
    assert view.channel_unit("scattering_cross_section") == "1/meV/V"
    assert "dynamic_susceptibility" in view.auxiliary_channels
    assert view.channel_unit("dynamic_susceptibility") == "arb. units"
    points = project_gui._point_data_from_mdhisto_view(view)
    np.testing.assert_allclose(points.H, [0.4, 0.6])
    np.testing.assert_allclose(points.E, [0.5, 0.5])
    assert points.metadata["coordinate_units"] == "1/angstrom"

    dataset.parameters["temperature"] = 8.0
    project_path = tmp_path / "powder.nfit"
    save_project(NfitProject([group]), project_path)
    restored = load_project(project_path).data_groups[0].datasets[0]
    assert restored.data is None
    restored_view = dataset_for_slice_viewer(restored)
    np.testing.assert_allclose(restored_view.signal, view.signal)
    assert restored.parameters["temperature"] == 8.0
    assert restored.parameters["constant_energy_meV"] == 0.5


def test_powder_ins_csv_dialog_controls_have_tooltips(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    path = tmp_path / "cut.csv"
    path.write_text("0.4,1.0,0.1\n", encoding="utf-8")
    explorer = NfitProjectExplorer(NfitProject([DataGroup("Powder")]))
    checked = []

    def reject_after_inspection(dialog):
        for name in (
            "powder_csv_observable",
            "powder_csv_units",
            "powder_csv_basis",
            "powder_csv_atom_label",
            "powder_csv_kinematic",
            "powder_csv_map_sigma",
            "powder_csv_cut_type_0",
            "powder_csv_temperature_0",
            "powder_csv_fixed_value_0",
            "powder_csv_observable_0",
            "powder_csv_units_0",
        ):
            widget = dialog.findChild(QtWidgets.QWidget, name)
            assert widget is not None
            assert widget.toolTip()
            checked.append(name)
        return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(QtWidgets.QDialog, "exec", reject_after_inspection)
    assert explorer._prompt_powder_ins_csv_options([path]) is False
    assert len(checked) == 11


def test_powder_ins_csv_dialog_allows_mixed_observables(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    intensity = tmp_path / "constant_e.csv"
    chipp = tmp_path / "constant_q.csv"
    for path in (intensity, chipp):
        path.write_text("0.4,1.0,0.1\n", encoding="utf-8")
    explorer = NfitProjectExplorer(NfitProject([DataGroup("Powder")]))

    def configure_and_accept(dialog):
        basis = dialog.findChild(QtWidgets.QComboBox, "powder_csv_basis")
        basis.setCurrentIndex(basis.findData("per_magnetic_ion"))
        dialog.findChild(QtWidgets.QLineEdit, "powder_csv_atom_label").setText("V")
        defaults = dialog.findChild(QtWidgets.QComboBox, "powder_csv_units")
        defaults.setCurrentIndex(defaults.findData("1/meV"))
        for row, fixed in ((0, "0.5"), (1, "0.6")):
            dialog.findChild(QtWidgets.QLineEdit, f"powder_csv_temperature_{row}").setText("6")
            dialog.findChild(QtWidgets.QLineEdit, f"powder_csv_fixed_value_{row}").setText(fixed)
        cut = dialog.findChild(QtWidgets.QComboBox, "powder_csv_cut_type_1")
        cut.setCurrentIndex(cut.findData("constant_q"))
        observable = dialog.findChild(QtWidgets.QComboBox, "powder_csv_observable_1")
        observable.setCurrentIndex(observable.findData("chi_double_prime"))
        units = dialog.findChild(QtWidgets.QComboBox, "powder_csv_units_1")
        units.setCurrentIndex(units.findData("mu_B^2/meV"))
        dialog.findChild(QtWidgets.QDialogButtonBox).accepted.emit()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", configure_and_accept)
    options = explorer._prompt_powder_ins_csv_options([intensity, chipp])
    assert options[str(intensity)]["source_unit"] == "1/meV/V"
    assert options[str(intensity)]["cut_type"] == "constant_energy"
    assert options[str(chipp)]["source_representation"] == "chi_double_prime"
    assert options[str(chipp)]["source_unit"] == "mu_B^2/meV/V"
    assert options[str(chipp)]["cut_type"] == "constant_q"


def test_set_dataset_data_type_reloads_and_resets(hb2a_file):
    group = DataGroup("Datagroup1")
    entry = import_dataset_paths(group, [hb2a_file], data_type="powder_elastic")[0]
    assert isinstance(entry.data, PointListData)

    # Switch to an MDHisto/nxs type: point data is dropped for lazy reload.
    set_dataset_data_type(entry, "single_crystal_inelastic")
    assert entry.data is None
    assert entry.data_type == "single_crystal_inelastic"

    # Switch back to a point-list type: importer reloads the columns.
    set_dataset_data_type(entry, "powder_elastic")
    assert isinstance(entry.data, PointListData)
    assert entry.data.coordinate_names == ["2theta"]


def test_project_explorer_data_type_dropdown_switches_type(monkeypatch, hb2a_file):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    import_dataset_paths(group, [hb2a_file], data_type="powder_elastic")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    combo = explorer.details_widget.findChild(QtWidgets.QComboBox, "dataset_data_type")
    assert combo is not None
    assert combo.currentData() == "powder_elastic"
    labels = [label for _name, label in available_data_types()]
    assert combo.count() == len(labels)

    # Switch to an MDHisto/nxs type: data resets for lazy reload.
    combo.setCurrentIndex(combo.findData("single_crystal_inelastic"))
    assert group.datasets[0].data_type == "single_crystal_inelastic"
    assert group.datasets[0].data is None


def test_nested_dataset_groups_share_masks_and_round_trip(tmp_path):
    from nfit.project_gui import (
        effective_dataset_masks,
        slice_viewer_datasets,
    )

    d1 = DatasetEntry("d1", _grid_mdhisto_data(), kind="mdhisto")
    d2 = DatasetEntry("d2", _grid_mdhisto_data(), kind="mdhisto")
    sub = DatasetGroup("Group1", datasets=[d2])
    group = DataGroup("Datagroup1", datasets=[d1], subgroups=[sub])
    project_gui.create_group_mask(sub, d2)
    sub.masks[0].parameters["H"] = [-10.0, 0.9]

    # Recursive iteration sees every dataset.
    assert group.dataset_names == ["d1", "d2"]
    assert [d.name for d in group.select()] == ["d1", "d2"]

    # Group mask applies only to the subgroup's descendants (d2), not d1.
    assert [m.name for m in effective_dataset_masks(group, d2)] == ["Mask1"]
    assert effective_dataset_masks(group, d1) == []
    data, names = slice_viewer_datasets(group)
    by_name = dict(zip(names, data, strict=False))
    assert by_name["d2"].metadata["nfit_mask_count"] > 0
    assert by_name["d1"].metadata["nfit_mask_count"] == 0

    # Round-trip nesting + shared masks through JSON (lazy placeholders).
    save_group = DataGroup("Datagroup2")
    import_dataset_paths(save_group, [tmp_path / "a.nxs"])[0]
    save_sub = DatasetGroup("SubB", enabled=False)
    save_group.subgroups.append(save_sub)
    p2 = DatasetEntry(
        "b",
        None,
        kind="nxs",
        metadata={"source_file": str(tmp_path / "b.nxs"), "import_status": "pending"},
    )
    save_sub.datasets.append(p2)
    project_gui.create_group_mask(save_sub, None)
    path = tmp_path / "proj.nfit"
    save_project(NfitProject([save_group]), path)
    reloaded = load_project(path).data_groups[0]
    assert reloaded.dataset_names == ["a", "b"]
    assert reloaded.subgroups[0].name == "SubB"
    assert reloaded.subgroups[0].enabled is False
    assert reloaded.subgroups[0].masks[0].name == "Mask1"


def test_disabled_dataset_group_is_omitted_from_fit_inputs_without_changing_children():
    direct = DatasetEntry("direct", _grid_mdhisto_data(), kind="mdhisto")
    nested = DatasetEntry("nested", _grid_mdhisto_data(), kind="mdhisto")
    deeper = DatasetEntry("deeper", _grid_mdhisto_data(), kind="mdhisto")
    subgroup = DatasetGroup(
        "Group1",
        datasets=[nested],
        subgroups=[DatasetGroup("Group2", datasets=[deeper])],
        enabled=False,
    )
    group = DataGroup("Workspace1", datasets=[direct], subgroups=[subgroup])

    inputs, _bundles = project_gui.fit_dataset_inputs(group)

    assert [item.name for item in inputs] == ["direct"]
    assert nested.enabled is True
    assert deeper.enabled is True

    subgroup.enabled = True
    inputs, _bundles = project_gui.fit_dataset_inputs(group)
    assert [item.name for item in inputs] == ["direct", "nested", "deeper"]


def test_multi_select_move_and_import_into_subgroup(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    d1 = DatasetEntry("d1", _grid_mdhisto_data(), kind="mdhisto")
    d2 = DatasetEntry("d2", _grid_mdhisto_data(), kind="mdhisto")
    d3 = DatasetEntry("d3", _grid_mdhisto_data(), kind="mdhisto")
    sub = DatasetGroup("Group1")
    group = DataGroup("Datagroup1", datasets=[d1, d2, d3], subgroups=[sub])
    explorer = NfitProjectExplorer(NfitProject([group]))

    def dataset_item(name):
        datasets_item = explorer.tree.topLevelItem(0).child(0)
        return next(
            datasets_item.child(i)
            for i in range(datasets_item.childCount())
            if datasets_item.child(i).text(0) == name
        )

    # Multi-select d1 + d3 and drop them into the subgroup.
    explorer.tree.setCurrentItem(dataset_item("d1"))
    dataset_item("d1").setSelected(True)
    dataset_item("d3").setSelected(True)
    assert explorer.move_or_copy_selected_to_item(dataset_item("Group1"), copy_item=False)
    assert [d.name for d in group.datasets] == ["d2"]
    assert [d.name for d in sub.datasets] == ["d1", "d3"]

    # Import button is available on a subgroup, and imports land inside it.
    subgroup_item = dataset_item("Group1")
    explorer.tree.setCurrentItem(subgroup_item)
    explorer._sync_details()
    assert not explorer.import_dataset_button.isHidden()
    resolved_group, into = explorer._selected_import_target()
    assert resolved_group is group and into is sub
    explorer.import_dataset_paths(
        group, ["/fake/new.nxs"], data_type="single_crystal_inelastic", into=sub
    )
    assert [d.name for d in sub.datasets] == ["d1", "d3", "new"]


def test_project_explorer_drag_reorders_groups_datasets_and_masks(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    d1 = DatasetEntry("d1", _grid_mdhisto_data(), kind="mdhisto")
    d2 = DatasetEntry("d2", _grid_mdhisto_data(), kind="mdhisto")
    d3 = DatasetEntry("d3", _grid_mdhisto_data(), kind="mdhisto")
    create_mask(d1, "Mask1")
    create_mask(d1, "Mask2")
    create_mask(d1, "Mask3")
    group1 = DataGroup("Datagroup1", datasets=[d1, d2, d3])
    group2 = DataGroup("Datagroup2")
    group3 = DataGroup("Datagroup3")
    explorer = NfitProjectExplorer(NfitProject([group1, group2, group3]))
    below = QtWidgets.QAbstractItemView.DropIndicatorPosition.BelowItem

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    explorer.tree.topLevelItem(0).setSelected(True)
    assert explorer.move_or_copy_selected_to_item(
        explorer.tree.topLevelItem(2), copy_item=False, drop_position=below
    )
    assert [group.name for group in explorer.project.data_groups] == [
        "Datagroup2",
        "Datagroup3",
        "Datagroup1",
    ]

    datasets_item = explorer.tree.topLevelItem(2).child(0)
    dataset_items = {
        datasets_item.child(i).text(0): datasets_item.child(i)
        for i in range(datasets_item.childCount())
    }
    explorer.tree.clearSelection()
    explorer.tree.setCurrentItem(dataset_items["d1"])
    dataset_items["d1"].setSelected(True)
    dataset_items["d2"].setSelected(True)
    assert explorer.move_or_copy_selected_to_item(
        dataset_items["d3"], copy_item=False, drop_position=below
    )
    assert [dataset.name for dataset in group1.datasets] == ["d3", "d1", "d2"]

    datasets_item = explorer.tree.topLevelItem(2).child(0)
    masks_item = next(
        datasets_item.child(i).child(0)
        for i in range(datasets_item.childCount())
        if datasets_item.child(i).text(0) == "d1"
    )
    mask_items = {
        masks_item.child(i).text(0): masks_item.child(i) for i in range(masks_item.childCount())
    }
    explorer.tree.clearSelection()
    explorer.tree.setCurrentItem(mask_items["Mask1"])
    mask_items["Mask1"].setSelected(True)
    mask_items["Mask2"].setSelected(True)
    assert explorer.move_or_copy_selected_to_item(
        mask_items["Mask3"], copy_item=False, drop_position=below
    )
    assert [mask.name for mask in d1.masks] == ["Mask3", "Mask1", "Mask2"]


def test_project_explorer_nested_group_bulk_edit_and_tree(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    d1 = DatasetEntry("d1", _grid_mdhisto_data(), kind="mdhisto")
    d2 = DatasetEntry("d2", _grid_mdhisto_data(), kind="mdhisto")
    sub = DatasetGroup("Group1", datasets=[d2])
    group = DataGroup("Datagroup1", datasets=[d1], subgroups=[sub])
    explorer = NfitProjectExplorer(NfitProject([group]))

    datasets_item = explorer.tree.topLevelItem(0).child(0)
    assert [datasets_item.child(i).text(0) for i in range(datasets_item.childCount())] == [
        "d1",
        "Group1",
    ]
    subgroup_item = next(
        datasets_item.child(index)
        for index in range(datasets_item.childCount())
        if datasets_item.child(index).text(0) == "Group1"
    )
    assert [subgroup_item.child(i).text(0) for i in range(subgroup_item.childCount())] == [
        "d2",
    ]

    # Bulk-set scale on the subgroup overwrites all descendants.
    explorer.tree.setCurrentItem(subgroup_item)
    explorer._set_group_bulk_value("scale_factor", "5")
    assert [d.scale_factor for d in sub.iter_datasets()] == [5.0]

    # The top-level workspace is organizational: fit weight and scale stay neutral
    # there, so the bulk editor is hidden and helper edits are ignored.
    top_item = explorer.tree.topLevelItem(0)
    explorer.tree.setCurrentItem(top_item)
    explorer._sync_details()
    assert explorer.group_bulk_widget.isHidden()
    explorer._set_group_bulk_value("fit_weight", "3")
    assert [d.fit_weight for d in group.iter_datasets()] == [1.0, 1.0]

    # Nested dataset groups still support bulk dataset fit controls.
    d1.fit_weight = 2.0
    explorer.tree.setCurrentItem(subgroup_item)
    explorer._sync_details()
    assert not explorer.group_bulk_widget.isHidden()
    assert explorer.group_fit_weight_edit.text() == "1"
    assert not explorer.enabled_check.isHidden()
    assert explorer.enabled_check.isChecked()

    explorer.enabled_check.setChecked(False)

    assert sub.enabled is False
    assert d2.enabled is True
    assert explorer.tree.currentItem().text(0) == "Group1"
    assert not explorer.enabled_check.isChecked()


def test_nested_group_can_share_one_fitted_dataset_scale(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    first = DatasetEntry("first", _grid_mdhisto_data(), scale_factor=2.0)
    second = DatasetEntry("second", _grid_mdhisto_data(), scale_factor=3.0)
    subgroup = DatasetGroup("same_run", datasets=[first, second])
    group = DataGroup("workspace", subgroups=[subgroup])
    explorer = NfitProjectExplorer(NfitProject([group]))
    datasets_item = explorer.tree.topLevelItem(0).child(0)
    subgroup_item = next(
        datasets_item.child(index)
        for index in range(datasets_item.childCount())
        if datasets_item.child(index).text(0) == "same_run"
    )
    explorer.tree.setCurrentItem(subgroup_item)
    monkeypatch.setattr(explorer, "_record_data_group_state_change", lambda _group: False)

    shared = explorer.window.findChild(
        QtWidgets.QCheckBox, "group_scale_factor_vary"
    )
    assert shared is not None
    assert "one calibration scale" in shared.toolTip()
    shared.setChecked(True)
    assert [dataset.scale_factor for dataset in subgroup.datasets] == [2.0, 2.0]
    assert all(dataset.scale_factor_vary for dataset in subgroup.datasets)
    assert {
        dataset.scale_factor_group for dataset in subgroup.datasets
    } == {"same_run"}

    shared.setChecked(False)
    assert all(dataset.scale_factor_vary for dataset in subgroup.datasets)
    assert all(
        dataset.scale_factor_group is None for dataset in subgroup.datasets
    )


def test_dataset_scale_factor_scales_viewed_data_and_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    scale_spin = explorer.window.findChild(QtWidgets.QDoubleSpinBox, "dataset_scale_factor")
    scale_fit_check = explorer.window.findChild(QtWidgets.QCheckBox, "dataset_scale_factor_vary")
    assert scale_spin is not None
    assert scale_fit_check is not None
    assert scale_spin.value() == 1.0
    assert not scale_fit_check.isChecked()
    scale_spin.setValue(3.0)
    refresh_calls = []
    monkeypatch.setattr(explorer, "refresh_slice_viewer", refresh_calls.append)
    scale_fit_check.setChecked(True)
    assert dataset.scale_factor == 3.0
    assert dataset.scale_factor_vary is True
    assert refresh_calls == []
    with monkeypatch.context() as scale_guard:
        scale_guard.setattr(
            project_gui,
            "_apply_dataset_scale",
            lambda *_args, **_kwargs: pytest.fail("fit summary reapplied dataset scale"),
        )
        assert project_gui._dataset_fit_summary_lines(dataset, group=group)

    refresh_tree_calls = []
    monkeypatch.setattr(explorer, "_record_data_group_state_change", lambda _group: True)
    monkeypatch.setattr(
        explorer, "_refresh_tree", lambda **kwargs: refresh_tree_calls.append(kwargs)
    )
    scale_fit_check.setChecked(False)
    assert refresh_tree_calls == [
        {
            "select_group": group,
            "select_dataset": dataset,
            "refresh_viewers": False,
        }
    ]

    viewed = dataset_for_slice_viewer(dataset)
    np.testing.assert_allclose(viewed.signal, np.asarray(data.signal, dtype=float) * 3.0)
    np.testing.assert_allclose(viewed.errors, np.asarray(data.errors, dtype=float) * 3.0)

    # Round-trip through JSON using a lazy placeholder dataset.
    save_group = DataGroup("Datagroup2")
    placeholder = import_dataset_paths(save_group, [tmp_path / "scan.nxs"])[0]
    placeholder.scale_factor = 3.0
    placeholder.scale_factor_vary = True
    placeholder.scale_factor_group = "same_run"
    path = tmp_path / "proj.nfit"
    save_project(NfitProject([save_group]), path)
    payload = read_project_manifest(path)["data_groups"][0]["datasets"][0]
    assert payload["scale_factor"] == 3.0
    assert payload["scale_factor_vary"] is True
    assert payload["scale_factor_group"] == "same_run"
    loaded = load_project(path).data_groups[0].datasets[0]
    assert loaded.scale_factor == 3.0
    assert loaded.scale_factor_vary is True
    assert loaded.scale_factor_group == "same_run"


def test_point_list_scale_and_susceptibility_transforms(mpms_file):
    from nfit.project_gui import point_list_config, prepared_point_list_data

    group = DataGroup("Datagroup1")
    dataset = import_dataset_paths(group, [mpms_file], data_type="magnetization")[0]
    raw = dataset.data
    config = point_list_config(dataset)
    config["scale"] = {"factor": 2.0, "channel": "Moment", "units": "emu/mol"}
    config["susceptibility"] = {"enabled": True, "field": "Magnetic Field", "moment": "Moment"}

    prepared = prepared_point_list_data(dataset)
    assert prepared_point_list_data(dataset) is prepared

    # Scale multiplies value and error and relabels units.
    np.testing.assert_allclose(
        prepared.channel_values("Moment"), raw.channel_values("Moment") * 2.0
    )
    np.testing.assert_allclose(
        prepared.channel_errors("Moment"), raw.channel_errors("Moment") * 2.0
    )
    assert prepared.unit(prepared.channel("Moment")["value"]) == "emu/mol"

    # Susceptibility = scaled moment / field.
    assert "Susceptibility" in prepared.channel_labels
    expected = raw.channel_values("Moment") * 2.0 / raw.column("Magnetic Field")
    np.testing.assert_allclose(prepared.channel_values("Susceptibility"), expected)
    assert prepared.unit(prepared.channel("Susceptibility")["value"]) == "cm^3/mol"

    config["scale"]["factor"] = 3.0
    updated = prepared_point_list_data(dataset)
    assert updated is not prepared
    np.testing.assert_allclose(updated.channel_values("Moment"), raw.channel_values("Moment") * 3.0)


def test_mpms_import_seeds_sample_normalization_metadata(mpms_file):
    group = DataGroup("Datagroup1")
    dataset = import_dataset_paths(group, [mpms_file], data_type="magnetization")[0]
    assert dataset.parameters["sample_mass_mg"] == pytest.approx(9.73)
    assert dataset.parameters["molar_mass_g_mol"] == pytest.approx(172.8)


def test_absolute_mpms_susceptibility_is_molar_and_becomes_fit_channel(mpms_file):
    from nfit.project_gui import fit_data_bundle, point_list_config, prepared_point_list_data

    group = DataGroup("Datagroup1")
    dataset = import_dataset_paths(group, [mpms_file], data_type="magnetization")[0]
    dataset.parameters.update(
        {"absolute_units": True, "sample_mass_mg": 10.0, "molar_mass_g_mol": 200.0}
    )
    config = point_list_config(dataset)
    config["susceptibility"].update(
        {"enabled": True, "field": "Magnetic Field", "moment": "Moment", "output_unit": "cm^3/mol"}
    )
    prepared = prepared_point_list_data(dataset)
    moles = (10.0 / 1000.0) / 200.0
    expected = dataset.data.column("Moment") / dataset.data.column("Magnetic Field") / moles
    np.testing.assert_allclose(prepared.channel_values("Susceptibility"), expected)
    assert prepared.unit(prepared.channel("Susceptibility")["value"]) == "cm^3/mol"
    assert prepared.channel_quantity_type("Susceptibility") == "bulk_susceptibility"
    assert "Inverse susceptibility" in prepared.channel_labels
    np.testing.assert_allclose(prepared.channel_values("Inverse susceptibility"), 1.0 / expected)
    np.testing.assert_allclose(
        prepared.channel_errors("Inverse susceptibility"),
        prepared.channel_errors("Susceptibility") / expected**2,
    )
    assert prepared.unit(prepared.channel("Inverse susceptibility")["value"]) == "mol/cm^3"
    assert prepared.channel_quantity_type("Inverse susceptibility") == "inverse_bulk_susceptibility"

    bundle = fit_data_bundle(group, dataset)
    assert bundle.points.metadata["fit_channel"] == "Susceptibility"
    assert bundle.points.metadata["quantity_type"] == "bulk_susceptibility"
    assert bundle.points.metadata["unit"] == "cm^3/mol"


def test_absolute_mpms_susceptibility_can_use_si_units(mpms_file):
    from nfit.project_gui import point_list_config, prepared_point_list_data

    group = DataGroup("Datagroup1")
    dataset = import_dataset_paths(group, [mpms_file], data_type="magnetization")[0]
    dataset.parameters.update(
        {"absolute_units": True, "sample_mass_mg": 10.0, "molar_mass_g_mol": 200.0}
    )
    config = point_list_config(dataset)
    config["susceptibility"].update(
        {"enabled": True, "field": "Magnetic Field", "moment": "Moment", "output_unit": "m^3/mol"}
    )
    prepared = prepared_point_list_data(dataset)
    moles = (10.0 / 1000.0) / 200.0
    expected_cgs = dataset.data.column("Moment") / dataset.data.column("Magnetic Field") / moles
    np.testing.assert_allclose(
        prepared.channel_values("Susceptibility"), expected_cgs * 4.0 * np.pi * 1.0e-6
    )
    assert prepared.unit(prepared.channel("Susceptibility")["value"]) == "m^3/mol"


def test_heat_capacity_transform_and_unit_selector_tooltips(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import point_list_config, prepared_point_list_data

    data = PointListData(
        {"Sample Temp": [2.0, 3.0], "Samp HC": [1.0, 2.0], "Samp HC Err": [0.1, 0.2]},
        units={"Sample Temp": "K", "Samp HC": "J/(mol K)", "Samp HC Err": "J/(mol K)"},
        coordinate_names=["Sample Temp"],
        channels=[
            {
                "label": "Sample heat capacity",
                "value": "Samp HC",
                "error": "Samp HC Err",
                "quantity_type": "heat_capacity",
                "unit": "J/(mol K)",
            }
        ],
    )
    dataset = DatasetEntry("HC", data, data_type="heat_capacity")
    dataset.parameters["absolute_units"] = True
    config = point_list_config(dataset)
    config["heat_capacity"]["source_unit"] = "J/(mol K)"
    prepared = prepared_point_list_data(dataset)
    np.testing.assert_allclose(prepared.channel_values("Heat capacity"), [1000.0, 2000.0])
    np.testing.assert_allclose(prepared.channel_values("C/T"), [500.0, 2000.0 / 3.0])
    np.testing.assert_allclose(prepared.column("Temperature squared"), [4.0, 9.0])

    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))
    unit_combo = explorer.details_widget.findChild(QtWidgets.QComboBox, "heat_capacity_source_unit")
    fit_combo = explorer.details_widget.findChild(QtWidgets.QComboBox, "heat_capacity_fit_channel")
    atoms_edit = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "heat_capacity_atoms_per_formula_unit"
    )
    assert unit_combo is not None and unit_combo.count() == 10 and unit_combo.toolTip()
    assert fit_combo is not None and fit_combo.toolTip()
    assert atoms_edit is not None and atoms_edit.toolTip()


def test_absolute_mpms_moment_can_be_normalized_per_formula_unit(mpms_file):
    from nfit.project_gui import fit_data_bundle, point_list_config, prepared_point_list_data
    from nfit.sum_rules import EMU_PER_MOL_PER_MU_B

    group = DataGroup("Datagroup1")
    dataset = import_dataset_paths(group, [mpms_file], data_type="magnetization")[0]
    dataset.parameters.update(
        {
            "absolute_units": True,
            "sample_mass_mg": 10.0,
            "molar_mass_g_mol": 200.0,
            "magnetization_output_unit": "mu_B/f.u.",
        }
    )
    config = point_list_config(dataset)
    config["susceptibility"].update({"moment_unit": "emu", "field_unit": "Oe"})

    prepared = prepared_point_list_data(dataset)
    moles = (10.0 / 1000.0) / 200.0
    expected = dataset.data.column("Moment") / (moles * EMU_PER_MOL_PER_MU_B)
    np.testing.assert_allclose(prepared.channel_values("Moment"), expected)
    assert prepared.unit(prepared.channel("Moment")["value"]) == "mu_B/f.u."

    bundle = fit_data_bundle(group, dataset)
    assert bundle.points.metadata["fit_channel"] == "Moment"
    assert bundle.points.metadata["unit"] == "mu_B/f.u."


def test_magnetization_bundle_maps_temperature_and_field(mpms_file):
    """An imported MPMS dataset produces per-point T and field fit points."""
    from nfit.project_gui import fit_data_bundle

    group = DataGroup("Datagroup1")
    dataset = import_dataset_paths(group, [mpms_file], data_type="magnetization")[0]
    bundle = fit_data_bundle(group, dataset)
    assert bundle is not None
    points = bundle.points
    assert points.metadata["data_type"] == "magnetization"
    # Momentum coordinates are all zero (magnetization carries no Q).
    assert np.all(points.H == 0) and np.all(points.E == 0)
    # Per-point temperature and field are populated.
    assert isinstance(points.temperature, np.ndarray)
    assert points.temperature.size == points.size
    assert points.magnetic_field is not None and points.magnetic_field.ndim == 2


def test_magnetization_absolute_units_box(monkeypatch, mpms_file):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    dataset = import_dataset_paths(group, [mpms_file], data_type="magnetization")[0]
    explorer = NfitProjectExplorer(NfitProject([group]))
    # Select the dataset to build its details pane.
    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(dataset_item)

    enable = explorer.window.findChild(QtWidgets.QCheckBox, "magnetization_absolute_enabled")
    assert enable is not None and enable.toolTip().strip()
    mass_edit = explorer.window.findChild(QtWidgets.QLineEdit, "magnetization_sample_mass_mg")
    molar_edit = explorer.window.findChild(QtWidgets.QLineEdit, "magnetization_molar_mass_g_mol")
    output_combo = explorer.window.findChild(QtWidgets.QComboBox, "magnetization_output_unit")
    moment_unit_combo = explorer.window.findChild(
        QtWidgets.QComboBox, "point_list_moment_input_unit"
    )
    field_unit_combo = explorer.window.findChild(QtWidgets.QComboBox, "point_list_field_input_unit")
    assert mass_edit is not None and molar_edit is not None and output_combo is not None
    assert moment_unit_combo is not None and moment_unit_combo.toolTip().strip()
    assert field_unit_combo is not None and field_unit_combo.toolTip().strip()

    enable.setChecked(True)
    mass_edit.setText("12.5")
    mass_edit.editingFinished.emit()
    molar_edit.setText("250.0")
    molar_edit.editingFinished.emit()
    output_combo.setCurrentIndex(output_combo.findData("mu_B/f.u."))
    assert dataset.parameters["absolute_units"] is True
    assert dataset.parameters["sample_mass_mg"] == 12.5
    assert dataset.parameters["molar_mass_g_mol"] == 250.0
    assert dataset.parameters["magnetization_output_unit"] == "mu_B/f.u."


def test_sample_environment_panel_is_hidden_for_magnetization(monkeypatch, mpms_file):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    import_dataset_paths(group, [mpms_file], data_type="magnetization")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))
    titles = [box.title() for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)]
    assert "Conditions" not in titles


def test_powder_wavelength_to_q_and_point_rebin(hb2a_file):
    from nfit.project_gui import (
        dataset_for_slice_viewer,
        dataset_rebin_config,
        point_list_config,
        prepared_point_list_data,
    )

    group = DataGroup("Datagroup1")
    dataset = import_dataset_paths(group, [hb2a_file], data_type="powder_elastic")[0]
    config = point_list_config(dataset)
    config["wavelength"] = {"value": 2.41, "two_theta": "2theta"}
    dataset.parameters["temperature"] = 10.0

    prepared = prepared_point_list_data(dataset)
    # Powder is 1D: q is the single coordinate; d and 2theta remain columns.
    assert prepared.coordinate_names == ["q"]
    assert "d" in prepared.column_names
    two_theta = dataset.data.column("2theta")
    theta = np.deg2rad(two_theta) / 2.0
    expected_q = 4.0 * np.pi * np.sin(theta) / 2.41
    np.testing.assert_allclose(prepared.column("q"), expected_q)
    # d-spacing in angstroms equals 2*pi/q and lambda/(2 sin theta).
    np.testing.assert_allclose(prepared.column("d"), 2.41 / (2.0 * np.sin(theta)))
    np.testing.assert_allclose(prepared.column("d"), 2.0 * np.pi / prepared.column("q"))
    assert prepared.unit("q") == "Å⁻¹"
    assert prepared.unit("d") == "Å"

    rebin = dataset_rebin_config(dataset)
    assert rebin["axes"][0]["name"] == "q"
    rebin["enabled"] = True
    rebin["axes"][0]["mode"] = "bins"
    rebin["axes"][0]["num_bins"] = 100
    viewed = dataset_for_slice_viewer(dataset)
    assert isinstance(viewed, PointListData)
    assert viewed.size < dataset.data.size
    assert "q" in viewed.coordinate_names

    bundle = project_gui.fit_data_bundle(group, dataset)
    assert bundle is not None
    np.testing.assert_allclose(bundle.points.H, viewed.column("q"))
    np.testing.assert_allclose(bundle.points.E, 0.0)
    assert bundle.points.metadata["powder_q_modulus_axis"] is True
    assert bundle.points.metadata["coordinate_units"] == "1/angstrom"


def test_point_list_variables_panel_edits_config(monkeypatch, mpms_file):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import point_list_config, prepared_point_list_data

    group = DataGroup("Datagroup1")
    import_dataset_paths(group, [mpms_file], data_type="magnetization")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))
    dataset = group.datasets[0]

    panel = next(
        box
        for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)
        if box.title() == "Variables and Channels"
    )
    assert panel is not None
    assert (
        explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "point_list_scale_factor")
        is None
    )
    susc_check = explorer.details_widget.findChild(
        QtWidgets.QCheckBox, "point_list_susceptibility_enabled"
    )
    susc_check.setChecked(True)

    config = point_list_config(dataset)
    assert config["susceptibility"]["enabled"] is True
    assert "Susceptibility" in prepared_point_list_data(dataset).channel_labels


def test_point_list_dataset_opens_in_data_viewer_as_1d(monkeypatch, mpms_file):
    from types import SimpleNamespace

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    group = DataGroup("Datagroup1")
    import_dataset_paths(group, [mpms_file], data_type="magnetization")
    datasets, names = project_gui.slice_viewer_datasets(group)
    assert isinstance(datasets[0], PointListData)

    viewer = QtMDHistoSliceViewer(datasets, dataset_names=names)
    viewer.show()
    viewer.update_plot(preserve_view=False)

    assert viewer._is_effective_1d()
    assert viewer.model.is_point_list
    # x-axis selector leads with the coordinates, then offers other non-channel
    # columns; the channel combo offers the channels.
    x_items = [viewer.x_combo.itemText(i) for i in range(viewer.x_combo.count())]
    assert x_items[:2] == ["Temperature", "Magnetic Field"]
    assert "Moment" not in x_items  # channels are not x-axis options
    assert "Moment" in [
        viewer.channel_combo.itemText(i) for i in range(viewer.channel_combo.count())
    ]

    viewer._set_channel("Moment")
    viewer._set_display_dim("x", x_items.index("Magnetic Field"))
    view = viewer.model.slice_arrays()
    assert view["signal"].size == datasets[0].size
    assert viewer.model._channel_label() == "Moment (emu)"
    assert viewer.model._axis_label(0) == "Magnetic Field (Oe)"

    # The channel has an error column, so error bars are drawn for point data.
    assert viewer.show_errorbars
    assert len(viewer.ax_image.containers) == 1

    event = SimpleNamespace(
        inaxes=viewer.ax_image,
        xdata=float(view["x_centers"][0]),
        ydata=float(view["signal"][0]),
    )
    viewer._on_motion(event)

    assert viewer.cursor_hkle_label.isHidden()
    assert viewer.cursor_q_label.isHidden()
    assert viewer.cursor_intensity_label.text().startswith("Signal = ")


def test_data_viewer_channel_change_resets_limits(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data(1.0)
    viewer = QtMDHistoSliceViewer(data)
    calls = []
    monkeypatch.setattr(viewer, "update_plot", lambda **kwargs: calls.append(kwargs))
    viewer._set_channel("signal")
    assert calls[-1] == {"preserve_view": False}


def test_energy_q_range_mask_defaults_are_neutral():
    parameters = default_mask_parameters("energy_q_range")
    assert parameters["energy"] == [0.0, 0.0]
    assert parameters["q_modulus"] == [0.0, 0.0]


def test_all_registered_mask_defaults_mask_nothing_for_histogram_and_points():
    histogram = DatasetEntry("grid", _grid_mdhisto_data())
    points = PointData4D(
        H=[0.0, 1.0],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[0.0, 5.0],
        intensity=[1.0, 2.0],
        sigma=[1.0, 1.0],
    )
    point_dataset = DatasetEntry("points", points)
    for mask_type in project_gui.available_mask_types():
        histogram.masks = []
        point_dataset.masks = []
        create_mask(histogram, type=mask_type)
        create_mask(point_dataset, type=mask_type)
        viewed = dataset_for_slice_viewer(histogram)
        assert viewed.metadata["nfit_mask_count"] == 0, mask_type
        point_mask = project_gui._nfit_mask_for_point_data(point_dataset, points)
        assert not np.any(point_mask), mask_type


def test_coordinate_range_axis_vectors_fall_back_when_projection_dimension_differs():
    hh_axis = MDHistoAxis("[H,H,0]", np.array([0.0, 0.5, 1.0]), "rlu", "momentum")
    l_axis = MDHistoAxis("[0,0,L]", np.array([0.0, 0.5, 1.0]), "rlu", "momentum")
    data = MDHistoData(
        axes=(hh_axis, l_axis),
        signal=np.zeros((2, 2)),
        errors=np.ones((2, 2)),
        mask=np.zeros((2, 2), dtype=bool),
        num_events=np.ones((2, 2)),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset)

    assert mask.parameters["axis_0"] == [1.0, 0.0]
    assert mask.parameters["axis_1"] == [0.0, 1.0]


def test_coordinate_range_mask_defaults_match_four_dimensional_rebin_axes():
    axes = (
        MDHistoAxis("DeltaE", np.array([0.0, 1.0, 2.0]), "meV", "energy"),
        MDHistoAxis("[H,-H,0]", np.array([-1.0, 0.0, 1.0]), "rlu", "momentum"),
        MDHistoAxis("[0,0,L]", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),
        MDHistoAxis("[H,H,0]", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),
    )
    signal = np.zeros((2, 2, 2, 2))
    dataset = DatasetEntry(
        "scan",
        MDHistoData(
            axes=axes,
            signal=signal,
            errors=np.ones_like(signal),
            mask=np.zeros_like(signal, dtype=bool),
            num_events=np.ones_like(signal),
            metadata={},
        ),
    )

    mask = create_mask(dataset)

    assert [mask.parameters[f"axis_{index}"] for index in range(4)] == [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


def test_rebin_axis_vector_projects_new_coordinate():
    h_axis = MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum")
    e_axis = MDHistoAxis("E", np.array([0.0, 10.0, 20.0]), "meV", "energy")
    signal = np.array([[1.0, 2.0], [3.0, 4.0]])
    data = MDHistoData(
        axes=(h_axis, e_axis),
        signal=signal,
        errors=np.ones((2, 2)),
        mask=np.zeros((2, 2), dtype=bool),
        num_events=np.ones((2, 2)),
        metadata={
            "signal_semantics": "bin_integral",
            "signal_semantics_source": "user_selected",
        },
    )
    dataset = DatasetEntry("scan", data)
    config = dataset_rebin_config(dataset)
    config["enabled"] = True
    for axis in config["axes"]:
        axis["mode"] = "bins"
    # Swap which physical coordinate maps to each output axis.
    config["axes"][0].update({"vector": [0.0, 1.0], "lower": 5.0, "upper": 15.0, "num_bins": 2})
    config["axes"][1].update({"vector": [1.0, 0.0], "lower": 0.5, "upper": 1.5, "num_bins": 2})
    for axis in config["axes"]:
        axis.update({"auto_lower": False, "auto_upper": False})

    rebinned = project_gui.rebinned_dataset_data(dataset)

    assert rebinned.shape == (2, 2)
    np.testing.assert_allclose(rebinned.signal, signal.T)
    assert rebinned.metadata["rebin"]["vectors"] == [[0.0, 1.0], [1.0, 0.0]]
    assert rebinned.metadata["signal_semantics"] == "density"
    assert rebinned.metadata["signal_semantics_source"] == "nfit_normalized_rebin"


def test_rebin_defaults_follow_mdhisto_axis_coordinate_vectors():
    axes = (
        MDHistoAxis("DeltaE", np.array([0.0, 1.0, 2.0]), "meV", "energy"),
        MDHistoAxis("[H,-H,0]", np.array([-1.0, 0.0, 1.0]), "rlu", "momentum"),
        MDHistoAxis("[0,0,L]", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),
        MDHistoAxis("[H,H,0]", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),
    )
    signal = np.arange(16, dtype=float).reshape((2, 2, 2, 2))
    data = MDHistoData(
        axes=axes,
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)

    config = dataset_rebin_config(dataset)

    assert [axis["vector"] for axis in config["axes"]] == [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]

    config["enabled"] = True
    for axis in config["axes"]:
        axis["mode"] = "discrete"
    rebinned = project_gui.rebinned_dataset_data(dataset)

    for axis, step in zip(
        rebinned.axes, rebinned.metadata["rebin"]["step_size"], strict=True
    ):
        np.testing.assert_allclose(axis.centers / step, np.rint(axis.centers / step))
    np.testing.assert_allclose(
        np.sort(rebinned.signal[np.isfinite(rebinned.signal)]),
        np.sort(signal.ravel()),
    )
    assert rebinned.metadata["rebin"]["vectors"] == [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


def test_rebin_basis_survives_lazy_dataset_reload():
    axes = (
        MDHistoAxis("DeltaE", np.array([0.0, 1.0, 2.0]), "meV", "energy"),
        MDHistoAxis("[H,-H,0]", np.array([-1.0, 0.0, 1.0]), "rlu", "momentum"),
        MDHistoAxis("[0,0,L]", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),
        MDHistoAxis("[H,H,0]", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),
    )
    data = MDHistoData(
        axes=axes,
        signal=np.zeros((2, 2, 2, 2)),
        errors=np.ones((2, 2, 2, 2)),
        mask=np.zeros((2, 2, 2, 2), dtype=bool),
        num_events=np.ones((2, 2, 2, 2)),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)
    config = dataset_rebin_config(dataset)
    config["axes"][1]["vector"] = [1.0, 1.0, 1.0, 0.0]
    config["axes"][1]["name"] = "[H,H,H]"
    config["axes"][2]["vector"] = [1.0, 1.0, -2.0, 0.0]
    config["axes"][2]["name"] = "[L,L,-2L]"
    saved_axes = [dict(axis) for axis in config["axes"]]

    # Project reload and fit-history changes can temporarily leave a
    # file-backed dataset unloaded before it is drawn again.
    dataset.unload_data()
    assert dataset_rebin_config(dataset)["axes"] == saved_axes

    dataset.replace_data(data)
    restored = dataset_rebin_config(dataset)
    assert restored["axes"] == saved_axes


def test_rebin_basis_change_regenerates_names_and_keeps_full_data_extent():
    axes = (
        MDHistoAxis("DeltaE", np.array([-12.7, 0.0, 12.7]), "meV", "energy"),
        MDHistoAxis("[H,-H,0]", np.array([-0.5, 0.0, 0.5]), "rlu", "momentum"),
        MDHistoAxis("[0,0,L]", np.array([-5.5, 0.0, 5.5]), "rlu", "momentum"),
        MDHistoAxis("[H,H,0]", np.array([-4.0, 0.0, 4.0]), "rlu", "momentum"),
    )
    signal = np.arange(16, dtype=float).reshape((2, 2, 2, 2))
    data = MDHistoData(
        axes=axes,
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)
    config = dataset_rebin_config(dataset)

    project_gui._update_mdhisto_rebin_basis(data, config["axes"], 2, [1.0, 0.0, -2.0, 0.0])
    project_gui._update_mdhisto_rebin_basis(data, config["axes"], 3, [1.0, 1.0, 1.0, 0.0])

    assert [axis["variable"] for axis in config["axes"]] == ["E", "H", "L", "H"]
    assert [axis["name"] for axis in config["axes"]] == [
        "DeltaE",
        "[H,-H,0]",
        "[L,0,-2L]",
        "[H,H,H]",
    ]

    transform = project_gui._mdhisto_rebin_basis_transform(data, config["axes"])
    source_bounds = [project_gui._axis_bounds(axis, 2) for axis in axes]
    corners = np.array(
        [
            [bounds[(bits >> index) & 1] for index, bounds in enumerate(source_bounds)]
            for bits in range(16)
        ],
        dtype=float,
    )
    transformed_corners = corners @ transform
    for index, axis in enumerate(config["axes"]):
        assert axis["lower"] == pytest.approx(np.min(transformed_corners[:, index]))
        assert axis["upper"] == pytest.approx(np.max(transformed_corners[:, index]))
        assert axis["num_bins"] >= 1

    config["enabled"] = True
    for axis in config["axes"]:
        axis["mode"] = "discrete"
    rebinned = project_gui.rebinned_dataset_data(dataset)
    assert [axis.name for axis in rebinned.axes] == [
        "DeltaE",
        "[H,-H,0]",
        "[L,0,-2L]",
        "[H,H,H]",
    ]


def test_rebin_basis_rejects_energy_mixing_and_dependent_axes():
    axes = (
        MDHistoAxis("DeltaE", np.array([0.0, 1.0, 2.0]), "meV", "energy"),
        MDHistoAxis("[H,-H,0]", np.array([-1.0, 0.0, 1.0]), "rlu", "momentum"),
        MDHistoAxis("[0,0,L]", np.array([-1.0, 0.0, 1.0]), "rlu", "momentum"),
        MDHistoAxis("[H,H,0]", np.array([-1.0, 0.0, 1.0]), "rlu", "momentum"),
    )
    data = MDHistoData(
        axes=axes,
        signal=np.zeros((2, 2, 2, 2)),
        errors=np.ones((2, 2, 2, 2)),
        mask=np.zeros((2, 2, 2, 2), dtype=bool),
        num_events=np.ones((2, 2, 2, 2)),
        metadata={},
    )
    config = dataset_rebin_config(DatasetEntry("scan", data))

    mixed_energy = [dict(axis) for axis in config["axes"]]
    mixed_energy[0]["vector"] = [1.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="energy variable"):
        project_gui._validate_mdhisto_rebin_basis(mixed_energy, 4)

    mixed_momentum = [dict(axis) for axis in config["axes"]]
    mixed_momentum[1]["vector"] = [1.0, -1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="cannot be mixed"):
        project_gui._validate_mdhisto_rebin_basis(mixed_momentum, 4)

    dependent = [dict(axis) for axis in config["axes"]]
    dependent[3]["vector"] = list(dependent[1]["vector"])
    with pytest.raises(ValueError, match="invertible basis"):
        project_gui._validate_mdhisto_rebin_basis(dependent, 4)


def test_saved_custom_rebin_basis_requires_current_basis_version():
    axes = (
        MDHistoAxis("DeltaE", np.array([-10.0, 0.0, 10.0]), "meV", "energy"),
        MDHistoAxis("[H,-H,0]", np.array([-0.5, 0.0, 0.5]), "rlu", "momentum"),
        MDHistoAxis("[0,0,L]", np.array([-5.0, 0.0, 5.0]), "rlu", "momentum"),
        MDHistoAxis("[H,H,0]", np.array([-4.0, 0.0, 4.0]), "rlu", "momentum"),
    )
    data = MDHistoData(
        axes=axes,
        signal=np.zeros((2, 2, 2, 2)),
        errors=np.ones((2, 2, 2, 2)),
        mask=np.zeros((2, 2, 2, 2), dtype=bool),
        num_events=np.ones((2, 2, 2, 2)),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)
    config = dataset_rebin_config(dataset)
    config["axes"][2]["vector"] = [1.0, 0.0, -2.0, 0.0]
    config["axes"][3]["vector"] = [1.0, 1.0, 1.0, 0.0]
    config["axes"][1].update({"lower": -0.5, "upper": 0.5, "num_bins": 2})
    config.pop("coordinate_basis_version")

    with pytest.raises(ValueError, match="coordinate_basis_version"):
        dataset_rebin_config(dataset)


def test_add_mask_and_slice_viewer_from_masks_node(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    opened = []

    class FakeViewer:
        def __init__(self, datasets, *, dataset_names, dataset_group_keys=None):
            self.dataset_names = list(dataset_names)
            self.selected = None
            self.shown = False
            self.window = None
            opened.append(self)

            class _Combo:
                def __init__(self, names):
                    self.names = names
                    self.index = 0

                def setCurrentIndex(self, index):
                    self.index = int(index)

                def currentText(self):
                    return self.names[self.index]

            self.dataset_combo = _Combo(self.dataset_names)

        def show(self):
            self.shown = True

    monkeypatch.setattr(project_gui, "QtMDHistoSliceViewer", FakeViewer)

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))

    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    assert dataset_item.childCount() == 0
    explorer.tree.setCurrentItem(dataset_item)

    action_names = explorer.context_menu_action_names(dataset_item)
    assert "Add mask" in action_names
    assert "View in data viewer" in action_names
    assert not explorer.add_mask_button.isHidden()
    assert not explorer.view_slice_button.isHidden()

    mask = explorer.add_mask_to_selection()
    assert mask is not None
    assert dataset.masks == [mask]

    # Re-select the Masks node and open the viewer for the owning dataset.
    masks_item = explorer.tree.topLevelItem(0).child(0).child(0).child(0)
    explorer.tree.setCurrentItem(masks_item)
    viewer = explorer.open_slice_viewer_for_selection()

    assert viewer is opened[0]
    assert viewer.dataset_combo.currentText() == "scan"


def test_dataset_details_text_summarizes_point_data_conditions():
    data = PointData4D(
        H=[0.0, 1.0],
        K=[0.0, 0.0],
        L=[0.0, 0.5],
        E=[3.0, 4.0],
        intensity=[10.0, 11.0],
        sigma=[1.0, 1.0],
        temperature=[5.0, 10.0],
        metadata={"field": "3 T"},
    )
    dataset = DatasetEntry("points", data, kind="point", parameters={"scan": 17})

    text = dataset_details_text(dataset)

    assert "Dimensions: 4" in text
    assert "H (rlu) - 2 points, range 0 to 1" in text
    assert "E (meV) - 2 points, range 3 to 4" in text
    assert "Valid points: 2" in text
    assert "Temperature: 5 to 10 K" in text
    assert "field: 3 T" in text
    assert "scan: 17" in text


def test_fit_points_carry_temperature_and_group_lattice():
    data = _grid_mdhisto_data()
    data.metadata["temperature"] = 4.2
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        lattice_parameters={
            "a": 4.0,
            "b": 4.0,
            "c": 8.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
    )

    bundle = project_gui.fit_data_bundle(group, dataset)
    assert bundle is not None
    assert bundle.points.temperature == 4.2
    matrix = np.asarray(bundle.points.metadata["rlu_to_inv_angstrom_matrix"])
    np.testing.assert_allclose(matrix[0, 0], 2.0 * np.pi / 4.0)
    np.testing.assert_allclose(matrix[2, 2], 2.0 * np.pi / 8.0)

    dataset.parameters["temperature"] = 100.0
    bundle = project_gui.fit_data_bundle(group, dataset)
    assert bundle.points.temperature == 100.0
    assert project_gui.effective_dataset_temperature(group, dataset) == 100.0


def test_fit_points_leave_temperature_unset_without_metadata():
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    bundle = project_gui.fit_data_bundle(group, dataset)
    assert bundle.points.temperature is None
    assert "rlu_to_inv_angstrom_matrix" not in bundle.points.metadata


def test_dataset_kinematic_kf_ki_normalization_uses_incident_energy_without_mutating_raw_data():
    data = MDHistoData(
        axes=(
            MDHistoAxis("H", np.array([0.0, 1.0]), "r.l.u.", "momentum"),
            MDHistoAxis("DeltaE", np.array([0.0, 2.0, 4.0]), "meV", "energy"),
        ),
        signal=np.array([[10.0, 10.0]]),
        errors=np.array([[2.0, 2.0]]),
        mask=np.zeros((1, 2), dtype=bool),
        num_events=np.ones((1, 2)),
    )
    dataset = DatasetEntry(
        "scan",
        data,
        parameters={project_gui.KINEMATIC_KF_KI_INCLUDED_KEY: False},
        metadata={"incident_energy": 10.0},
    )

    view = dataset_for_slice_viewer(dataset)

    factor = np.sqrt(np.array([0.9, 0.7]))
    np.testing.assert_allclose(view.signal[0], 10.0 * factor)
    np.testing.assert_allclose(view.errors[0], 2.0 * factor)
    np.testing.assert_allclose(data.signal, [[10.0, 10.0]])
    np.testing.assert_allclose(data.errors, [[2.0, 2.0]])


def test_dataset_temperature_spin_writes_override(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    spin = explorer.dataset_temperature_spin
    assert spin.specialValueText() == "(from data)"
    assert spin.value() == -1.0

    spin.setValue(4.2)
    assert dataset.parameters["temperature"] == 4.2

    spin.setValue(-1.0)
    assert "temperature" not in dataset.parameters


def test_dataset_temperature_edit_from_active_result_refreshes_current_state(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    create_model_component(group)
    project_gui.ensure_fit_history(group)
    result = project_gui.create_fit_result_entry(
        group,
        group.fits[0],
        goodness={"status": "converged", "parameters": {}},
    )
    result.optimizer_config = {
        "initialization": {"enabled": True, "method": "differential_evolution", "maxiter": 17},
        "sampler": {"enabled": True, "method": "emcee", "n_steps": 31},
    }
    explorer = NfitProjectExplorer(NfitProject([group]))

    result_item = explorer.tree.topLevelItem(0).child(2).child(1)
    explorer.tree.setCurrentItem(result_item)
    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(dataset_item)

    explorer.dataset_temperature_spin.setValue(12.5)

    assert [fit.kind for fit in group.fits] == ["initial", "result", "current"]
    assert group.fits[2].snapshot["datasets"][0]["parameters"]["temperature"] == 12.5
    assert group.fits[2].optimizer_config == result.optimizer_config
    assert group.fits[2].optimizer_config is not result.optimizer_config
    fits_item = explorer.tree.topLevelItem(0).child(2)
    assert fits_item.childCount() == 3
    assert explorer._fit_entry_for_item(fits_item.child(2)) is group.fits[2]
    assert result.children == []
    explorer.tree.setCurrentItem(fits_item.child(2))
    assert explorer.fit_de_check.isChecked()
    assert explorer.fit_emcee_check.isChecked()


def test_failed_fit_from_result_creates_current_state_for_temperature_fix(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    create_model_component(group)
    project_gui.ensure_fit_history(group)
    first_result = project_gui.create_fit_result_entry(
        group,
        group.fits[0],
        goodness={"status": "converged", "parameters": {}},
    )
    explorer = NfitProjectExplorer(NfitProject([group]))

    def fail_fit(group, *, optimizer_config=None, progress_callback=None):
        raise ValueError("this model requires a valid sample temperature")

    monkeypatch.setattr(project_gui, "perform_group_fit", fail_fit)
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(1))

    failed = explorer.fit_now_for_selection()

    assert failed is not None
    assert failed.goodness["status"] == "failed"
    assert [fit.kind for fit in group.fits] == ["initial", "result", "result", "current"]
    assert group.fits[1] is first_result
    assert group.fits[2] is failed
    current = group.fits[3]
    assert explorer._active_fit_entry(group) is current
    assert explorer.tree.currentItem().text(0) == "Current state"

    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(dataset_item)
    explorer.dataset_temperature_spin.setValue(8.0)

    assert [fit.kind for fit in group.fits] == ["initial", "result", "result", "current"]
    assert group.fits[3] is current
    assert current.snapshot["datasets"][0]["parameters"]["temperature"] == 8.0
    assert explorer._active_fit_entry(group) is current


def test_failed_fit_from_current_state_keeps_current_state_selected(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    create_model_component(group)
    project_gui.ensure_fit_history(group)
    current = project_gui.current_state_fit_entry(group)
    group.fits.append(current)
    explorer = NfitProjectExplorer(NfitProject([group]))

    def fail_fit(group, *, optimizer_config=None, progress_callback=None):
        raise ValueError("this model requires a valid sample temperature")

    monkeypatch.setattr(project_gui, "perform_group_fit", fail_fit)
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(1))

    failed = explorer.fit_now_for_selection()

    assert failed is not None
    assert failed.goodness["status"] == "failed"
    assert [fit.kind for fit in group.fits] == ["initial", "result", "current"]
    assert group.fits[1] is failed
    assert group.fits[2] is current
    assert explorer._active_fit_entry(group) is current
    assert explorer.tree.currentItem().text(0) == "Current state"
