"""Calibration dependencies invalidate reduced MACS arrays and histograms."""

from nfit import DataGroup
from nfit.project_imports import import_dataset_paths
from tests.test_macs import _write_macs_nexus


def test_loaded_macs_calibration_refreshes_when_source_changes(tmp_path, monkeypatch):
    from nfit import project_imports
    from nfit.macs_vanadium import vanadium_source_provenance

    source = _write_macs_nexus(tmp_path / "run.ng0")
    calibration = tmp_path / "vanadium.ng0"
    calibration.write_bytes(b"first measurement")
    root = DataGroup("MACS")
    import_dataset_paths(root, [source])
    dataset = root.subgroups[0].datasets[0]
    dataset.metadata["import_options"]["vanadium_files"] = [str(calibration)]
    metadata = dict(dataset.data.metadata)
    metadata["macs_vanadium"] = {"source_files": vanadium_source_provenance([calibration])}
    dataset.replace_data(dataset.data.with_updates(metadata=metadata), source_backed=True)
    original = dataset.data
    assert project_imports._ensure_dataset_data_loaded(dataset) is original
    calibration.write_bytes(b"updated longer measurement")
    sentinel = object()
    calls = []

    def reload(entry, **kwargs):
        calls.append(entry)
        assert entry.data is None
        return sentinel

    monkeypatch.setattr(project_imports, "_load_dataset_source", reload)
    assert project_imports._ensure_dataset_data_loaded(dataset) is sentinel
    assert calls == [dataset]


def test_diff_ignores_vanadium_dependencies(tmp_path):
    from nfit.project_imports import _macs_reduction_is_stale

    source = _write_macs_nexus(tmp_path / "run.ng0")
    root = DataGroup("MACS")
    import_dataset_paths(root, [source])
    dataset = root.subgroups[0].datasets[0]
    dataset.metadata["import_options"].update(stream="diff", vanadium_files=["missing.ng0"])
    assert not _macs_reduction_is_stale(dataset, dataset.data)


def test_reload_without_calibration_clears_previous_diagnostics(tmp_path):
    from nfit.project_imports import _ensure_dataset_data_loaded

    source = _write_macs_nexus(tmp_path / "run.ng0")
    root = DataGroup("MACS")
    import_dataset_paths(root, [source])
    dataset = root.subgroups[0].datasets[0]
    dataset.metadata["macs_vanadium"] = {"source_files": [{"path": "obsolete.ng0"}]}
    dataset.metadata["vanadium_sensitivity_applied"] = True
    dataset.unload_data()
    _ensure_dataset_data_loaded(dataset)
    assert "macs_vanadium" not in dataset.metadata
    assert dataset.metadata["vanadium_sensitivity_applied"] is False


def test_lazy_artifact_rejects_changed_calibration(tmp_path, monkeypatch):
    from nfit import project_imports
    from nfit.macs_vanadium import vanadium_source_provenance

    source = _write_macs_nexus(tmp_path / "run.ng0")
    calibration = tmp_path / "vanadium.ng0"
    calibration.write_bytes(b"original")
    root = DataGroup("MACS")
    import_dataset_paths(root, [source])
    dataset = root.subgroups[0].datasets[0]
    metadata = dict(dataset.data.metadata)
    metadata["macs_vanadium"] = {"source_files": vanadium_source_provenance([calibration])}
    artifact = dataset.data.with_updates(metadata=metadata)
    dataset.metadata.update(project_artifact_path="data.npz", _project_path=str(tmp_path / "test.nfit"))
    dataset.metadata["import_options"]["vanadium_files"] = [str(calibration)]
    dataset.unload_data()
    monkeypatch.setattr(project_imports, "read_project_dataset_artifact", lambda *a, **k: artifact)
    calibration.write_bytes(b"updated longer measurement")
    result = project_imports._artifact_source_load(dataset, None)
    assert result is project_imports._SOURCE_NOT_HANDLED
    assert dataset.data is None


def test_macs_histogram_signature_tracks_calibration_file_contents(tmp_path):
    from nfit.project_composites import _composite_cache_signature

    source = _write_macs_nexus(tmp_path / "run.ng0")
    calibration = tmp_path / "vanadium.ng0"
    calibration.write_bytes(b"first measurement")
    root = DataGroup("MACS")
    import_dataset_paths(root, [source])
    group = root.subgroups[0]
    group.metadata["macs"]["vanadium_files"] = [str(calibration)]
    before = _composite_cache_signature(group)
    assert _composite_cache_signature(group) == before
    calibration.write_bytes(b"updated longer measurement")
    assert _composite_cache_signature(group) != before
    calibration.unlink()
    assert _composite_cache_signature(group) != before


def test_dataset_and_parent_signatures_track_calibration_before_loading(tmp_path):
    from nfit.cache_utils import dataset_content_signature
    from nfit.project_composites import _composite_cache_signature
    from nfit.project_data import _viewer_view_signature

    source = _write_macs_nexus(tmp_path / "run.ng0")
    calibration = tmp_path / "vanadium.ng0"
    calibration.write_bytes(b"original")
    root = DataGroup("MACS")
    import_dataset_paths(root, [source])
    dataset = root.subgroups[0].datasets[0]
    dataset.metadata["import_options"]["vanadium_files"] = [str(calibration)]
    dataset.unload_data()
    signature = dataset_content_signature(dataset)
    viewer = _viewer_view_signature(dataset, None)
    parent = _composite_cache_signature(root)
    calibration.write_bytes(b"changed longer measurement")
    assert dataset_content_signature(dataset) != signature
    assert _viewer_view_signature(dataset, None) != viewer
    assert _composite_cache_signature(root) != parent
    assert dataset.data is None
