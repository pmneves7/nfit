from pathlib import Path

import numpy as np
import pytest

from nfit import DataGroup, bin_mdevent_group, bin_raw_dgs_group
from nfit.source_selection import SourceSelection, resolve_source_selection
from nfit.source_selection_imports import (
    import_source_selection,
    source_selection_script,
    validate_source_selection_combination,
)
from tests.test_mdevent import _write_mdevent
from tests.test_raw_dgs import _write_raw_dgs


def _raw_selection(tmp_path, expression):
    for number in (1, 2):
        _write_raw_dgs(tmp_path / f"SEQ_{number}.nxs.h5")
    return SourceSelection(tmp_path, "SEQ_", ".nxs.h5", expression)


def test_flat_default_preserves_unique_sources_and_repeat_provenance(tmp_path):
    selection = _raw_selection(tmp_path, "1+1,2,1,0")
    parent = DataGroup("NiO")
    group = import_source_selection(parent, resolve_source_selection(selection))
    assert parent.subgroups == [group]
    assert group.subgroups == []
    assert group.metadata["raw_dgs"]["format"] == "raw-direct-geometry-nexus"
    assert len(group.datasets) == 2
    assert [dataset.fit_weight for dataset in group.datasets] == [1.0, 1.0]
    assert all(dataset.data is None for dataset in group.datasets)
    lineage = group.datasets[0].metadata["source_selection_lineage"]
    assert lineage["multiplicity"] == 3
    assert lineage["coaddition_coefficient"] == 1
    assert lineage["appearance_indices"] == [0, 1, 3]
    assert group.metadata["source_selection"]["preserve_groups"] is False
    assert group.metadata["reduction_recipe"]["source_selection"]["expression"] == selection.expression
    validate_source_selection_combination(group.datasets)


def test_flat_range_separators_change_provenance_without_changing_membership(tmp_path):
    from nfit.project_composites import _composite_cache_signature, data_group_composite_config
    from nfit.source_selection_imports import update_source_selection

    for number in (1, 2, 4, 5):
        _write_raw_dgs(tmp_path / f"SEQ_{number}.nxs.h5")
    root = DataGroup("sample")
    group = import_source_selection(
        root, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1:2+4:5"),
    )
    previous = [(d.id, d.fit_weight, d.metadata['source_file']) for d in group.datasets]
    config = data_group_composite_config(group)
    config['stale'] = False
    signature = _composite_cache_signature(group)
    edit = update_source_selection(
        group, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1:2,4:5"), parent=root,
    )
    assert edit.settings_changed and not edit.membership_changed
    assert not edit.added_dataset_ids and not edit.removed_dataset_ids
    assert [(d.id, d.fit_weight, d.metadata['source_file']) for d in group.datasets] == previous
    assert config['stale'] is False
    assert _composite_cache_signature(group) == signature
    assert group.metadata['source_selection']['expression'] == "1:2,4:5"
    assert group.metadata['reduction_recipe']['source_selection']['expression'] == "1:2,4:5"


def test_preserved_groups_pool_runs_and_keep_empty_and_alias_groups(tmp_path):
    selection = _raw_selection(tmp_path, "1+2,0,1|2|")
    group = import_source_selection(DataGroup("sample"), selection, preserve_groups=True)
    assert [len(child.datasets) for child in group.subgroups] == [2, 0, 1, 1]
    assert [child.enabled for child in group.subgroups] == [True, False, True, True]
    assert group.subgroups[1].metadata["source_selection_group"]["empty"] is True
    left, right = (child.datasets[0] for child in group.subgroups[2:])
    assert left.id != right.id
    assert left.metadata["source_selection_lineage"]["source_identity"] == right.metadata["source_selection_lineage"]["source_identity"]
    assert group.subgroups[0].metadata["raw_dgs"]["source_files"] == [
        str(tmp_path / "SEQ_1.nxs.h5"), str(tmp_path / "SEQ_2.nxs.h5")]
    from nfit.measurement_dependencies import SourceReplayRequired

    validate_source_selection_combination(group.subgroups[0].datasets)
    with pytest.raises(SourceReplayRequired, match="reuse"):
        validate_source_selection_combination(group.iter_datasets())
    # Removing one alias enables combining disjoint sources; disabling it also
    # makes it irrelevant without modifying its saved appearance provenance.
    right.enabled = False
    validate_source_selection_combination([left, right])


@pytest.mark.parametrize("family", ["raw", "mdevent"])
def test_repeated_coaddition_retains_same_counting_precision(tmp_path, family):
    if family == "raw":
        selection = _raw_selection(tmp_path, "1+1")
        baseline_selection = SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1")
        reducer = bin_raw_dgs_group
        settings = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 40], num_bins=[1]*4)
    else:
        _write_mdevent(tmp_path / "MDE_1.nxs")
        selection = SourceSelection(tmp_path, "MDE_", ".nxs", "1+1")
        baseline_selection = SourceSelection(tmp_path, "MDE_", ".nxs", "1")
        reducer = bin_mdevent_group
        settings = dict(lower=[-1]*4, upper=[1]*4, num_bins=[1]*4)
    baseline = import_source_selection(DataGroup("sample"), baseline_selection)
    repeated = import_source_selection(DataGroup("sample"), selection, preserve_groups=True).subgroups[0]
    assert all(dataset.fit_weight == 2.0 for dataset in repeated.datasets)
    validate_source_selection_combination(repeated.datasets)
    if family == "mdevent":
        assert [dataset.metadata["source_selection_lineage"]["source_identity"].split("#")[-1]
                for dataset in repeated.datasets] == ["experiment=0", "experiment=1"]
    original = reducer(baseline, **settings)
    coadded = reducer(repeated, **settings)
    np.testing.assert_allclose(coadded.signal, original.signal, equal_nan=True)
    np.testing.assert_allclose(coadded.errors, original.errors, equal_nan=True)
    np.testing.assert_allclose(coadded.metadata["normalization_denominator"],
                               2*np.asarray(original.metadata["normalization_denominator"]))


def test_missing_sources_are_atomic_and_skip_keeps_group_provenance(tmp_path):
    selection = _raw_selection(tmp_path, "1,3,0")
    parent = DataGroup("sample")
    with pytest.raises(FileNotFoundError, match="SEQ_3"):
        import_source_selection(parent, selection)
    assert parent.subgroups == []
    grouped = import_source_selection(parent, selection, missing="skip", preserve_groups=True)
    assert [len(child.datasets) for child in grouped.subgroups] == [1, 0, 0]
    assert grouped.subgroups[1].metadata["source_selection_group"]["skipped_appearance_indices"] == [1]
    assert grouped.metadata["source_selection"]["import_skipped_files"] == [str(tmp_path / "SEQ_3.nxs.h5")]


def test_unsupported_duplicate_coaddition_rejects_before_parent_mutation(tmp_path):
    from tests.test_corelli import _write_corelli

    _write_corelli(tmp_path / "CORELLI_1.nxs.h5")
    parent = DataGroup("sample")
    selection = SourceSelection(tmp_path, "CORELLI_", ".nxs.h5", "1+1")
    with pytest.raises(ValueError, match="covariance-aware adapter"):
        import_source_selection(parent, selection, preserve_groups=True)
    assert parent.subgroups == []
    flat = import_source_selection(parent, selection)
    assert len(flat.datasets) == 1 and flat.datasets[0].fit_weight == 1


def test_late_import_failure_is_atomic(tmp_path):
    selection = _raw_selection(tmp_path, "1,2")
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "SEQ_2.nxs.h5", "r+") as handle:
        handle["entry/DASlogs/BL17:Det:TH:BL:Ei/average_value"][0] = -1.0
    parent = DataGroup("sample")
    with pytest.raises((ValueError, OSError)):
        import_source_selection(parent, selection, preserve_groups=True)
    assert parent.subgroups == []


def test_flat_ordinary_files_use_archive_import_and_grouped_repeats_are_rejected(tmp_path):
    from nfit import DatasetEntry, MDHistoAxis, MDHistoData, save_dataset_file

    axis = MDHistoAxis("E", np.array([0.0, 1.0, 2.0]), "meV", "energy")
    data = MDHistoData([axis], np.array([2.0, 4.0]), np.array([0.2, 0.4]),
                       mask=np.zeros(2, dtype=bool), num_events=np.ones(2))
    source = tmp_path / "scan1.npz"
    save_dataset_file(DatasetEntry("measured", data), source, use_view=False)
    selection = SourceSelection(tmp_path, "scan", ".npz", "1+1")
    parent = DataGroup("measurements")
    flat = import_source_selection(parent, selection)
    assert len(flat.datasets) == 1
    np.testing.assert_array_equal(flat.datasets[0].data.signal, data.signal)
    assert not flat.datasets[0].data.signal.flags.writeable
    assert flat.datasets[0].fit_weight == 1
    with pytest.raises(ValueError, match="covariance-aware adapter"):
        import_source_selection(parent, selection, preserve_groups=True)
    assert parent.subgroups == [flat]


def test_source_selection_script_replays_editable_selection_and_import_settings(tmp_path):
    selection = _raw_selection(tmp_path, "1:2")
    script = source_selection_script(selection, preserve_groups=True, name="Selected runs")
    assert "load_project" not in script and "PySide" not in script
    namespace = {"__name__": "selection_replay"}
    exec(compile(script, "sources.py", "exec"), namespace)
    parent, group = namespace["run"]()
    assert parent.subgroups == [group]
    assert group.name == "Selected runs"
    assert [Path(child.datasets[0].metadata["source_file"]).name
            for child in group.subgroups] == ["SEQ_1.nxs.h5", "SEQ_2.nxs.h5"]
    namespace["IMPORT_SETTINGS"]["preserve_groups"] = False
    _, flat = namespace["run"]()
    assert len(flat.datasets) == 2 and not flat.subgroups


def test_source_selection_archive_roundtrip_is_lazy_and_preserves_run_recipe(tmp_path, monkeypatch):
    from nfit import NfitProject, load_project, save_project
    from nfit import raw_dgs as raw_dgs_module
    from nfit.reduction_recipes import effective_reduction_config, set_reduction_settings

    selection = _raw_selection(tmp_path, "1+1,2,0")
    parent = DataGroup("NiO")
    collection = import_source_selection(parent, selection, preserve_groups=True)
    first = collection.subgroups[0]
    set_reduction_settings(first, {"t0_override": -2.5})
    project_path = tmp_path / "selection.nfit"
    save_project(NfitProject([parent]), project_path)

    def unexpected_source_read(*args, **kwargs):
        raise AssertionError("opening a saved selection must not inspect raw sources")

    monkeypatch.setattr(raw_dgs_module, "inspect_raw_dgs_run", unexpected_source_read)
    restored = load_project(project_path).data_groups[0].subgroups[0]
    assert restored.metadata["source_selection"] == collection.metadata["source_selection"]
    assert [len(group.datasets) for group in restored.subgroups] == [1, 1, 0]
    assert restored.subgroups[2].enabled is False
    restored_run = restored.subgroups[0].datasets[0]
    assert restored_run.id == first.datasets[0].id
    assert restored_run.data is None
    assert restored_run.fit_weight == 2.0
    assert restored_run.metadata["source_selection_lineage"] == first.datasets[0].metadata["source_selection_lineage"]
    assert effective_reduction_config(restored.subgroups[0], restored_run)["t0_override"] == -2.5


def test_composite_service_rejects_alias_merge_and_allows_one_group(tmp_path):
    from nfit import composite_dataset_data
    from nfit.measurement_dependencies import SourceReplayRequired
    from nfit.project_composites import data_group_composite_config

    parent = DataGroup("sample")
    collection = import_source_selection(parent, _raw_selection(tmp_path, "1|2|"), preserve_groups=True)
    config = data_group_composite_config(collection.subgroups[0])
    config.update(enabled=True, auto_rebin=False, minimum_coverage=0.0, resolution_mode="bins")
    for index, axis in enumerate(config["axes"]):
        axis.update(lower=-10 if index < 3 else -100, upper=10 if index < 3 else 40,
                    auto_lower=False, auto_upper=False, num_bins=1, auto_step_size=False,
                    mode="bins", step_size=20 if index < 3 else 140)
    with pytest.raises(SourceReplayRequired, match="reuse"):
        composite_dataset_data(parent, node=collection, config_override=config)
    collection.subgroups[1].enabled = False
    result = composite_dataset_data(parent, node=collection, config_override=config)
    assert result.shape == (1, 1, 1, 1)
    assert np.isfinite(result.signal).any()


@pytest.mark.parametrize("family", ["raw", "mdevent", "corelli"])
def test_native_companion_calibration_is_visible_and_can_be_cleared(tmp_path, family):
    from nfit.reduction_recipes import set_reduction_settings
    from tests.test_corelli import _write_corelli
    from tests.test_mdevent import _write_normalization

    if family == "raw":
        selection = _raw_selection(tmp_path, "1")
    elif family == "mdevent":
        _write_mdevent(tmp_path / "MDE_1.nxs")
        selection = SourceSelection(tmp_path, "MDE_", ".nxs", "1")
    else:
        _write_corelli(tmp_path / "CORELLI_1.nxs.h5")
        selection = SourceSelection(tmp_path, "CORELLI_", ".nxs.h5", "1")
    companion = tmp_path / "vanadium.nxs"
    _write_normalization(companion, 2.0)
    parent = DataGroup("sample")
    group = import_source_selection(parent, selection)
    config_key = "mdevent" if family == "mdevent" else "raw_dgs"
    assert group.metadata[config_key]["normalization_file"] == str(companion)
    assert group.metadata[config_key]["mask_file"] == str(companion)
    assert group.metadata["source_selection_calibration"]["discovery"] == "single_van_companion"
    assert group.metadata["reduction_recipe"]["source_selection"]["calibration_discovery"]["mask_file"] == str(companion)
    if family == "mdevent":
        assert parent.lattice_parameters == group.metadata["mdevent"]["lattice_parameters"]
    set_reduction_settings(group, {"normalization_file": None, "mask_file": None})
    assert group.metadata[config_key]["normalization_file"] is None
    assert group.metadata["reduction_recipe"]["shared_defaults"]["mask_file"] is None


def test_registered_multistream_sources_have_distinct_lineage_but_same_stream_aliases_do_not(tmp_path):
    from nfit.measurement_dependencies import SourceReplayRequired
    from tests.test_macs import _write_macs_nexus

    _write_macs_nexus(tmp_path / "MACS_1.nxs", points=2)
    selection = SourceSelection(tmp_path, "MACS_", ".nxs", "1")
    group = import_source_selection(DataGroup("MACS"), selection)
    entries = list(group.iter_datasets())
    assert len(entries) == 2
    assert {dataset.metadata["import_options"]["stream"] for dataset in entries} == {"spec", "diff"}
    assert len({dataset.metadata["source_selection_lineage"]["source_identity"] for dataset in entries}) == 2
    validate_source_selection_combination(entries)

    repeated = import_source_selection(DataGroup("MACS"),
        SourceSelection(tmp_path, "MACS_", ".nxs", "1|2|"), preserve_groups=True)
    validate_source_selection_combination(repeated.subgroups[0].iter_datasets())
    with pytest.raises(SourceReplayRequired, match="reuse"):
        validate_source_selection_combination(repeated.iter_datasets())


def test_source_membership_edit_reuses_entries_reductions_and_only_inspects_new(tmp_path, monkeypatch):
    from nfit import raw_dgs as module
    from nfit.reduction_recipes import effective_reduction_config, set_reduction_settings
    from nfit.source_selection_imports import update_source_selection

    parent = DataGroup("sample")
    group = import_source_selection(parent, _raw_selection(tmp_path, "1"))
    original = group.datasets[0]
    set_reduction_settings(group, {"t0_override": -2.5})
    cache = original._raw_dgs_reduction_cache = object()
    original.parameters["rebin"] = {"stale": False}
    group.metadata["composite"]["stale"] = False
    parent.metadata["composite"] = {"stale": False}
    seen = []
    inspect = module.inspect_raw_dgs_run

    def inspecting(path, **kwargs):
        seen.append(Path(path).name)
        assert Path(path).name == "SEQ_2.nxs.h5"
        return inspect(path, **kwargs)

    monkeypatch.setattr(module, "inspect_raw_dgs_run", inspecting)
    edit = update_source_selection(group, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1:2"), parent=parent)
    assert seen == ["SEQ_2.nxs.h5"]
    assert group.datasets[0] is original
    assert original._raw_dgs_reduction_cache is cache
    assert original.parameters["rebin"]["stale"] is False
    assert effective_reduction_config(group, group.datasets[1])["t0_override"] == -2.5
    assert edit.added_dataset_ids == (group.datasets[1].id,)
    assert edit.retained_dataset_ids == (original.id,)
    assert group.metadata["composite"]["stale"] and parent.metadata["composite"]["stale"]


def test_source_membership_missing_and_invalid_new_import_are_atomic(tmp_path):
    from nfit.source_selection_imports import update_source_selection
    parent = DataGroup("sample")
    group = import_source_selection(parent, _raw_selection(tmp_path, "1"))
    original = group.datasets[0]
    metadata = group.metadata.copy()
    with pytest.raises(FileNotFoundError):
        update_source_selection(group, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1,3"))
    assert group.datasets == [original] and group.metadata == metadata
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "SEQ_2.nxs.h5", "r+") as handle:
        handle["entry/DASlogs/BL17:Det:TH:BL:Ei/average_value"][0] = -1.0
    with pytest.raises(ValueError):
        update_source_selection(group, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1,2"))
    assert group.datasets == [original] and group.metadata == metadata


def test_source_membership_edit_regroup_repeat_and_clear(tmp_path):
    from nfit.source_selection_imports import update_source_selection
    parent = DataGroup("sample")
    group = import_source_selection(parent, _raw_selection(tmp_path, "1+1,2"), preserve_groups=True)
    first, second = group.subgroups
    original = first.datasets[0]
    second_original = second.datasets[0]
    first.metadata["composite"]["stale"] = False
    second.metadata["composite"]["stale"] = False
    update_source_selection(group, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1+1+1,2"))
    assert group.subgroups == [first, second]
    assert first.datasets[0] is original and original.fit_weight == 3.0
    assert first.metadata["composite"]["stale"] is True
    assert second.datasets[0] is second_original
    assert second.metadata["composite"]["stale"] is False
    update_source_selection(group, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1+1,2"), preserve_groups=False)
    assert not group.subgroups
    assert list(group.iter_datasets()) == [original, second_original]
    assert original.fit_weight == 1.0
    edit = update_source_selection(group, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "x"))
    assert not list(group.iter_datasets()) and not group.enabled
    assert edit.removed_dataset_ids == (original.id, second_original.id)


def test_source_membership_edit_preserves_macs_stream_folders_and_entries(tmp_path):
    from nfit.source_selection_imports import update_source_selection
    from tests.test_macs import _write_macs_nexus
    for run in (1, 2):
        _write_macs_nexus(tmp_path / f"MACS_{run}.nxs", points=2)
    parent = DataGroup("MACS")
    group = import_source_selection(parent, SourceSelection(tmp_path, "MACS_", ".nxs", "1"))
    folders = list(group.subgroups)
    entries = list(group.iter_datasets())
    edit = update_source_selection(group, SourceSelection(tmp_path, "MACS_", ".nxs", "1:2"))
    assert group.subgroups == folders
    assert len(list(group.iter_datasets())) == 4
    assert set(edit.retained_dataset_ids) == {entry.id for entry in entries}
    assert all(any(candidate is entry for candidate in group.iter_datasets()) for entry in entries)
    assert all(len(folder.datasets) == 2 for folder in folders)
    validate_source_selection_combination(group.iter_datasets())


def test_editing_summed_subfolder_keeps_correlated_coefficients_without_extra_folders(tmp_path):
    from nfit.source_selection_imports import update_source_selection
    root = DataGroup("sample")
    collection = import_source_selection(root, _raw_selection(tmp_path, "1+1,2"), preserve_groups=True)
    child = collection.subgroups[0]
    original = child.datasets[0]
    update_source_selection(child, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1+1+1+2"), parent=root,
                            preserve_groups=False)
    assert not child.subgroups
    assert child.datasets[0] is original and original.fit_weight == 3
    assert child.datasets[1].fit_weight == 1
    assert original.metadata["source_selection_lineage"]["coaddition_coefficient"] == 3


def test_membership_update_matches_fresh_reduction_and_saved_project_reopens_lazily(tmp_path, monkeypatch):
    from nfit import NfitProject, load_project, save_project
    from nfit import raw_dgs as module
    from nfit.reduction_recipes import set_reduction_settings
    from nfit.source_selection_imports import update_source_selection

    root = DataGroup("sample")
    group = import_source_selection(root, _raw_selection(tmp_path, "1"))
    settings = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 40], num_bins=[1]*4)
    bin_raw_dgs_group(group, **settings)
    retained = group.datasets[0]
    reduced_cache = retained._raw_dgs_reduction_cache
    update_source_selection(group, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1:2"), parent=root)
    actual = bin_raw_dgs_group(group, **settings)
    fresh = import_source_selection(DataGroup("fresh"), SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1:2"))
    expected = bin_raw_dgs_group(fresh, **settings)
    assert retained._raw_dgs_reduction_cache is reduced_cache
    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
    np.testing.assert_allclose(actual.metadata["normalization_denominator"], expected.metadata["normalization_denominator"])
    set_reduction_settings(group, {"t0_override": -1.25}, dataset_ids=[retained.id])
    path = tmp_path / "edited.nfit"
    save_project(NfitProject([root]), path)

    def unexpected_inspection(*args, **kwargs):
        raise AssertionError("saved membership must reopen without source inspection")

    monkeypatch.setattr(module, "inspect_raw_dgs_run", unexpected_inspection)
    reopened = load_project(path).data_groups[0].subgroups[0]
    assert [entry.id for entry in reopened.datasets] == [entry.id for entry in group.datasets]
    assert all(entry.data is None for entry in reopened.datasets)
    assert reopened.metadata["source_selection"]["expression"] == "1:2"


@pytest.mark.parametrize("external,disabled", [(False, False), (True, False), (True, True)])
def test_source_membership_removal_rejects_dangling_background_links_atomically(tmp_path, external, disabled):
    import copy

    from nfit import BackgroundSpec, DatasetEntry
    from nfit.source_selection_imports import update_source_selection

    root = DataGroup("sample")
    collection = import_source_selection(root, _raw_selection(tmp_path, "1:2"))
    removed = collection.datasets[1]
    consumer = DatasetEntry("consumer", None)
    consumer.backgrounds.append(BackgroundSpec("linked source", source_dataset_id=removed.id,
                                              enabled=not disabled))
    target = DataGroup("other", datasets=[consumer]) if external else root
    if not external:
        root.datasets.append(consumer)
    before_entries = list(collection.datasets)
    before_metadata = copy.deepcopy(collection.metadata)
    options = {"parent": root}
    if external:
        options["reference_roots"] = (node for node in (root, target))
    with pytest.raises(ValueError, match="edit or remove that background link first"):
        update_source_selection(collection, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1"), **options)
    assert collection.datasets == before_entries
    assert all(left is right for left, right in zip(collection.datasets, before_entries, strict=True))
    assert collection.metadata == before_metadata
    assert consumer.backgrounds[0].source_dataset_id == removed.id


def test_source_membership_regroup_rejects_removing_referenced_subfolder(tmp_path):
    import copy

    from nfit import BackgroundSpec
    from nfit.source_selection_imports import update_source_selection

    root = DataGroup("sample")
    collection = import_source_selection(root, _raw_selection(tmp_path, "1,2"), preserve_groups=True)
    removed_folder = collection.subgroups[1]
    root.backgrounds.append(BackgroundSpec("folder source", source_group_id=removed_folder.id, enabled=False))
    before = list(collection.subgroups)
    metadata = copy.deepcopy(collection.metadata)
    with pytest.raises(ValueError, match="folder source"):
        update_source_selection(collection, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1,2"),
                                parent=root, preserve_groups=False)
    assert collection.subgroups == before
    assert all(left is right for left, right in zip(collection.subgroups, before, strict=True))
    assert collection.metadata == metadata


def test_source_removal_allows_links_owned_only_by_removed_members(tmp_path):
    import shutil

    from nfit import BackgroundSpec
    from nfit.source_selection_imports import update_source_selection

    root = DataGroup("sample")
    selection = _raw_selection(tmp_path, "1:3")
    shutil.copyfile(tmp_path / "SEQ_2.nxs.h5", tmp_path / "SEQ_3.nxs.h5")
    collection = import_source_selection(root, selection)
    collection.datasets[1].backgrounds.append(
        BackgroundSpec("removed link", source_dataset_id=collection.datasets[2].id)
    )
    update_source_selection(collection, SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1"), parent=root)
    assert len(collection.datasets) == 1
