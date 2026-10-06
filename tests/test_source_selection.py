"""GRASP-inspired selection grammar, bounded expansion and source previews."""

import json
from pathlib import Path

import pytest

from nfit.source_selection import (
    MAX_SOURCE_APPEARANCES,
    MAX_SOURCE_NUMBER_PADDING,
    RUN_EXPRESSION_HELP,
    SourceSelection,
    parse_run_expression,
    resolve_source_selection,
)


def test_filename_wildcards_resolve_variable_macs_prefix_and_reject_ambiguity(tmp_path):
    first = tmp_path / "Ef3p7_et_1.1_271.nxs.ng0"
    second = tmp_path / "Ef3p7_et_1.2_244.nxs.ng0"
    first.touch()
    second.touch()
    selection = SourceSelection(tmp_path, "Ef*_et*_", ".nxs.ng0", "271,244")
    plan = resolve_source_selection(selection)
    assert plan.resolved_files == (str(first), str(second)) and not plan.missing
    (tmp_path / "Ef5p0_et_1.1_271.nxs.ng0").touch()
    with pytest.raises(ValueError, match="ambiguous source pattern for run 271"):
        resolve_source_selection(selection)


def test_suffix_wildcards_preserve_order_padding_repeats_and_missing_sources(tmp_path):
    first = tmp_path / "SEQ_001_a.nxs"
    second = tmp_path / "SEQ_002_b.nxs.h5"
    first.touch()
    second.touch()
    plan = resolve_source_selection(SourceSelection(tmp_path, "SEQ_", "_*.nxs*", "2,1,2,3", padding=3))
    assert [item.path for item in plan.appearances[:3]] == [str(second), str(first), str(second)]
    assert plan.resolved_files == (str(second), str(first))
    assert plan.duplicates == (str(second),)
    assert plan.missing == (str(tmp_path / "SEQ_003_*.nxs*"),)


def _triples(expression, **kwargs):
    return [(item.run_number, item.depth, item.stack) for item in parse_run_expression(expression, **kwargs)]


@pytest.mark.parametrize("expression, expected", [
    ("10", [(10, 1, 1)]),
    ("10,12,15", [(10, 1, 1), (12, 2, 1), (15, 3, 1)]),
    ("10+12+15", [(10, 1, 1), (12, 1, 1), (15, 1, 1)]),
    ("10:13", [(10, 1, 1), (11, 2, 1), (12, 3, 1), (13, 4, 1)]),
    ("13:10", [(13, 1, 1), (12, 2, 1), (11, 3, 1), (10, 4, 1)]),
    ("10>13", [(10, 1, 1), (11, 1, 1), (12, 1, 1), (13, 1, 1)]),
    ("10>>>19", [(10, 1, 1), (13, 1, 1), (16, 1, 1), (19, 1, 1)]),
    ("10:::19", [(10, 1, 1), (13, 2, 1), (16, 3, 1), (19, 4, 1)]),
    ("10:3:19", [(10, 1, 1), (13, 2, 1), (16, 3, 1), (19, 4, 1)]),
    ("19:-3:10", [(19, 1, 1), (16, 2, 1), (13, 3, 1), (10, 4, 1)]),
    ("10:3:18", [(10, 1, 1), (13, 2, 1), (16, 3, 1)]),
    ("10'12'15", [(10, 1, 1), (12, 1, 2), (15, 1, 3)]),
    ("10;13", [(10, 1, 1), (11, 1, 2), (12, 1, 3), (13, 1, 4)]),
    ("10;;;19", [(10, 1, 1), (13, 1, 2), (16, 1, 3), (19, 1, 4)]),
    ("10|3|", [(10, 1, 1), (10, 2, 1), (10, 3, 1)]),
    ("10!3!", [(10, 1, 1), (10, 1, 2), (10, 1, 3)]),
    ("0|3|", [(None, 1, 1), (None, 2, 1), (None, 3, 1)]),
    ("0!2!", [(None, 1, 1), (None, 1, 2)]),
    ("10'12+13,20", [(10, 1, 1), (12, 1, 2), (13, 1, 2), (20, 2, 1)]),
    ("10>12,20::24", [(10, 1, 1), (11, 1, 1), (12, 1, 1), (20, 2, 1), (22, 3, 1), (24, 4, 1)]),
    ("10:3:19,20:21", [(10, 1, 1), (13, 2, 1), (16, 3, 1), (19, 4, 1), (20, 5, 1), (21, 6, 1)]),
])
def test_lists_ranges_strides_sum_and_two_grouping_indices(expression, expected):
    assert _triples(expression) == expected


@pytest.mark.parametrize("brackets, slots", [
    ("(2;1;3;2)", [(1, 1), (1, 1), (1, 1), (1, 1)]),
    ("[2;1;3;2]", [(1, 1), (1, 1), (2, 1), (2, 1)]),
    ("{2;1;3;2}", [(1, 1), (2, 1), (3, 1), (4, 1)]),
    ("/2;1;3;2/", [(1, 1), (1, 2), (1, 3), (1, 4)]),
])
def test_block_skips_repeat_and_first_slot_correction(brackets, slots):
    items = parse_run_expression("100" + brackets)
    assert [item.run_number for item in items] == [100, 102, 106, 108]
    assert [item.slot for item in items] == slots


def test_block_defaults_chaining_and_existing_slot_members():
    assert _triples("10[2]") == [(10, 1, 1), (11, 1, 1)]
    assert _triples("10[2;1]") == [(10, 1, 1), (12, 1, 1)]
    assert _triples("10[2;1;3]") == [(10, 1, 1), (12, 1, 1)]
    assert _triples("1+10[2;0;0;2],20") == [(1, 1, 1), (10, 1, 1), (11, 1, 1),
                                          (12, 2, 1), (13, 2, 1), (20, 3, 1)]
    assert _triples("1'10/2;0;0;2/;15") == [(1, 1, 1), (10, 1, 2), (11, 1, 3),
                                         (12, 1, 4), (13, 1, 5), (14, 1, 6), (15, 1, 7)]
    assert _triples("10[2;0;0;2]>15") == [(10, 1, 1), (11, 1, 1), (12, 2, 1),
                                       (13, 2, 1), (14, 2, 1), (15, 2, 1)]


def test_clear_expression_and_whitespace():
    assert parse_run_expression(" x ") == ()
    assert parse_run_expression("X") == ()
    assert _triples(" 10 : 2 : 14 , 20 + 21 ") == [(10, 1, 1), (12, 2, 1), (14, 3, 1), (20, 4, 1), (21, 4, 1)]


@pytest.mark.parametrize("expression", [
    "", " ", "-1", "1,", "1++2", "1,,2", "1:0:10", "1:-2:10", "10:2:1",
    "1:-2", "1[2", "1[0]", "1[2;0;0;0]", "1[1;-1]", "1[1;0;0;1;2]",
    "1|0|", "1!2;1!", "x,1", "1x", "1.5", "1,eval(1)", "1,__import__('os')", "1..5", "1:2:3:4",
])
def test_malformed_or_ambiguous_input_is_rejected(expression):
    with pytest.raises(ValueError, match="run expression|enter at least"):
        parse_run_expression(expression)


@pytest.mark.parametrize("expression", ["1:100001", "1>100001", "1;100001", "1|100001|", "1!100001!",
                                       "1[1000;0;0;101]", "1/100001/", "1:" + "9"*64])
def test_expansion_limit_is_checked_without_allocating_large_ranges(expression):
    with pytest.raises(ValueError, match="expansion limit"):
        parse_run_expression(expression)
    assert len(parse_run_expression("1:5", max_appearances=5)) == 5
    with pytest.raises(ValueError, match="expansion limit"):
        parse_run_expression("1:6", max_appearances=5)


def test_custom_expansion_budget_is_bounded():
    for limit in (0, -1, True, 1.5, MAX_SOURCE_APPEARANCES + 1):
        with pytest.raises(ValueError, match="maximum appearances"):
            parse_run_expression("1", max_appearances=limit)


def test_file_preview_padding_missing_duplicates_and_empty_slots(tmp_path):
    for numor in (1, 2):
        (tmp_path / f"SEQ_{numor:04}.nxs.h5").write_bytes(b"source")
    selection = SourceSelection(tmp_path, "SEQ_", ".nxs.h5", "1+2,3,1,0|2|", padding=4)
    plan = resolve_source_selection(selection)
    first, second, missing = [str((tmp_path / f"SEQ_{number:04}.nxs.h5").resolve()) for number in (1, 2, 3)]
    assert plan.resolved_files == (first, second)
    assert plan.missing == (missing,)
    assert plan.duplicates == (first,)
    assert plan.slots[0].run_numbers == (1, 2)
    assert plan.slots[0].appearance_indices == (0, 1)
    assert plan.slots[-1].empty and plan.slots[-2].empty
    assert plan.appearances[-1].path is None
    assert not plan.ready
    assert not plan.clear
    serialized = json.loads(json.dumps(plan.to_dict()))
    assert serialized["expression"] == selection.expression
    assert serialized["slots"][0]["run_numbers"] == [1, 2]
    assert SourceSelection.from_dict(serialized) == selection


def test_metadata_is_opt_in_once_per_physical_source_and_missing_not_read(tmp_path):
    (tmp_path / "1.nxs").write_bytes(b"events")
    selection = SourceSelection(tmp_path, "", ".nxs", "1|3|,2")
    calls = []

    def reader(path):
        calls.append(path)
        return {"instrument": "SEQUOIA", "Ei_meV": 60}

    assert resolve_source_selection(selection, metadata_reader=reader).metadata == {}
    assert calls == []
    plan = resolve_source_selection(selection, inspect_metadata=True, metadata_reader=reader)
    assert len(calls) == 1
    assert plan.metadata[plan.resolved_files[0]]["Ei_meV"] == 60


def test_symlinks_are_the_same_source_identity(tmp_path):
    physical = tmp_path / "1.nxs"
    physical.write_bytes(b"events")
    (tmp_path / "2.nxs").symlink_to(physical)
    plan = resolve_source_selection(SourceSelection(tmp_path, "", ".nxs", "1,2"))
    assert plan.ready
    assert plan.resolved_files == (str(physical.resolve()),)
    assert plan.duplicates == plan.resolved_files
    assert plan.appearances[0].source_identity == plan.appearances[1].source_identity


def test_empty_and_clear_previews_do_not_create_files(tmp_path):
    plan = resolve_source_selection(SourceSelection(tmp_path, "", ".nxs", "0|3|"))
    assert len(plan.slots) == 3
    assert all(slot.empty for slot in plan.slots)
    assert plan.resolved_files == plan.missing == plan.duplicates == ()
    cleared = resolve_source_selection(SourceSelection(tmp_path, "", ".nxs", "x"))
    assert cleared.clear and not cleared.ready and not cleared.appearances
    assert not list(tmp_path.iterdir())


def test_default_hdf_header_preview_reads_scalar_metadata_not_event_arrays(tmp_path):
    import h5py

    path = tmp_path / "1.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry["run_number"] = [123]
        entry["title"] = [b"test"]
        entry["instrument/name"] = [b"SEQUOIA"]
        entry["DASlogs/EnergyRequest/average_value"] = [60.0]
        entry["DASlogs/T0/average_value"] = [-3.5]
        entry["DASlogs/SampleTemp/average_value"] = [5.0]
        entry["DASlogs/temperature/value"] = list(range(10000))
        bank = entry.create_group("bank1_events")
        bank.create_dataset("event_id", shape=(10000,), dtype="int64")
    plan = resolve_source_selection(SourceSelection(tmp_path, "", ".nxs", "1"), inspect_metadata=True)
    metadata = plan.metadata[str(path)]
    assert metadata["run_number"] == 123
    assert metadata["title"] == "test"
    assert metadata["event_count"] == 10000
    assert metadata["instrument_name"] == "SEQUOIA"
    assert metadata["incident_energy_meV"] == 60.0
    assert metadata["t0_microseconds"] == -3.5
    assert metadata["temperature_K"] == 5.0
    json.dumps(plan.to_dict())


def test_saved_numors_compatibility_and_naming_validation(tmp_path):
    selector = SourceSelection.from_dict({"directory": str(tmp_path), "numors": "1:3", "prefix": "SEQ_"})
    assert selector.expression == "1:3" and selector.padding == 0
    assert RUN_EXPRESSION_HELP and str(MAX_SOURCE_APPEARANCES//1000) in RUN_EXPRESSION_HELP
    for padding in (-1, True, 1.5, MAX_SOURCE_NUMBER_PADDING + 1):
        with pytest.raises(ValueError, match="padding"):
            SourceSelection(tmp_path, "", "", "1", padding)
    for prefix, suffix in (("../", ".nxs"), ("", "/run"), ("run\\", ".nxs")):
        with pytest.raises(ValueError, match="filename fragment"):
            SourceSelection(tmp_path, prefix, suffix, "1")
    with pytest.raises(ValueError, match="directory"):
        SourceSelection("", "", "", "1")
    with pytest.raises(TypeError, match="text"):
        SourceSelection(tmp_path, "", "", 123)


def test_preview_missing_directory_and_header_errors_are_actionable(tmp_path):
    missing = resolve_source_selection(SourceSelection(tmp_path / "missing", "", ".nxs", "1:2"))
    assert len(missing.missing) == 2 and not missing.resolved_files
    path = tmp_path / "1.nxs"
    path.write_bytes(b"not HDF5")
    plan = resolve_source_selection(SourceSelection(tmp_path, "", ".nxs", "1"), inspect_metadata=True)
    assert plan.metadata[str(path)]["header_status"] == "not a readable HDF5 source"
    assert Path(plan.appearances[0].path).exists()
