import copy

import pytest

from nfit import DataGroup, DatasetEntry, PointData4D


def test_data_group_selects_raw_and_prepared_data():
    data_a = PointData4D([0.0], [0.0], [0.0], [1.0], [2.0], [0.1])
    data_b = PointData4D([0.0], [0.0], [0.0], [1.0], [4.0], [0.1])
    data_b_prepared = PointData4D(
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        [4.0],
        [0.1],
        mask=[True],
    )
    group = DataGroup(
        name="field_series",
        lattice_parameters={"a": 3.8, "b": 3.8, "c": 12.0},
        spacegroup="P4/mmm",
        datasets=[
            DatasetEntry("low_field", data_a, kind="neutron", parameters={"field": 0.0}),
            DatasetEntry(
                "high_field",
                data_b,
                kind="neutron",
                parameters={"field": 9.0},
                transforms=[lambda _data: data_b_prepared],
            ),
        ],
    )
    assert group.dataset_names == ["low_field", "high_field"]
    assert group.data_sequence(["high_field"]) == [data_b]
    assert group.prepared_sequence(["high_field"]) == [data_b_prepared]


def test_dataset_replacement_advances_revision_and_freezes_arrays(tmp_path):
    source = tmp_path / "scan.npz"
    source.touch()
    original = PointData4D([0.0], [0.0], [0.0], [1.0], [2.0], [0.1])
    dataset = DatasetEntry(
        "scan",
        original,
        metadata={"source_file": str(source)},
    )

    assert dataset.data_matches_source
    assert dataset.data_revision == 0
    editable = dataset.data.mutable_copy()
    editable.intensity[0] = 3.0
    installed = dataset.replace_data(editable)

    assert dataset.data_revision == 1
    assert not dataset.data_matches_source
    assert installed is dataset.data
    assert installed.intensity[0] == 3.0
    assert not installed.intensity.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        installed.intensity[0] = 4.0


def test_dataset_detects_unsupported_direct_data_replacement(tmp_path):
    source = tmp_path / "scan.npz"
    source.touch()
    dataset = DatasetEntry(
        "scan",
        PointData4D([0.0], [0.0], [0.0], [1.0], [2.0], [0.1]),
        metadata={"source_file": str(source)},
    )

    dataset.data = PointData4D([0.0], [0.0], [0.0], [1.0], [9.0], [0.1])

    assert not dataset.data_matches_source


def test_dataset_copy_recanonicalizes_arrays_and_preserves_source_state(tmp_path):
    source = tmp_path / "scan.npz"
    source.touch()
    original = DatasetEntry(
        "scan",
        PointData4D([0.0], [0.0], [0.0], [1.0], [2.0], [0.1]),
        metadata={"source_file": str(source)},
    )

    copied = copy.deepcopy(original).copy(name="copy")

    assert copied.id != original.id
    assert copied.data_matches_source
    assert not copied.data.intensity.flags.writeable
