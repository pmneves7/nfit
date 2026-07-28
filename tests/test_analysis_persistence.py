import numpy as np

from nfit.analysis import (
    AnalysisContext,
    AnalysisEntry,
    AnalysisExecution,
    AnalysisInput,
    AnalysisOperationDefinition,
    AnalysisOutputRef,
    AnalysisResultRecord,
    TableOutput,
    register_analysis_operation,
)
from nfit.analysis.artifacts import read_dataset_artifact, write_dataset_artifact
from nfit.analysis.runner import analysis_is_fresh, execute_to_artifacts
from nfit.dataset import PointData4D, PointListData
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.pipeline import DataGroup, DatasetEntry
from nfit.project_gui import (
    NfitProject,
    _copy_analysis_assets_for_save_as,
    _project_from_dict,
    _project_to_dict,
)


def test_project_v3_round_trips_analysis_ids_and_results():
    dataset = DatasetEntry("scan", None, metadata={"source_file": "scan.npz"})
    analysis = AnalysisEntry("QFI", "spectral_integration", [dataset.id], {"kernel": "qfi"})
    analysis.result = AnalysisResultRecord(
        "recipe", {dataset.id: "input"},
        [AnalysisOutputRef("qfi", "QFI", "dataset", "assets/qfi.npz", "b" * 32)],
        "success", "2026-07-14T12:00:00Z",
    )
    payload = _project_to_dict(NfitProject(data_groups=[DataGroup("group", datasets=[dataset], analyses=[analysis])]))
    restored = _project_from_dict(payload).data_groups[0]

    assert payload["version"] == 3
    assert restored.datasets[0].id == dataset.id
    assert restored.analyses[0] == analysis


def test_version_one_project_generates_dataset_id():
    payload = {
        "format": "nfit-project", "version": 1, "settings": {},
        "data_groups": [{"name": "group", "datasets": [{"name": "scan"}]}],
    }
    restored = _project_from_dict(payload)
    assert len(restored.data_groups[0].datasets[0].id) == 32


def test_duplicate_dataset_ids_are_repaired_deterministically():
    payload = {"format": "nfit-project", "version": 2, "settings": {}, "data_groups": [{"name": "group", "datasets": [{"name": "a", "id": "f" * 32}, {"name": "b", "id": "f" * 32}]}]}
    first = _project_from_dict(payload).data_groups[0]
    second = _project_from_dict(payload).data_groups[0]
    assert first.datasets[0].id != first.datasets[1].id
    assert first.datasets[1].id == second.datasets[1].id
    assert first.metadata["project_load_warnings"]


def test_point_list_artifact_round_trip(tmp_path):
    data = PointListData(
        {"Q": [1.0, 2.0], "QFI": [0.2, 0.3], "dQFI": [0.01, 0.02]},
        units={"Q": "1/angstrom"}, coordinate_names=["Q"],
        channels=[{"label": "QFI", "value": "QFI", "error": "dQFI"}],
        metadata={"kernel": "qfi"},
        quantity_types={"Q": "momentum", "QFI": "dynamic_susceptibility", "dQFI": "dynamic_susceptibility"},
    )
    path = tmp_path / "qfi.npz"
    write_dataset_artifact(data, path)
    restored = read_dataset_artifact(path)
    assert restored.channel_values("QFI").tolist() == [0.2, 0.3]
    assert restored.metadata == {"kernel": "qfi"}
    assert restored.quantity_type("Q") == "momentum"


def test_mdhisto_artifact_preserves_axes_and_auxiliary_channels(tmp_path):
    axes = (MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum", frame="HKL"), MDHistoAxis("K", np.array([0.0, 1.0]), "rlu", "momentum"))
    shape = (2, 1)
    data = MDHistoData(axes, np.ones(shape), np.ones(shape), np.zeros(shape, bool), np.ones(shape), coordinate_system=1, auxiliary_channels={"coverage": MDHistoChannel(np.full(shape, 0.5), label="Coverage", unit="fraction", quantity_type="scattering_intensity")})
    path = tmp_path / "map.npz"
    write_dataset_artifact(data, path)
    restored = read_dataset_artifact(path)
    assert restored.axes[0].frame == "HKL"
    assert restored.coordinate_system == 1
    assert restored.auxiliary_channels["coverage"].label == "Coverage"
    assert restored.auxiliary_channels["coverage"].quantity_type == "scattering_intensity"


def test_runner_publishes_artifact_and_fresh_result(tmp_path):
    key = "persistence_test_operation"
    table = PointListData({"H": [1.0], "I": [2.0]}, coordinate_names=["H"], channels=[{"label": "I", "value": "I", "error": None}])
    register_analysis_operation(AnalysisOperationDefinition(key, "Test", 2, "Test", 1, 1, ("PointData4D",), (), lambda *_: None, lambda *_args, **_kwargs: AnalysisExecution({"table": TableOutput(table, "Table")})))
    data = PointData4D([0], [0], [0], [1], [2], [0.1])
    item = AnalysisInput("a" * 32, "scan", data, AnalysisContext("g", None, None, None, None), "fingerprint")
    analysis = AnalysisEntry("Analysis", key, [item.dataset_id], {})
    project = tmp_path / "sample.nfit"
    project.write_text("{}")

    result = execute_to_artifacts(analysis, [item], project)
    analysis.result = result

    assert analysis.operation_version == 2
    assert analysis_is_fresh(analysis, [item])
    artifact = tmp_path / result.outputs[0].artifact_path
    assert artifact.exists()
    assert read_dataset_artifact(artifact).column("I")[0] == 2.0


def test_save_as_copies_and_retargets_analysis_assets(tmp_path):
    old = tmp_path / "old.nfit"
    new = tmp_path / "sub" / "new.nfit"
    artifact = tmp_path / "old.nfit-assets" / "analyses" / ("a" * 32) / "table.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    dataset = DatasetEntry("derived", None, metadata={"source_file": str(artifact), "derived_from_analysis": {"analysis_id": "a" * 32}})
    analysis = AnalysisEntry("Analysis", "bragg_integration", [], {}, id="a" * 32, result=AnalysisResultRecord("r", {}, [AnalysisOutputRef("table", "Table", "table", "old.nfit-assets/analyses/" + "a" * 32 + "/table.npz")], "success", "now"))
    project = NfitProject([DataGroup("group", datasets=[dataset], analyses=[analysis])])
    new.parent.mkdir()

    _copy_analysis_assets_for_save_as(project, old, new)

    assert (new.parent / "new.nfit-assets" / "analyses" / ("a" * 32) / "table.npz").read_bytes() == b"artifact"
    assert analysis.result.outputs[0].artifact_path.startswith("new.nfit-assets/")
    assert dataset.metadata["source_file"].startswith(str(new.parent / "new.nfit-assets"))


def test_derived_source_serializes_as_relative_artifact_path(tmp_path):
    relative = "sample.nfit-assets/analyses/a/table.npz"
    dataset = DatasetEntry("derived", None, metadata={"source_file": str(tmp_path / relative), "analysis_artifact_path": relative, "derived_from_analysis": {"analysis_id": "a"}})
    payload = _project_to_dict(NfitProject([DataGroup("group", datasets=[dataset])]))
    assert payload["data_groups"][0]["datasets"][0]["metadata"]["source_file"] == relative
