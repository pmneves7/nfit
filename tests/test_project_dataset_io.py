from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import nfit.project_data as project_data
import nfit.project_dataset_io as project_dataset_io
from nfit.dataset import PointListData
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.pipeline import DatasetEntry

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"


def test_project_data_reexports_dataset_io_service_objects() -> None:
    names = (
        "save_dataset_file",
        "_save_point_list_file",
        "_load_nfit_dataset_file",
        "_load_nfit_mdhisto_archive",
        "_load_nfit_point_list_archive",
        "_nfit_archive_text",
        "_nfit_archive_optional_int",
        "_nfit_archive_json_mapping",
        "_json_safe_value",
    )
    for name in names:
        assert getattr(project_data, name) is getattr(project_dataset_io, name)


def test_dataset_io_service_round_trips_mdhisto_data_and_context(tmp_path) -> None:
    data = MDHistoData(
        axes=(
            MDHistoAxis(
                "Energy", np.asarray([0.0, 1.0, 2.0]), "meV", "energy"
            ),
        ),
        signal=np.asarray([2.0, 4.0]),
        errors=np.asarray([0.2, 0.4]),
        mask=np.asarray([False, True]),
        num_events=np.asarray([3.0, 0.0]),
        metadata={"signal_semantics": "density", "array": np.asarray([1, 2])},
    )
    dataset = DatasetEntry(
        "scan", data, parameters={"temperature": 5.0, "magnetic_field": None}
    )
    path = tmp_path / "scan.npz"

    project_dataset_io.save_dataset_file(dataset, path, use_view=False)
    restored, context = project_dataset_io._load_nfit_dataset_file(path)

    assert isinstance(restored, MDHistoData)
    np.testing.assert_allclose(restored.signal, data.signal)
    np.testing.assert_array_equal(restored.mask, data.mask)
    assert restored.axes[0].name == "Energy"
    assert restored.metadata["array"] == [1, 2]
    assert context == {"temperature": 5.0, "magnetic_field": None}


def test_dataset_io_service_round_trips_point_list_data(tmp_path) -> None:
    data = PointListData(
        {"x": [1.0, 2.0], "signal": [3.0, 4.0]},
        units={"x": "K", "signal": "emu"},
        coordinate_names=["x"],
        channels=[{"label": "signal", "value": "signal"}],
        metadata={"nested": {"array": np.asarray([1.0])}},
    )
    path = tmp_path / "points.npz"

    project_dataset_io.save_dataset_file(
        DatasetEntry("points", data), path, use_view=False
    )
    restored, context = project_dataset_io._load_nfit_dataset_file(path)

    assert isinstance(restored, PointListData)
    np.testing.assert_allclose(restored.column("signal"), [3.0, 4.0])
    assert restored.coordinate_names == ["x"]
    assert restored.metadata["nested"] == {"array": [1.0]}
    assert context == {}


def test_dataset_io_preserves_grid_normalization_without_json_expansion(tmp_path) -> None:
    axis = MDHistoAxis("Energy", np.asarray([0.0, 1.0, 2.0]), "meV", "energy")
    denominator = np.asarray([4.0, 0.0])
    denominator.setflags(write=False)
    common = dict(
        axes=(axis,), signal=np.asarray([2.0, 0.0]),
        errors=np.asarray([0.2, 0.0]), mask=np.asarray([False, True]),
        num_events=np.asarray([3.0, 0.0]),
        metadata={
            "signal_semantics": "density",
            "normalization_denominator": denominator,
            "zero_event_bins_are_measured": True,
        },
    )
    with_channel = MDHistoData(
        **common,
        auxiliary_channels={
            "normalization_denominator": MDHistoChannel(
                denominator, label="Normalization"
            )
        },
    )
    without_channel = MDHistoData(**common)

    for label, data in (("shared", with_channel), ("separate", without_channel)):
        path = tmp_path / f"{label}.npz"
        project_dataset_io.save_dataset_file(
            DatasetEntry(label, data), path, use_view=False
        )
        with np.load(path, allow_pickle=False) as archive:
            assert "normalization_denominator" not in str(
                archive["metadata_json"].item()
            )
            assert (
                "normalization_denominator_auxiliary_channel" in archive
            ) == (label == "shared")
            assert ("normalization_denominator" in archive) == (
                label == "separate"
            )
        restored, _ = project_dataset_io._load_nfit_dataset_file(path)
        np.testing.assert_array_equal(
            restored.metadata["normalization_denominator"], denominator
        )
        if label == "shared":
            assert np.shares_memory(
                restored.metadata["normalization_denominator"],
                restored.auxiliary_channels["normalization_denominator"].values,
            )


def test_dataset_io_service_has_no_gui_dependency() -> None:
    tree = ast.parse((PACKAGE_ROOT / "project_dataset_io.py").read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any(
        "PySide" in module or module.startswith("qt_") for module in imported_modules
    )
    assert "project_gui" not in imported_modules


def _archive_example(container: str):
    from nfit.histogram_statistics import (
        EVENT_STATISTICS_KEY,
        EVENT_STATISTICS_METADATA,
        event_statistics_channels,
    )
    from nfit.measurement_dependencies import (
        CountingDependencies,
        independent_source_dependencies,
        ratio_source_dependencies,
    )

    if container == "point_list":
        return PointListData(
            {"x": [1.0, 2.0], "signal": [3.0, 4.0], "error": [0.3, 0.4]},
            units={"x": "K", "signal": "emu", "error": "emu"},
            coordinate_names=["x"],
            channels=[{"label": "Moment", "value": "signal", "error": "error"}],
            quantity_types={"signal": "magnetic_moment"},
            metadata={"nested": {"array": np.asarray([1.0])}},
        )
    if container == "point4d":
        from nfit.dataset import PointData4D

        return PointData4D(
            H=[1, 2], K=[0, 0], L=[0, 0], E=[3, 4],
            intensity=[4, 5], sigma=[2, 3], mask=[True, False],
            temperature=[5, 10], magnetic_field=[0, 0, 1],
            normalization_denominator=[2, 4],
            measurement_payload={"measurement_value_numerator": [8, 20]},
            source_dependencies=independent_source_dependencies(np.array([4., 9.]), "points"),
            metadata={"nested": {"array": np.asarray([1.0])}},
        )
    numerator, exposure = np.array([10., 4.]), np.array([2., 2.])
    bundle = CountingDependencies(
        numerator_dependencies=independent_source_dependencies(numerator, "counts"),
        exposure_dependencies=independent_source_dependencies(np.array([.04, .04]), "exposure"),
    )
    primary = ratio_source_dependencies(bundle, numerator, exposure)
    channels = event_statistics_channels(numerator, numerator, exposure)
    return MDHistoData(
        axes=(MDHistoAxis("Energy", np.array([0., 1., 2.]), "meV", "energy",
                          frame="sample", path="/axes/energy", metadata={"label": "E"}),),
        signal=numerator/exposure, errors=np.sqrt(primary.variance()),
        mask=np.array([False, True]), num_events=numerator,
        coordinate_system=1, visual_normalization=2,
        metadata={EVENT_STATISTICS_KEY: EVENT_STATISTICS_METADATA,
                  "signal_semantics": "density", "nested": {"array": np.asarray([1.0])},
                  "normalization_denominator": channels["normalization_denominator"].values},
        auxiliary_channels=channels, counting_dependencies=bundle,
    )


def _assert_dependencies_equal(actual, expected) -> None:
    assert actual.shape == expected.shape
    assert actual.source_ids == expected.source_ids
    for name in ("source_variances", "observation_indices", "source_indices", "coefficients"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("container", ["mdhisto", "point4d", "point_list"])
@pytest.mark.parametrize("use_view", [False, True])
def test_archive_codec_matches_project_save_and_retains_payload(tmp_path, container, use_view) -> None:
    from nfit.dataset import PointData4D
    from nfit.dataset_archive import write_dataset_archive

    source = _archive_example(container)
    dataset = DatasetEntry("scan", source, scale_factor=3,
                           parameters={"temperature": 12.5, "magnetic_field": [0, 0, 1],
                                       "kf_ki_included": True, "unrelated": "omit"})
    selected = project_data.dataset_for_slice_viewer(dataset) if use_view else source
    context = {key: dataset.parameters[key] for key in ("temperature", "magnetic_field", "kf_ki_included")}
    direct_path, project_path = tmp_path / "direct.npz", tmp_path / "project.npz"
    write_dataset_archive(selected, direct_path, context=context)
    project_dataset_io.save_dataset_file(dataset, project_path, use_view=use_view)

    # Both public routes must write exactly the same arrays, JSON and ZIP bytes.
    assert direct_path.read_bytes() == project_path.read_bytes()
    restored, saved_context = project_dataset_io._load_nfit_dataset_file(project_path)
    assert restored.metadata["nested"] == {"array": [1.0]}
    assert saved_context == ({} if container == "point_list" else context)
    factor = 3 if use_view else 1
    with np.load(project_path, allow_pickle=False) as archive:
        assert str(archive["nfit_data_container"].item()) == container
        assert int(archive["nfit_dataset_version"].item()) == (3 if container == "mdhisto" else 1)
    if isinstance(restored, MDHistoData):
        np.testing.assert_array_equal(restored.signal, source.signal * factor)
        np.testing.assert_array_equal(restored.errors, source.errors * factor)
        np.testing.assert_array_equal(restored.mask, selected.mask)
        np.testing.assert_array_equal(restored.num_events, selected.num_events)
        assert restored.coordinate_system == 1
        assert restored.visual_normalization == 2
        assert restored.axes[0].frame == "sample"
        assert restored.axes[0].path == "/axes/energy"
        assert restored.axes[0].metadata == {"label": "E"}
        for name, channel in selected.auxiliary_channels.items():
            np.testing.assert_array_equal(restored.auxiliary_channels[name].values, channel.values)
        assert np.shares_memory(restored.metadata["normalization_denominator"],
                                restored.auxiliary_channels["normalization_denominator"].values)
        _assert_dependencies_equal(restored.source_dependencies, selected.source_dependencies)
        for name in ("numerator_dependencies", "exposure_dependencies"):
            _assert_dependencies_equal(getattr(restored.counting_dependencies, name),
                                       getattr(selected.counting_dependencies, name))
    elif isinstance(restored, PointData4D):
        for name in ("H", "K", "L", "E", "mask", "temperature", "magnetic_field", "normalization_denominator"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(selected, name))
        np.testing.assert_array_equal(restored.intensity, source.intensity * factor)
        np.testing.assert_array_equal(restored.sigma, source.sigma * factor)
        for name, values in selected.measurement_payload.items():
            np.testing.assert_array_equal(restored.measurement_payload[name], values)
        _assert_dependencies_equal(restored.source_dependencies, selected.source_dependencies)
    else:
        assert restored.channels == selected.channels
        assert restored.units == selected.units
        assert restored.quantity_types == selected.quantity_types
        assert restored.coordinate_names == selected.coordinate_names
        for name in selected.column_names:
            np.testing.assert_array_equal(restored.column(name), selected.column(name))
        np.testing.assert_array_equal(restored.column("signal"), source.column("signal") * factor)


def test_raw_save_skips_preparation_and_default_save_prepares(tmp_path, monkeypatch) -> None:
    source = _archive_example("point_list")
    prepared = source.with_updates(columns={"x": [1, 2], "signal": [30, 40], "error": [3, 4]})
    calls = []

    def prepare(dataset):
        calls.append(dataset)
        return prepared

    monkeypatch.setattr(project_data, "dataset_for_slice_viewer", prepare)
    dataset = DatasetEntry("scan", source)
    raw, view = tmp_path / "raw.npz", tmp_path / "view.npz"
    project_dataset_io.save_dataset_file(dataset, raw, use_view=False)
    assert calls == []
    project_dataset_io.save_dataset_file(dataset, view)
    assert calls == [dataset]
    restored, _ = project_dataset_io._load_nfit_dataset_file(view)
    np.testing.assert_array_equal(restored.column("signal"), [30, 40])


def test_archive_codec_retains_binary_background_recovery_metadata(tmp_path) -> None:
    from nfit.background_channels import BACKGROUND_EXCEPTIONS_KEY
    from nfit.dataset_archive import write_dataset_archive

    data = _archive_example("mdhisto")
    exceptions = {"indices": np.array([1]), "signal": np.array([7.]), "errors": np.array([.5])}
    data = data.with_updates(metadata={**data.metadata, BACKGROUND_EXCEPTIONS_KEY: exceptions})
    path = tmp_path / "background.npz"
    write_dataset_archive(data, path)
    with np.load(path, allow_pickle=False) as archive:
        assert BACKGROUND_EXCEPTIONS_KEY not in str(archive["metadata_json"].item())
        assert "background_original_fields_json" in archive
    restored, _ = project_dataset_io._load_nfit_dataset_file(path)
    for name, values in exceptions.items():
        np.testing.assert_array_equal(restored.metadata[BACKGROUND_EXCEPTIONS_KEY][name], values)
        assert not restored.metadata[BACKGROUND_EXCEPTIONS_KEY][name].flags.writeable


@pytest.mark.parametrize("first", ["project_dataset_io", "project_data"])
def test_archive_save_compatibility_import_order_and_public_identity(first) -> None:
    code = f"""
import importlib
first = importlib.import_module('nfit.{first}')
from nfit.project_dataset_io import save_dataset_file
from nfit.project_dataset_io import *
import nfit
from nfit import project_data, project_dataset_io, project_gui
assert save_dataset_file is project_data.save_dataset_file
assert save_dataset_file is project_dataset_io.save_dataset_file
assert save_dataset_file is project_gui.save_dataset_file
assert save_dataset_file is nfit.save_dataset_file
assert 'save_dataset_file' in dir(project_dataset_io)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_archive_codec_and_consumers_do_not_depend_on_preparation_facade() -> None:
    codec_tree = ast.parse((PACKAGE_ROOT / "dataset_archive.py").read_text())
    codec_imports = [node.module or "" for node in ast.walk(codec_tree) if isinstance(node, ast.ImportFrom)]
    codec_imports.extend(alias.name for node in ast.walk(codec_tree) if isinstance(node, ast.Import) for alias in node.names)
    assert not any(module.startswith(("project_", "qt_", "PySide", "PyQt")) for module in codec_imports)
    for filename in ("project_data", "project_resources", "measurement_rebinning", "measurement_replay",
                     "composite_workflow", "viewer_export", "source_selection_imports"):
        tree = ast.parse((PACKAGE_ROOT / f"{filename}.py").read_text())
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert "dataset_archive" in modules
        assert "project_dataset_io" not in modules


def test_archive_codec_can_write_and_load_without_project_preparation(tmp_path) -> None:
    code = f"""
import importlib
import sys
import types
from pathlib import Path
import numpy as np

# Isolate the service's imports from nfit.__init__, which intentionally exports
# project preparation APIs. Dependencies use their normal package initializers.
package = types.ModuleType('nfit')
package.__path__ = [{str(PACKAGE_ROOT)!r}]
sys.modules['nfit'] = package
codec = importlib.import_module('nfit.dataset_archive')
compat = importlib.import_module('nfit.project_dataset_io')
assert compat._load_nfit_dataset_file is codec._load_nfit_dataset_file
from nfit.dataset import PointData4D, PointListData
from nfit.mdhisto import MDHistoAxis, MDHistoData
payloads = [
    PointData4D(H=[0], K=[0], L=[0], E=[1], intensity=[2], sigma=[1]),
    PointListData({{'x': [1], 'signal': [2]}}, coordinate_names=['x'],
                  channels=[{{'value': 'signal'}}]),
    MDHistoData(axes=(MDHistoAxis('E', np.array([0., 1.]), 'meV', 'energy'),),
                signal=np.array([2.]), errors=np.array([1.]),
                mask=np.array([False]), num_events=np.array([2.]),
                metadata={{'signal_semantics': 'density'}}),
]
for index, data in enumerate(payloads):
    path = Path({str(tmp_path)!r}) / f'{{index}}.npz'
    codec.write_dataset_archive(data, path)
    restored, _ = codec._load_nfit_dataset_file(path)
    assert type(restored) is type(data)
assert 'nfit.project_data' not in sys.modules
assert 'nfit.project_gui' not in sys.modules
assert not any(name.startswith(('PySide', 'PyQt')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
