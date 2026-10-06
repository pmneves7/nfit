"""Automatic native extents use reduced-event caches and scientific identities."""

import numpy as np
import pytest

from nfit import DataGroup, bin_raw_dgs_group, raw_dgs_coordinate_bounds, raw_dgs_dataset_group
from nfit.project_data import _derived_shared_grid_config
from nfit.reduction_recipes import set_reduction_settings
from tests.test_raw_dgs import _rewrite_instrument_xml, _write_raw_dgs


@pytest.fixture
def group(tmp_path):
    path = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(path)
    result = raw_dgs_dataset_group([path])
    set_reduction_settings(
        result,
        {
            "bad_pulse_threshold": 0.0,
            "ki_kf_normalization": False,
            "he3_detector_efficiency_correction": False,
            "ub_matrix": np.eye(3).tolist(),
        },
    )
    return result


def test_bounds_use_cached_events_and_retain_rotation_and_ub(group, monkeypatch):
    from nfit import raw_dgs

    bin_raw_dgs_group(group, lower=[-10] * 3 + [-19], upper=[10] * 3 + [19], num_bins=[1] * 4)
    with group.datasets[0]._raw_dgs_reduction_cache.open() as cache:
        events = np.asarray(cache["events_0"])
    expected = np.r_[events[0, :3] / (2 * np.pi), events[0, 3]]
    monkeypatch.setattr(
        raw_dgs,
        "inspect_raw_dgs_run",
        lambda *_a, **_k: pytest.fail("bounds reopened a cached raw file"),
    )
    bounds = raw_dgs_coordinate_bounds(group)
    np.testing.assert_allclose(bounds, np.column_stack((expected, expected)))
    set_reduction_settings(group, {"ub_matrix": (2 * np.eye(3)).tolist()})
    changed = raw_dgs_coordinate_bounds(group)
    np.testing.assert_allclose(np.asarray(changed)[:3], np.asarray(bounds)[:3] / 2)
    powder = raw_dgs_coordinate_bounds(group, powder=True)
    np.testing.assert_allclose(powder[0], [np.linalg.norm(events[0, :3])] * 2)


def test_native_derived_grid_avoids_loading_point_arrays(group, monkeypatch):
    from nfit import project_data
    from nfit.project_composites import data_group_composite_config

    root = DataGroup("sample", subgroups=[group])
    config = data_group_composite_config(group)
    config["enabled"] = True
    monkeypatch.setattr(
        project_data,
        "_ensure_dataset_data_loaded",
        lambda *_a, **_k: pytest.fail("native runs are not point arrays"),
    )
    result = _derived_shared_grid_config(root, [f"group-composite:{group.id}"], config)
    assert all(not a["auto_lower"] and not a["auto_upper"] for a in result["axes"])
    assert all(d.data is None for d in group.datasets)
    assert hasattr(group.datasets[0], "_raw_dgs_reduction_cache")


def test_automatic_native_subtraction_reduces_and_bins_sources(group, tmp_path):
    from nfit import project_data
    from nfit.analysis.core import AnalysisEntry
    from nfit.project_composites import data_group_composite_config

    second_path = tmp_path / "SEQ_43.nxs.h5"
    _write_raw_dgs(second_path)
    # Give every automatic axis a nonzero extent in this tiny fixture.
    for path in (group.datasets[0].metadata["source_file"], second_path):
        _rewrite_instrument_xml(path, 'x="1" y="0" z="2"', 'x="1" y="1" z="2"')
        with pytest.importorskip("h5py").File(path, "r+") as archive:
            archive["entry/bank1_events/event_time_offset"][:] = [8500.0, 9500.0]
    second = raw_dgs_dataset_group([second_path])
    set_reduction_settings(
        second,
        {
            "bad_pulse_threshold": 0.0,
            "ki_kf_normalization": False,
            "he3_detector_efficiency_correction": False,
            "ub_matrix": np.eye(3).tolist(),
        },
    )
    root = DataGroup("temperature comparison", subgroups=[group, second])
    for node in root.subgroups:
        data_group_composite_config(node)["enabled"] = True
    analysis = AnalysisEntry(
        "low minus high",
        "histogram_arithmetic",
        [f"group-composite:{node.id}" for node in root.subgroups],
        {"operation": "subtract", "right_scale": 1.0},
    )
    derived = project_data.create_derived_analysis_dataset(root, analysis)
    result = project_data.derived_analysis_dataset_data(derived)
    assert result.signal.size > 0
    assert np.any(~result.mask)
    np.testing.assert_allclose(result.signal[~result.mask], 0.0)
    assert np.any(result.errors[~result.mask] > 0)
    assert all(d.data is None for node in root.subgroups for d in node.datasets)
