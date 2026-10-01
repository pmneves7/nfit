"""Trajectory energy conventions are independent of event reconstruction."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nfit import (
    DataGroup,
    NfitProject,
    NfitProjectExplorer,
    bin_mdevent_group,
    bin_mdevent_powder_group,
    bin_raw_dgs_group,
    bin_raw_dgs_powder_group,
    load_project,
    mdevent_dataset_group,
    raw_dgs_dataset_group,
    save_project,
    set_dgs_trajectory_energy_policy,
)
from nfit.dgs_normalization import trajectory_incident_energies
from nfit.histogram_statistics import EVENT_STATISTICS_VERSION
from nfit.pipeline import MaskSpec
from nfit.project_composites import _composite_cache_signature, _composite_scope
from nfit.raw_dgs_cache import reduction_signature
from tests.test_mdevent import _write_mdevent
from tests.test_raw_dgs import _write_raw_dgs


def test_incident_energy_conventions_preserve_precision_and_override_priority():
    first = 60.7701998715345
    assert trajectory_incident_energies({}, [first, 60.8]) == (first, first)
    assert trajectory_incident_energies(
        {"trajectory_energy_policy": "per_run"}, [first, 60.8]
    ) == (first, 60.8)
    for policy in ("first_run", "per_run"):
        assert trajectory_incident_energies(
            {"trajectory_energy_policy": policy, "incident_energy_override": 60.0},
            [first, 60.8],
        ) == (60.0, 60.0)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_energy_override_is_rejected(value):
    with pytest.raises(ValueError, match="positive and finite"):
        trajectory_incident_energies({"incident_energy_override": value}, [20.0])


def test_pulse_energy_policy_is_not_misrepresented():
    with pytest.raises(ValueError, match="trajectory_energy_policy"):
        trajectory_incident_energies({"trajectory_energy_policy": "per_pulse"}, [20.0])


@pytest.mark.parametrize("accelerated", [False, True])
def test_first_run_mdevent_normalization_matches_analytic_common_energy(
    monkeypatch, tmp_path, accelerated
):
    import nfit.mdevent as mdevent

    if accelerated and mdevent._MDEVENT_NUMBA is None:
        pytest.skip("Numba MDEvent normalization is unavailable")
    if not accelerated:
        monkeypatch.setattr(mdevent, "_MDEVENT_NUMBA", None)
    monkeypatch.setattr(mdevent._parallel, "num_threads", lambda: 1)
    source = tmp_path / "different_energies.nxs"
    _write_mdevent(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(source, "r+") as handle:
        for index, ei in enumerate((10.0, 40.0)):
            logs = handle[f"MDEventWorkspace/experiment{index}/logs"]
            logs["Ei/value"][...] = [ei]
            logs["processed_histogram_bins/value"][...] = [-3.0, 3.0]
    group = mdevent_dataset_group(source)
    edges = ([-0.01, 0.01], [-0.01, 0.01], [0.09, 0.11], [0.0, 3.0])
    options = dict(
        lower=[0, 0, 0.1, 1.5], upper=[0, 0, 0.1, 1.5],
        num_bins=[1, 1, 1, 1], bin_edges=edges,
    )
    coefficient = mdevent.ENERGY_TO_K2

    def span(ei):
        low, high = edges[2]
        return 2 * np.sqrt(coefficient * ei) * (high - low) - coefficient * (high**2 - low**2)

    default = bin_mdevent_group(group, **options)
    np.testing.assert_allclose(default.metadata["normalization_denominator"], 3 * span(10.0))
    set_dgs_trajectory_energy_policy(group, "per_run")
    individual = bin_mdevent_group(group, **options)
    np.testing.assert_allclose(individual.metadata["normalization_denominator"], span(10.0) + 2 * span(40.0))
    np.testing.assert_array_equal(default.num_events, individual.num_events)
    # Trajectory Ei is independent of already stored MDE coordinates/weights.
    np.testing.assert_allclose(
        default.signal * default.metadata["normalization_denominator"],
        individual.signal * individual.metadata["normalization_denominator"],
        equal_nan=True,
    )


@pytest.mark.parametrize("powder", [False, True])
def test_raw_normalization_policy_reuses_events_and_retains_run_energy_bounds(
    monkeypatch, tmp_path, powder
):
    import nfit.raw_dgs as raw_dgs

    paths = [tmp_path / "first.nxs.h5", tmp_path / "second.nxs.h5"]
    for path in paths:
        _write_raw_dgs(path)
    h5py = pytest.importorskip("h5py")
    with h5py.File(paths[1], "r+") as handle:
        handle["entry/DASlogs/BL17:Det:TH:BL:Ei/average_value"][...] = [40.0]
    group = raw_dgs_dataset_group(paths)
    monkeypatch.setattr(raw_dgs, "_MDEVENT_NUMBA", None)
    observed = []
    accumulator_name = "_accumulate_powder_detector_trajectory" if powder else "_accumulate_detector_trajectory"
    original = getattr(raw_dgs, accumulator_name)
    energy_index = 3 if powder else 4

    def accumulate(*args):
        observed.append((args[energy_index], args[energy_index + 1]))
        return original(*args)

    monkeypatch.setattr(raw_dgs, accumulator_name, accumulate)
    options = (
        dict(lower=[0, -100], upper=[20, 100], num_bins=[1, 1]) if powder else
        dict(lower=[-20, -20, -20, -100], upper=[20, 20, 20, 100], num_bins=[1] * 4)
    )
    bin_group = bin_raw_dgs_powder_group if powder else bin_raw_dgs_group
    first = bin_group(group, **options)
    assert [item[0] for item in observed] == [20.0, 20.0]
    assert observed[0][1] == (-19.0, 19.0)
    assert observed[1][1] == (-38.0, 38.0)
    signatures = [reduction_signature(run, group.metadata["raw_dgs"]) for run in group.datasets]
    set_dgs_trajectory_energy_policy(group, "per_run")
    observed.clear()
    individual = bin_group(group, **options)
    assert [item[0] for item in observed] == [20.0, 40.0]
    assert signatures == [reduction_signature(run, group.metadata["raw_dgs"]) for run in group.datasets]
    np.testing.assert_array_equal(first.num_events, individual.num_events)
    np.testing.assert_allclose(
        first.signal * first.metadata["normalization_denominator"],
        individual.signal * individual.metadata["normalization_denominator"],
        equal_nan=True,
    )


def test_mdevent_powder_trajectories_use_the_saved_energy_policy(monkeypatch, tmp_path):
    import nfit.mdevent as mdevent

    source = tmp_path / "different_energies.nxs"
    _write_mdevent(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(source, "r+") as handle:
        for index, ei in enumerate((10.0, 40.0)):
            handle[f"MDEventWorkspace/experiment{index}/logs/Ei/value"][...] = [ei]
    group = mdevent_dataset_group(source)
    monkeypatch.setattr(mdevent, "_MDEVENT_NUMBA", None)
    observed = []
    monkeypatch.setattr(mdevent, "_accumulate_powder_detector_trajectory", lambda *args: observed.append(args[3]))
    options = (group, group.datasets, ([0, 10], [-3, 3]), (1, 1))
    mdevent._powder_trajectory_normalization(*options)
    assert observed == [10.0, 10.0]
    observed.clear()
    set_dgs_trajectory_energy_policy(group, "per_run")
    mdevent._powder_trajectory_normalization(*options)
    assert observed == [10.0, 40.0]


@pytest.mark.parametrize("powder", [False, True])
@pytest.mark.parametrize("policy", ["first_run", "per_run"])
def test_mixed_energy_user_mask_partitions_preserve_full_run_convention(
    monkeypatch, tmp_path, powder, policy
):
    import nfit.mdevent as mdevent

    monkeypatch.setattr(mdevent, "_MDEVENT_NUMBA", None)
    source = tmp_path / "different_energies.nxs"
    _write_mdevent(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(source, "r+") as handle:
        for index, ei in enumerate((10.0, 40.0)):
            handle[f"MDEventWorkspace/experiment{index}/logs/Ei/value"][...] = [ei]
    group = mdevent_dataset_group(source, trajectory_energy_policy=policy)
    bin_group = bin_mdevent_powder_group if powder else bin_mdevent_group
    options = (
        dict(lower=[0, -3], upper=[10, 3], num_bins=[2, 2]) if powder else
        dict(lower=[-5, -5, -5, -3], upper=[5, 5, 5, 3], num_bins=[1, 1, 1, 2])
    )
    baseline = bin_group(group, **options)
    # Distinct masks whose ranges miss the grid force distinct partitions
    # without changing the physical observations or their accepted exposure.
    group.datasets[0].masks.append(MaskSpec("outside first", "energy_q_range", {"energy": [100, 110]}))
    group.datasets[1].masks.append(MaskSpec("outside second", "energy_q_range", {"energy": [200, 210]}))
    partitioned = bin_group(group, **options)
    for field in ("signal", "errors", "num_events", "mask"):
        np.testing.assert_allclose(getattr(partitioned, field), getattr(baseline, field), equal_nan=True)
    np.testing.assert_allclose(
        partitioned.metadata["normalization_denominator"],
        baseline.metadata["normalization_denominator"],
    )
    assert group.metadata["mdevent"]["incident_energy_override"] is None


@pytest.mark.parametrize("kind", ["raw_dgs", "mdevent"])
def test_policy_gui_persistence_and_public_script_replay(monkeypatch, tmp_path, kind):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    source = tmp_path / "source.nxs"
    if kind == "raw_dgs":
        _write_raw_dgs(source)
        subgroup = raw_dgs_dataset_group([source])
    else:
        _write_mdevent(source)
        subgroup = mdevent_dataset_group(source)
    root = DataGroup("Data", subgroups=[subgroup])
    project = NfitProject([root])
    explorer = NfitProjectExplorer(project)
    explorer._refresh_tree(select_dataset_group=subgroup)
    selector = explorer.details_widget.findChild(QtWidgets.QComboBox, f"{kind}_trajectory_energy_policy")
    assert selector is not None and selector.toolTip()
    assert selector.currentData() == "first_run"
    before = _composite_cache_signature(_composite_scope(root, subgroup))
    selector.setCurrentIndex(selector.findData("per_run"))
    assert subgroup.metadata[kind]["trajectory_energy_policy"] == "per_run"
    assert before != _composite_cache_signature(_composite_scope(root, subgroup))
    path = tmp_path / "project.nfit"
    save_project(project, path)
    loaded = load_project(path)
    restored = loaded.data_groups[0].subgroups[0]
    assert restored.metadata[kind]["trajectory_energy_policy"] == "per_run"
    exec("from nfit import set_dgs_trajectory_energy_policy\nset_dgs_trajectory_energy_policy(group, 'first_run')", {"group": restored})
    assert restored.metadata[kind]["trajectory_energy_policy"] == "first_run"
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_effective_policy_and_event_statistics_version_enter_cache_signature(tmp_path):
    source = tmp_path / "source.nxs"
    _write_mdevent(source)
    subgroup = mdevent_dataset_group(source)
    root = DataGroup("Data", subgroups=[subgroup])
    scope = _composite_scope(root, subgroup)
    explicit = _composite_cache_signature(scope)
    subgroup.metadata["mdevent"].pop("trajectory_energy_policy")
    assert _composite_cache_signature(scope) == explicit
    settings = json.loads(explicit)[1]
    assert settings["trajectory_energy_policy"] == "first_run"
    assert settings["histogram_statistics_version"] == EVENT_STATISTICS_VERSION
