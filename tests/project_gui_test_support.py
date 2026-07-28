# ruff: noqa: F401
import json
import signal
import time
from pathlib import Path

import numpy as np
import pytest

import nfit
import nfit.project_gui as project_gui
from nfit.dataset import PointData4D, PointListData
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import (
    DataGroup,
    DatasetEntry,
    DatasetGroup,
    FitTimelineEntry,
    MaskSpec,
    ModelComponentSpec,
    PlotEntry,
    PlotSourceRef,
)
from nfit.project_gui import (
    NfitProject,
    NfitProjectExplorer,
    available_data_types,
    copy_dataset_to_group,
    copy_mask_to_dataset,
    create_data_group,
    create_mask,
    create_model_component,
    dataset_details_text,
    dataset_entry_from_path,
    dataset_for_slice_viewer,
    dataset_rebin_config,
    default_mask_parameters,
    default_model_config,
    default_model_fit_parameters,
    default_model_parameters,
    forget_missing_recent_projects,
    import_dataset_paths,
    load_project,
    mask_parameter_tooltip,
    model_parameter_tooltip,
    next_data_group_name,
    recent_project_paths,
    remember_recent_project,
    save_dataset_file,
    save_project,
    set_dataset_data_type,
)


def _he_mdhisto_data():
    # Bin edges chosen so H centers are {0, 1, 2} and E centers are {0, 10}.
    h_axis = MDHistoAxis("H", np.array([-0.5, 0.5, 1.5, 2.5]), "rlu", "h")
    e_axis = MDHistoAxis("E", np.array([-5.0, 5.0, 15.0]), "meV", "energy_transfer")
    shape = (3, 2)
    return MDHistoData(
        axes=(h_axis, e_axis),
        signal=np.zeros(shape),
        errors=np.ones(shape),
        mask=np.zeros(shape, dtype=bool),
        num_events=np.ones(shape),
        metadata={},
    )


def _points_for_dynamic_writeback() -> PointData4D:
    return PointData4D(
        H=[0.0],
        K=[0.0],
        L=[0.0],
        E=[1.0],
        intensity=[1.0],
        sigma=[1.0],
        temperature=5.0,
    )


def _standard_shortcut_text(QtGui, standard_key):
    return QtGui.QKeySequence.keyBindings(standard_key)[0].toString(
        QtGui.QKeySequence.SequenceFormat.PortableText
    )


def _tree_items_with_children(tree):
    items = []

    def append_item(item):
        if item.childCount():
            items.append(item)
        for index in range(item.childCount()):
            append_item(item.child(index))

    for index in range(tree.topLevelItemCount()):
        append_item(tree.topLevelItem(index))
    return items


def _tiny_mdhisto_data(value):
    axis_x = MDHistoAxis("H", np.array([0.0, 1.0]), "rlu", "momentum")
    axis_e = MDHistoAxis("E", np.array([0.0, 1.0]), "meV", "energy")
    signal_values = np.array([[value]])
    return MDHistoData(
        axes=(axis_x, axis_e),
        signal=signal_values,
        errors=np.ones_like(signal_values),
        mask=np.zeros_like(signal_values, dtype=bool),
        num_events=np.ones_like(signal_values),
        metadata={},
    )


def _grid_mdhisto_data():
    axis_x = MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum")
    axis_e = MDHistoAxis("E", np.array([0.0, 10.0, 20.0]), "meV", "energy")
    signal_values = np.array([[1.0, 2.0], [3.0, 4.0]])
    return MDHistoData(
        axes=(axis_x, axis_e),
        signal=signal_values,
        errors=np.ones_like(signal_values),
        mask=np.zeros_like(signal_values, dtype=bool),
        num_events=np.ones_like(signal_values),
        metadata={},
    )


def _explorer_with_fit_result(monkeypatch):
    """Explorer whose tree carries one stored fit result; returns both."""
    from nfit.project_gui import ensure_fit_history

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    group = DataGroup("Datagroup1")
    ensure_fit_history(group)
    result = FitTimelineEntry(
        name="fit 1",
        kind="result",
        snapshot={
            "datasets": [
                {
                    "name": "scan",
                    "parameters": {"temperature": 5.0},
                    "enabled": True,
                    "fit_weight": 1.0,
                    "scale_factor": 1.0,
                }
            ],
            "models": [
                {
                    "name": "bg",
                    "type": "constant_background",
                    "enabled": True,
                    "parameters": {"constant": 1.0},
                    "fit_parameters": {},
                    "sharing": {},
                    "limits": {},
                    "constraints": [],
                    "applies_to": None,
                    "metadata": {},
                }
            ],
        },
        created_at="now",
        goodness={
            "status": "converged",
            "chi2": 1.0,
            "reduced_chi2": 1.0,
            "n_points": 10,
            "n_variables": 1,
            "parameters": {"bg.constant": 1.0},
            "stderr": {},
            "dataset_chi2": {"scan": 1.0},
            "dataset_reduced_chi2": {"scan": 1.0},
            "dataset_n_points": {"scan": 10},
            "skipped_datasets": [],
        },
    )
    group.fits[0].children.append(result)
    explorer = NfitProjectExplorer(NfitProject([group]))

    def find_item(item):
        if explorer._fit_entry_for_item(item) is result:
            return item
        for index in range(item.childCount()):
            found = find_item(item.child(index))
            if found is not None:
                return found
        return None

    result_item = None
    root = explorer.tree
    for top in range(root.topLevelItemCount()):
        result_item = find_item(root.topLevelItem(top))
        if result_item is not None:
            break
    assert result_item is not None
    explorer.tree.setCurrentItem(result_item)
    # Selection may rebuild the tree (items are recreated); use the live
    # current item, which the rebuild re-selects.
    current = explorer.tree.currentItem()
    assert explorer._fit_entry_for_item(current) is result
    return explorer, current


def _rpa_overlay_group():
    data = _grid_mdhisto_data()
    data.metadata["temperature"] = 5.0
    dataset = DatasetEntry("scan", data, kind="mdhisto", data_type="single_crystal_inelastic")
    model = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.3, "gamma0": 2.0, "J1": 0.1},
        fit_parameters={"scale": True, "chi0": True, "gamma0": True, "J1": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11]],
            "orbits": [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]}],
        },
    )
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
    group.models[model.name] = model
    return group, dataset, model
