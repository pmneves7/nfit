"""Search separation and bounded fit-result presentation regressions."""

import numpy as np
import pytest

from nfit import project_gui as pg
from nfit.fit_exploration import ExplorationConfig
from nfit.fit_exploration_gui import (
    ExplorationContext,
    exploration_panel,
    promote_exploration,
    start_exploration,
)
from nfit.fit_provenance import fit_problem_fingerprint
from nfit.fit_sampling import SamplingResult, sampling_result_to_dict
from nfit.pipeline import DataGroup, DatasetEntry, FitTimelineEntry
from tests.project_gui_test_support import _tiny_mdhisto_data


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def _entry():
    chain = np.arange(24, dtype=float).reshape(6, 2, 2)
    return FitTimelineEntry(
        "fit",
        goodness={
            "parameters": {"a": 1.0, "b": 2.0},
            "covariance": {"variables": ["a", "b"], "matrix": [[1, 0], [0, 1]]},
        },
        metadata={
            "posterior_samples": sampling_result_to_dict(
                SamplingResult(chain.reshape(-1, 2), ["a", "b"], chain=chain)
            )
        },
    )


def test_diagnostics_lazy_tabs_release_replaced_canvases_and_mark_labels_dirty(qt):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtCore

    from nfit.fit_diagnostics_gui import _FitDiagnosticsPlotWindow

    explorer = pg.NfitProjectExplorer(pg.NfitProject())
    window = _FitDiagnosticsPlotWindow(_entry(), explorer)
    assert len(window.window.findChildren(FigureCanvasQTAgg)) == 1
    assert not hasattr(window.tabs.widget(2), "figure")  # corner awaits explicit selection
    window.tabs.setCurrentIndex(2)
    assert hasattr(window.tabs.widget(2), "figure")
    for label in ["one", "two", "three"]:
        window.label_table.item(0, 1).setText(label)
        qt.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        assert len(window.window.findChildren(FigureCanvasQTAgg)) == 1
    assert explorer.has_unsaved_changes
    window.window.close()


def test_posterior_actions_and_encoded_metadata_do_not_decode_chain(qt, monkeypatch):
    from PySide6 import QtWidgets

    from nfit import fit_results

    explorer = pg.NfitProjectExplorer(pg.NfitProject())
    entry = _entry()
    monkeypatch.setattr(
        fit_results, "sampling_result_from_dict", lambda _: pytest.fail("eager decode")
    )
    panel = explorer._posterior_sampler_group_box(DataGroup("g"), entry)
    assert panel.findChild(QtWidgets.QPushButton, "fit_posterior_apply_button").isEnabled()
    tree = QtWidgets.QTreeWidget()
    item = pg._add_metadata_tree_item(tree, "chain", entry.metadata["posterior_samples"]["chain"])
    assert item.childCount() == 0
    assert "stored" in item.text(1)
    assert len(item.toolTip(1)) < 200


def test_exploration_panel_has_distinct_controls_and_tooltips(qt):
    from PySide6 import QtWidgets

    received = []
    panel = exploration_panel(
        _entry(), start=received.append, promote=lambda: None, copy_script=received.append
    )
    assert "optimization only" in panel.title()
    assert panel.findChild(QtWidgets.QDoubleSpinBox, "fit_exploration_temperature").value() == 10
    for widget in panel.findChildren(QtWidgets.QAbstractButton):
        assert widget.toolTip()
    panel.findChild(QtWidgets.QPushButton, "fit_exploration_run").click()
    assert received[0].temperature == 10
    assert received[0].initialization == "bounds"


def _compiled_group():
    dataset = DatasetEntry("scan", _tiny_mdhisto_data(1.0))
    group = DataGroup("g", datasets=[dataset])
    model = pg.create_model_component(group)
    model.parameters["constant"] = 0.0
    model.fit_parameters["constant"] = True
    model.limits["constant"] = [-5.0, 5.0]
    entry = FitTimelineEntry("fit", snapshot=pg.snapshot_data_group_state(group))
    compiled = pg._compiled_problem_for_fit_entry(group, entry)
    entry.goodness["parameters"] = {
        spec.name: spec.value for spec in compiled.problem.parameter_specs
    }
    return group, entry, compiled


def test_exploration_storage_stays_separate_and_promotion_rechecks_target(qt, monkeypatch):
    from nfit import fit_exploration_gui as gui

    group, entry, compiled = _compiled_group()
    posterior = entry.metadata["posterior_samples"] = {"sentinel": "official"}
    fingerprint = fit_problem_fingerprint(compiled.problem)
    entry.metadata["fit_target_fingerprint"] = fingerprint
    name = next(spec.name for spec in compiled.problem.parameter_specs if spec.vary)
    result = SamplingResult(
        np.array([[1.0]]),
        [name],
        np.array([-0.05]),
        metadata={"purpose": "exploration", "temperature": 10.0},
        chain=np.array([[[1.0]]]),
        log_probability_chain=np.array([[-0.05]]),
    )
    monkeypatch.setattr(gui, "explore_problem_parameters", lambda *a, **k: result)
    applied = []

    def start_task(**kwargs):
        kwargs["on_success"](kwargs["task"](lambda event: None))
        return True

    context = ExplorationContext(
        lambda *a: compiled,
        start_task,
        lambda: None,
        lambda e: None,
        lambda *args: applied.append(args) or True,
        lambda: False,
    )
    assert start_exploration(context, group, entry, ExplorationConfig())
    assert entry.metadata["posterior_samples"] is posterior
    assert (
        entry.metadata["exploration_samples"]["metadata"]["fit_target_fingerprint"] == fingerprint
    )
    assert group.models["Model1"].parameters["constant"] == 0.0
    assert promote_exploration(context, group, entry)
    qt.processEvents()
    assert applied[0][2]["parameters"][name] == 1.0
    entry.metadata["exploration_samples"]["metadata"]["fit_target_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="target has changed"):
        promote_exploration(context, group, entry)
    assert len(applied) == 1
    with pytest.raises(ValueError, match="separately"):
        pg._store_sampling_result_on_fit_entry(entry, result)


def test_posterior_append_rejects_changed_target_and_legacy_chain(qt, monkeypatch):
    group, entry, compiled = _compiled_group()
    explorer = pg.NfitProjectExplorer(pg.NfitProject([group]))
    name = next(spec.name for spec in compiled.problem.parameter_specs if spec.vary)
    result = SamplingResult(np.zeros((2, 1)), [name], chain=np.zeros((1, 2, 1)))
    entry.metadata["posterior_samples"] = sampling_result_to_dict(result)
    monkeypatch.setattr(
        pg, "sample_problem_parameters", lambda *a, **k: pytest.fail("must reject before sampling")
    )
    kwargs = dict(
        n_walkers=2,
        n_steps=1,
        burn_in=0,
        thin=1,
        random_seed=1,
        workers=1,
        append=True,
        progress_callback=None,
    )
    with pytest.raises(ValueError, match="matching fit target"):
        explorer._posterior_sampler_result_for_fit(group, entry, **kwargs)
    entry.metadata["fit_target_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="target has changed"):
        explorer._posterior_sampler_result_for_fit(group, entry, **kwargs)


def test_candidate_application_carries_dataset_scale_and_starts_plain_refinement(qt, monkeypatch):
    group, entry, _compiled = _compiled_group()
    group.datasets[0].scale_factor_vary = True
    entry.snapshot = pg.snapshot_data_group_state(group)
    group.fits = [FitTimelineEntry("Initial", kind="initial", snapshot=entry.snapshot), entry]
    entry.optimizer_config = {
        "loss": "cauchy",
        "sampler": {"enabled": True},
        "initialization": {"enabled": True},
    }
    compiled = pg._compiled_problem_for_fit_entry(group, entry)
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    scales = [
        instance.name
        for instance in compiled.parameter_instances.values()
        if instance.component == "dataset" and instance.parameter == "scale_factor"
    ]
    assert len(scales) == 1
    params[scales[0]] = 2.5
    explorer = pg.NfitProjectExplorer(pg.NfitProject([group]))
    optimized = []
    monkeypatch.setattr(explorer, "start_fit_for_selection", lambda: optimized.append(True) or True)
    assert explorer._apply_sampling_candidate(
        group, entry, {"parameters": params}, optimize=True, compiled=compiled
    )
    assert group.datasets[0].scale_factor == 2.5
    promoted = explorer._active_fit_entry(group)
    assert promoted.kind == "current"
    assert promoted.optimizer_config == {"loss": "cauchy"}
    assert optimized == [True]


def test_posterior_promotion_compares_original_robust_objective(monkeypatch):
    from types import SimpleNamespace

    from nfit.dataset import PointData4D
    from nfit.fitting import FitDataset, FitProblem, ParameterSpec

    points = PointData4D(
        H=np.zeros(3),
        K=np.zeros(3),
        L=np.zeros(3),
        E=np.zeros(3),
        intensity=np.array([0.0, 0.0, 100.0]),
        sigma=np.ones(3),
    )
    problem = FitProblem(
        [FitDataset("data", points)],
        model=lambda data, params: np.full(data.size, params["p"]),
        parameter_specs=[ParameterSpec("p", 0.005, min=-10.0, max=100.0)],
    )
    result = SamplingResult(np.array([[33.33]]), ["p"], np.array([-1.0]))
    entry = FitTimelineEntry(
        "fit",
        goodness={"parameters": {"p": 0.005}, "chi2": 9999.0},
        optimizer_config={"loss": "cauchy"},
        metadata={"posterior_samples": sampling_result_to_dict(result)},
    )
    assert pg._best_posterior_promotion_candidate(entry, SimpleNamespace(problem=problem)) is None


def test_copy_exploration_script_uses_selected_optimizer_and_restores_live_state(qt, monkeypatch):
    from nfit import workflow

    group, entry, _compiled = _compiled_group()
    entry.optimizer_config = {"loss": "cauchy", "f_scale": 2.0}
    explorer = pg.NfitProjectExplorer(pg.NfitProject([group]))
    group.models["Model1"].parameters["constant"] = 3.0
    seen = []

    def script(project, name, config, **kwargs):
        seen.append((group.models["Model1"].parameters["constant"], kwargs))
        return "# editable exploration\n"

    monkeypatch.setattr(workflow, "exploration_workflow_script", script)
    assert explorer._copy_exploration_script(group, entry, ExplorationConfig(temperature=5.0))
    assert seen == [(0.0, {"optimizer_config": entry.optimizer_config})]
    assert group.models["Model1"].parameters["constant"] == 3.0
    assert "editable exploration" in qt.clipboard().text()


def test_historical_compilation_owns_model_settings_after_live_state_restore():
    from nfit.fitting import evaluate_problem_model

    data = _tiny_mdhisto_data(1.0).with_updates(
        metadata={"rlu_to_inv_angstrom_matrix": np.eye(3).tolist()}
    )
    group = DataGroup("g", datasets=[DatasetEntry("scan", data)])
    group.lattice_parameters = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
    group.datasets[0].parameters["temperature"] = 5.0
    model = pg.create_model_component(group, type="local_relaxational")

    class TransientCache:
        def __deepcopy__(self, memo):
            pytest.fail("Historical compilation must not clone model numerical caches")

    model._nfit_electronic_model_cache = TransientCache()
    model.config.update(form_factor_mode="custom", form_factor_coefficients=[0, 0, 0, 0, 0, 0, 1])
    entry = FitTimelineEntry("old", snapshot=pg.snapshot_data_group_state(group))
    old = pg._compiled_problem_for_fit_entry(group, entry)
    params = {spec.name: spec.value for spec in old.problem.parameter_specs}
    expected = evaluate_problem_model(old.problem, "scan", params)
    model.config["form_factor_coefficients"] = [0, 0, 0, 0, 0, 0, 2]
    historical = pg._compiled_problem_for_fit_entry(group, entry)
    assert model.config["form_factor_coefficients"][-1] == 2
    np.testing.assert_allclose(evaluate_problem_model(historical.problem, "scan", params), expected)
    np.testing.assert_allclose(evaluate_problem_model(old.problem, "scan", params), expected)
    assert fit_problem_fingerprint(historical.problem) == fit_problem_fingerprint(old.problem)
