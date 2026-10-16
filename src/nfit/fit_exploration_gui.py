"""Focused presentation and asynchronous coordination for optimization searches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .fit_exploration import (
    ExplorationConfig,
    best_exploration_candidate,
    explore_problem_parameters,
)
from .fit_provenance import fit_problem_fingerprint
from .fit_sampling import SamplingCancelled, sampling_result_from_dict, sampling_result_to_dict
from .fit_settings import optimization_config_from_settings

EXPLORATION_SAMPLES_KEY = "exploration_samples"
EXPLORATION_CONFIG_KEY = "exploration_config"


@dataclass(frozen=True)
class ExplorationContext:
    """Only the project operations needed by the search presentation."""

    compile_entry: Callable[..., Any]
    start_task: Callable[..., bool]
    mark_dirty: Callable[[], None]
    refresh_details: Callable[..., None]
    apply_candidate: Callable[..., bool]
    task_active: Callable[[], bool]


def start_exploration(
    context: ExplorationContext, group: Any, entry: Any, config: ExplorationConfig
) -> bool:
    """Run a separate search without changing saved posterior or fit parameters."""

    def task(progress_callback: Any) -> Any:
        compiled = context.compile_entry(group, entry)
        fingerprint = fit_problem_fingerprint(compiled.problem)
        saved = entry.metadata.get("fit_target_fingerprint")
        if saved is not None and saved != fingerprint:
            raise ValueError(
                "The prepared target has changed. Run a new fit before exploring this result."
            )
        try:
            result = explore_problem_parameters(
                compiled.problem,
                config,
                initial_params=dict(entry.goodness.get("parameters", {})),
                optimization_config=optimization_config_from_settings(entry.optimizer_config),
                progress_callback=progress_callback,
            )
        except SamplingCancelled as exc:
            exc.result.metadata["fit_target_fingerprint"] = fingerprint
            raise
        result.metadata["fit_target_fingerprint"] = fingerprint
        return result

    def success(result: Any) -> bool:
        # Even a cancelled partial search remains separate from inference.
        entry.metadata[EXPLORATION_SAMPLES_KEY] = sampling_result_to_dict(result)
        entry.metadata[EXPLORATION_CONFIG_KEY] = asdict(config)
        context.mark_dirty()
        context.refresh_details(entry)
        return True

    return context.start_task(
        title="Exploring parameter space...",
        failure_title="Parameter exploration",
        task=task,
        on_success=success,
        success_message="Parameter exploration finished; posterior results are unchanged.",
        close_on_success=False,
        cancelled_message="Parameter exploration cancelled; partial search samples saved.",
    )


def promote_exploration(context: ExplorationContext, group: Any, entry: Any) -> bool:
    """Recheck a search candidate on the original target before optimization."""

    def task(progress_callback: Any) -> dict[str, Any]:
        progress_callback(
            {"stage": "exploration", "message": "Checking the original fit objective..."}
        )
        result = sampling_result_from_dict(entry.metadata.get(EXPLORATION_SAMPLES_KEY))
        if result is None:
            raise ValueError("Run an exploration before promoting a candidate.")
        compiled = context.compile_entry(group, entry)
        fingerprint = fit_problem_fingerprint(compiled.problem)
        if result.metadata.get("fit_target_fingerprint") != fingerprint:
            raise ValueError(
                "The prepared target has changed. Run a fresh exploration before promotion."
            )
        candidate = best_exploration_candidate(
            compiled.problem,
            result,
            baseline_params=dict(entry.goodness.get("parameters", {})),
            optimization_config=optimization_config_from_settings(entry.optimizer_config),
        )
        if candidate is None:
            raise ValueError(
                "The search has no candidate that improves the original fit objective."
            )
        return candidate

    def success(candidate: dict[str, Any]) -> bool:
        # Queue optimization after the worker has settled and released its guard.
        from PySide6 import QtCore

        def optimize_when_settled() -> None:
            if context.task_active():
                QtCore.QTimer.singleShot(10, optimize_when_settled)
            else:
                context.apply_candidate(group, entry, candidate)

        QtCore.QTimer.singleShot(0, optimize_when_settled)
        return True

    return context.start_task(
        title="Checking exploration candidate...",
        failure_title="Promote exploration candidate",
        task=task,
        on_success=success,
        success_message="Candidate verified on the original fit objective.",
    )


def exploration_panel(
    entry: Any,
    *,
    start: Callable[[ExplorationConfig], Any],
    promote: Callable[[], Any],
    copy_script: Callable[[ExplorationConfig], Any],
) -> Any:
    """Build controls whose wording distinguishes optimization from inference."""
    from PySide6 import QtWidgets

    panel = QtWidgets.QGroupBox("Parameter exploration (optimization only)")
    panel.setObjectName("fit_exploration_group")
    layout = QtWidgets.QGridLayout(panel)
    settings = dict(entry.metadata.get(EXPLORATION_CONFIG_KEY, {}))
    note = QtWidgets.QLabel(
        "A hotter search can cross likelihood barriers. Its samples do not provide posterior uncertainties or correlations."
    )
    note.setWordWrap(True)
    layout.addWidget(note, 0, 0, 1, 2)
    temperature = QtWidgets.QDoubleSpinBox()
    temperature.setObjectName("fit_exploration_temperature")
    temperature.setRange(1.01, 1000000.0)
    temperature.setDecimals(2)
    temperature.setValue(float(settings.get("temperature", 10.0)))
    temperature.setToolTip(
        "Dimensionless search temperature T > 1. Sampling uses exp(-original objective/T); this is an optimization search, not a physical temperature or posterior."
    )
    initialization = QtWidgets.QComboBox()
    initialization.setObjectName("fit_exploration_initialization")
    initialization.addItem("Across finite parameter bounds", "bounds")
    initialization.addItem("Near the saved fit", "local")
    initialization.setCurrentIndex(
        max(0, initialization.findData(settings.get("initialization", "bounds")))
    )
    initialization.setToolTip(
        "A broad search initializes across all finite parameter bounds. Choose near the saved fit when bounds are not finite; local initialization may miss distant modes."
    )
    widgets = []
    for _label, key, default, minimum, maximum, tip in (
        (
            "Walkers",
            "n_walkers",
            0,
            0,
            10000,
            "Search walkers; 0 chooses automatically from the varied parameter count.",
        ),
        ("Steps", "n_steps", 1000, 1, 1000000, "Search steps per walker."),
        (
            "Seed",
            "random_seed",
            -1,
            -1,
            2147483647,
            "Search random seed; -1 chooses a fresh initialization.",
        ),
    ):
        widget = QtWidgets.QSpinBox()
        widget.setObjectName("fit_exploration_" + key)
        widget.setRange(minimum, maximum)
        value = settings.get(key, default)
        widget.setValue(default if value is None else int(value))
        if key == "n_walkers":
            widget.setSpecialValueText("Automatic")
        elif key == "random_seed":
            widget.setSpecialValueText("Random")
        widget.setToolTip(tip)
        widgets.append(widget)
    for row, (label, widget) in enumerate(
        [
            ("Search temperature", temperature),
            ("Initialize", initialization),
            ("Walkers", widgets[0]),
            ("Steps", widgets[1]),
            ("Seed", widgets[2]),
        ],
        1,
    ):
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        layout.addWidget(widget, row, 1)

    def config() -> ExplorationConfig:
        return ExplorationConfig(
            temperature=temperature.value(),
            initialization=initialization.currentData(),
            n_walkers=widgets[0].value() or None,
            n_steps=widgets[1].value(),
            random_seed=None if widgets[2].value() < 0 else widgets[2].value(),
            kwargs={"workers": -1},
        )

    stored = entry.metadata.get(EXPLORATION_SAMPLES_KEY)
    meta = stored.get("metadata", {}) if isinstance(stored, dict) else {}
    shape = stored.get("chain", {}).get("shape", []) if isinstance(stored, dict) else []
    status = QtWidgets.QLabel(
        "No stored exploration."
        if not meta
        else f"Stored search: T={meta.get('temperature', '?')}, {shape[0] if shape else 0} steps. A new search replaces this exploration."
    )
    status.setWordWrap(True)
    layout.addWidget(status, 6, 0, 1, 2)
    buttons = QtWidgets.QHBoxLayout()
    for text, name, tip, callback in (
        (
            "Run exploration",
            "run",
            "Run a separate optimization search. A new run replaces only the stored exploration.",
            lambda: start(config()),
        ),
        (
            "Promote and optimize",
            "promote",
            "Re-evaluate the best search candidate and saved fit with the original objective. If improved, start a new least-squares fit from that candidate.",
            promote,
        ),
        (
            "Copy exploration script",
            "copy_script",
            "Copy editable GUI-free Python to rerun this search and optionally optimize its best candidate.",
            lambda: copy_script(config()),
        ),
    ):
        button = QtWidgets.QPushButton(text)
        button.setObjectName("fit_exploration_" + name)
        button.setToolTip(tip)
        button.clicked.connect(lambda _checked=False, action=callback: action())
        if name == "promote":
            button.setEnabled(bool(meta))
        buttons.addWidget(button)
    layout.addLayout(buttons, 7, 0, 1, 2)
    return panel
