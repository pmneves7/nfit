import copy
import json

import pytest

import nfit.project_data as project_data
from nfit.performance import (
    initialize_rebin_performance,
    load_performance_settings,
    save_performance_settings,
    transient_rebin_memory_limit_bytes,
)
from nfit.performance_benchmark import (
    BenchmarkCancelled,
    benchmark_candidates,
    benchmark_rebin,
    export_benchmark_script,
    rebin_benchmark_target,
    recommend_performance,
)


@pytest.fixture(autouse=True)
def local_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("NFIT_PERFORMANCE_FILE", str(tmp_path / "preferences.json"))


def test_performance_defaults_are_snapshots(monkeypatch):
    from nfit import _parallel

    monkeypatch.setattr(_parallel, "num_threads", lambda: 3)
    first = {}
    initialize_rebin_performance(first)
    assert first == {"max_batch_mb": 192, "workers": 3}
    save_performance_settings(
        max_batch_mb=64,
        workers=2,
        transient_memory_percent=40,
    )
    assert load_performance_settings() == {
        "max_batch_mb": 64,
        "workers": 2,
        "transient_memory_percent": 40,
    }
    assert transient_rebin_memory_limit_bytes(1_000) == 400
    initialize_rebin_performance(first)
    assert first == {"max_batch_mb": 192, "workers": 3}
    second = {}
    initialize_rebin_performance(second)
    assert second == {"max_batch_mb": 64, "workers": 2}
    with pytest.raises(ValueError):
        save_performance_settings(workers=-1)
    with pytest.raises(ValueError):
        save_performance_settings(transient_memory_percent=81)


def test_project_batch_memory_is_capped_by_machine_preference(monkeypatch):
    monkeypatch.setattr(
        "nfit.performance.transient_rebin_memory_limit_bytes",
        lambda: 1024,
    )
    assert project_data._rebin_max_batch_bytes({"max_batch_mb": 64}) == 1024


def test_scoped_workers_are_restored():
    from nfit._parallel import num_threads, thread_budget

    original = num_threads()
    with thread_budget(2):
        assert num_threads() == 2
        with thread_budget(1):
            assert num_threads() == 1
        assert num_threads() == 2
    assert num_threads() == original


def test_recommendation_prefers_resources_within_tolerance():
    rows = [dict(seconds=1, workers=4, max_batch_mb=192),
            dict(seconds=1.04, workers=2, max_batch_mb=32),
            dict(seconds=1.2, workers=1, max_batch_mb=32)]
    assert recommend_performance(rows) == {"max_batch_mb": 32, "workers": 2}


def test_benchmark_candidates_include_the_machine_worker_ceiling(monkeypatch):
    from nfit import _parallel, performance_benchmark

    monkeypatch.setattr(_parallel, "num_threads", lambda: 24)
    monkeypatch.setattr(performance_benchmark, "_available_memory_mib", lambda: 8192)

    candidates = benchmark_candidates()
    workers = {candidate["workers"] for candidate in candidates}
    assert workers == {1, 4, 8, 24}
    assert {candidate["max_batch_mb"] for candidate in candidates} == {
        32,
        192,
        512,
        1024,
    }


def test_benchmark_candidates_scale_to_large_allocations(monkeypatch):
    from nfit import _parallel, performance_benchmark

    monkeypatch.setattr(_parallel, "num_threads", lambda: 256)
    monkeypatch.setattr(
        performance_benchmark, "_available_memory_mib", lambda: 2 * 1024**2
    )
    candidates = benchmark_candidates()
    assert {candidate["workers"] for candidate in candidates} == {
        1,
        8,
        64,
        128,
        256,
    }
    assert {candidate["max_batch_mb"] for candidate in candidates} == {
        32,
        192,
        512,
        4096,
        65_536,
    }


def test_macos_available_memory_uses_free_inactive_and_speculative_pages(
    monkeypatch,
):
    from nfit import performance

    output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages active: 900.
Pages inactive: 200.
Pages speculative: 50.
"""
    monkeypatch.setattr(
        performance.subprocess,
        "check_output",
        lambda *args, **kwargs: output,
    )
    assert performance._darwin_available_memory_bytes() == 350 * 16384


def _project():
    from nfit import DataGroup, DatasetEntry, NfitProject
    from tests.project_gui_test_support import _tiny_mdhisto_data

    entry = DatasetEntry("test", _tiny_mdhisto_data(2.0))
    return NfitProject([DataGroup("test group", datasets=[entry])]), entry


def test_dataset_benchmark_is_isolated_and_reports_memory():
    project, entry = _project()
    before = copy.deepcopy(entry.parameters)
    rows = []
    result = benchmark_rebin(project, dataset_id=entry.id,
                             candidates=[dict(max_batch_mb=4, workers=1)], progress=rows.append)
    assert result["recommendation"] == {"max_batch_mb": 4, "workers": 1}
    assert rows[0]["seconds"] > 0
    assert rows[0]["peak_mib"] > 0
    assert entry.parameters == before


def test_benchmark_cancel_does_not_apply_anything():
    with pytest.raises(BenchmarkCancelled):
        benchmark_rebin(candidates=[dict(max_batch_mb=1, workers=1)], cancel=lambda: True)


def test_cancellation_kills_active_trial(monkeypatch):
    from nfit import performance_benchmark as module

    class Process:
        killed = False
        waited = False

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return None

        def kill(self):
            Process.killed = True

        def wait(self):
            Process.waited = True

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    checks = iter([False, True])
    with pytest.raises(BenchmarkCancelled):
        benchmark_rebin(candidates=[dict(max_batch_mb=1, workers=1)], cancel=lambda: next(checks))
    assert Process.killed and Process.waited


def test_saved_benchmark_script_roundtrip(tmp_path, monkeypatch):
    from nfit import dataset_entry_from_path
    from nfit.project_gui import save_dataset_file

    project, entry = _project()
    source = tmp_path / "source.npz"
    save_dataset_file(entry, source, use_view=False)
    entry = dataset_entry_from_path(source)
    project.data_groups[0].datasets = [entry]
    subject, _, config = rebin_benchmark_target(project, dataset_id=entry.id)
    config.update(workers=2, max_batch_mb=48)
    script = tmp_path / "benchmark.py"
    export_benchmark_script(script, project, dataset_id=entry.id)
    calls = []

    def record(loaded, **kwargs):
        calls.append((loaded, kwargs))
        return {}

    monkeypatch.setattr("nfit.performance_benchmark.benchmark_rebin", record)
    exec(compile(script.read_text(), str(script), "exec"), {"__file__": str(script)})
    loaded, kwargs = calls[0]
    _, _, saved = rebin_benchmark_target(loaded, dataset_id=kwargs["dataset_id"])
    assert saved == config
    assert not hasattr(project, "_project_path")
    assert subject is entry
    assert json.loads(json.dumps(saved))["workers"] == 2


def test_composite_and_dataset_use_their_worker_ceilings(monkeypatch):
    from nfit import _parallel, project_gui

    project, entry = _project()
    project_gui.dataset_rebin_config(entry)["workers"] = 2
    group = project.data_groups[0]
    project_gui.data_group_composite_config(group)["workers"] = 3
    monkeypatch.setattr(project_data, "_rebinned_dataset_data", lambda *a, **k: _parallel.num_threads())
    monkeypatch.setattr(project_data, "_composite_dataset_data", lambda *a, **k: _parallel.num_threads())
    assert project_gui.rebinned_dataset_data(entry) == 2
    assert project_gui.composite_dataset_data(group) == 3
