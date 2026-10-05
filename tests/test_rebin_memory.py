"""Preflight thresholds for ordinary output-array workspace."""

from nfit import performance
from nfit.project_rebinning import estimated_rebin_shape


def test_output_rebin_memory_warns_at_half_available(monkeypatch):
    bins = 10_000_000
    estimate, _, _ = performance.assess_output_rebin_memory(bins)
    monkeypatch.setattr(performance, "available_memory_bytes", lambda: estimate * 2)
    assert not performance.assess_output_rebin_memory(bins)[2]
    monkeypatch.setattr(performance, "available_memory_bytes", lambda: int(estimate * 1.8))
    assert performance.assess_output_rebin_memory(bins)[2]


def test_batch_cache_preflight_warns_at_eighty_percent_of_limit():
    mib = 1024**2
    added, projected, limit, warn = performance.assess_rebin_cache_memory(
        [250 * mib, 350 * mib],
        current_cache_bytes=200 * mib,
        cache_limit_bytes=1000 * mib,
    )
    assert (added, projected, limit, warn) == (
        600 * mib,
        800 * mib,
        1000 * mib,
        True,
    )
    assert not performance.assess_rebin_cache_memory(
        [599 * mib],
        current_cache_bytes=200 * mib,
        cache_limit_bytes=1000 * mib,
    )[3]


def test_rebin_result_estimate_matches_panel_payload_rule():
    assert performance.estimate_rebin_result_bytes(123) == 123 * 57


def test_step_grid_estimate_uses_saved_spacing():
    config = {"axes": [{"mode": "step", "lower": -5.0, "upper": 5.0,
                        "step_size": 0.1, "num_bins": 1},
                       {"mode": "bins", "num_bins": 20}]}
    assert estimated_rebin_shape(config) == (101, 20)


def test_gui_cache_preflight_uses_changed_central_ram_limit(monkeypatch):
    from types import SimpleNamespace

    from nfit import project_gui, project_resource_gui, resource_budget

    limit = [10_000]
    prompts = []
    monkeypatch.setattr(resource_budget, "_rss_provider", lambda: 0)
    monkeypatch.setattr(resource_budget, "_managed_provider", lambda: 0)
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: 100_000)
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: limit[0])
    monkeypatch.setattr(
        project_resource_gui, "handle_resource_limit",
        lambda _parent, error: prompts.append(error) or False,
    )
    explorer = SimpleNamespace(window=None, _compressed_cache_prompt=None)
    confirm = project_gui.NfitProjectExplorer._confirm_rebin_cache_memory
    assert confirm(explorer, [900], operation="Preparing viewer")
    assert prompts == []
    limit[0] = 800
    assert not confirm(explorer, [900], operation="Preparing viewer")
    assert prompts[0].limit_bytes == 800
    assert prompts[0].requested_bytes == 900
