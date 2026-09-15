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


def test_step_grid_estimate_uses_saved_spacing():
    config = {"axes": [{"mode": "step", "lower": -5.0, "upper": 5.0,
                        "step_size": 0.1, "num_bins": 1},
                       {"mode": "bins", "num_bins": 20}]}
    assert estimated_rebin_shape(config) == (101, 20)
