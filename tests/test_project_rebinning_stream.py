import copy

import numpy as np
import pytest

from nfit import project_rebinning as rebinning
from nfit.background_channels import BACKGROUND_PROVENANCE_KEY, BACKGROUND_PROVENANCE_VERSION
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData


def _data() -> MDHistoData:
    shape = (4, 3, 2, 3)
    grid = np.arange(np.prod(shape), dtype=float).reshape(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[0, 0, 0, 0] = True
    events = np.ones(shape)
    events[1, 1, 1, 1] = 0.0
    coverage = np.ones(shape)
    coverage[2:, :, :, :] = 0.6
    normalization = 1.0 + (grid % 4)
    normalization[3, 2, 1, 2] = np.nan
    normalization[0, 2, 1, 1] = 0.0
    return MDHistoData(
        axes=tuple(
            MDHistoAxis(name, edges, unit, kind, metadata={"variable": variable})
            for name, edges, unit, kind, variable in (
                ("H", np.linspace(-1.0, 1.0, 5), "rlu", "momentum", "H"),
                ("K", np.linspace(-0.6, 0.6, 4), "rlu", "momentum", "K"),
                ("L", np.linspace(-0.4, 0.4, 3), "rlu", "momentum", "L"),
                ("DeltaE", np.linspace(0.0, 3.0, 4), "meV", "energy", "E"),
            )
        ),
        signal=2.0 + grid / 10.0,
        errors=0.5 + grid / 100.0,
        mask=mask,
        num_events=events,
        auxiliary_channels={
            "coverage_fraction": MDHistoChannel(coverage),
            "normalization_denominator": MDHistoChannel(normalization),
        },
    )


def _config(data: MDHistoData) -> dict:
    config = {
        "fractional": True,
        "minimum_coverage": 0.35,
        "minimum_samples": 0.25,
        "mean_weighting": "uniform",
        "max_batch_mb": 0.002,
        "symmetry": {"mode": "space_group", "expression": "P -1"},
        "axes": rebinning._default_rebin_axes(data),
    }
    for axis, count in zip(config["axes"], (5, 4, 3, 3), strict=True):
        axis.update(mode="bins", num_bins=count, auto_lower=True, auto_upper=True)
    return config


def test_streamed_mdhisto_rebin_matches_dense_symmetry_fractional_weighted(monkeypatch):
    data = _data()
    config = _config(data)
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", data.signal.size + 1)
    dense = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", 0)
    streamed = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))

    assert [axis.values.tolist() for axis in streamed.axes] == [
        axis.values.tolist() for axis in dense.axes
    ]
    np.testing.assert_allclose(streamed.signal, dense.signal, equal_nan=True)
    np.testing.assert_allclose(streamed.errors, dense.errors, equal_nan=True)
    np.testing.assert_allclose(streamed.num_events, dense.num_events)
    np.testing.assert_allclose(
        streamed.auxiliary_channels["coverage_fraction"].values,
        dense.auxiliary_channels["coverage_fraction"].values,
    )
    np.testing.assert_array_equal(streamed.mask, dense.mask)
    np.testing.assert_allclose(
        streamed.auxiliary_channels["normalization_denominator"].values,
        dense.auxiliary_channels["normalization_denominator"].values,
    )


def test_rebin_preserves_background_with_primary_weights(monkeypatch):
    data = _data()
    background = MDHistoChannel(
        0.25 + np.asarray(data.signal, dtype=float) / 10.0,
        np.full(data.shape, 0.2),
        label="Background",
        unit="arb.",
    )
    data = data.with_updates(
        auxiliary_channels={**data.auxiliary_channels, "background": background},
        metadata={
            **data.metadata,
            BACKGROUND_PROVENANCE_KEY: BACKGROUND_PROVENANCE_VERSION,
            "background_original_exceptions_v1": {"indices": np.array([0]), "signal": np.array([1.0]), "errors": np.array([1.0]), "mask": np.array([False])},
        },
    )
    config = _config(data)
    config["mean_weighting"] = "inverse_variance"
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", data.signal.size + 1)
    dense = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", 0)
    streamed = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))

    assert dense.metadata[BACKGROUND_PROVENANCE_KEY] == BACKGROUND_PROVENANCE_VERSION
    np.testing.assert_allclose(
        streamed.auxiliary_channels["background"].values,
        dense.auxiliary_channels["background"].values,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        streamed.auxiliary_channels["background"].errors,
        dense.auxiliary_channels["background"].errors,
        equal_nan=True,
    )


@pytest.mark.parametrize("streamed", [False, True])
def test_rebin_preserves_original_uncertainty_after_correlated_subtraction(monkeypatch, streamed):
    from nfit.background_channels import background_channel, record_background_original_exceptions
    from nfit.backgrounds import subtract_aligned_background

    data = _data()
    config = _config(data)
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", 0 if streamed else data.signal.size + 1)
    reference = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))
    cancelled = subtract_aligned_background(data, data)
    cancelled = cancelled.with_updates(
        signal=np.where(cancelled.mask, np.nan, 0.0),
        errors=np.where(cancelled.mask, np.nan, 0.0),
    )
    cancelled = record_background_original_exceptions(cancelled, data, ~cancelled.mask)
    output = rebinning._rebin_mdhisto_data(cancelled, copy.deepcopy(config))
    original = background_channel(output, "unsubtracted")
    valid = ~output.mask
    np.testing.assert_allclose(output.signal[valid], 0.0)
    np.testing.assert_allclose(output.errors[valid], 0.0)
    np.testing.assert_allclose(original.values[valid], reference.signal[valid])
    np.testing.assert_allclose(original.errors[valid], reference.errors[valid])


@pytest.mark.parametrize("with_symmetry", [False, True])
def test_streamed_mdhisto_coverage_matches_dense_nonunit_basis(monkeypatch, with_symmetry):
    data = _data()
    config = _config(data)
    if not with_symmetry:
        config.pop("symmetry")
    for axis, vector in zip(
        config["axes"],
        ([2.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0]),
        strict=True,
    ):
        axis["vector"] = vector
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", data.signal.size + 1)
    dense = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", 0)
    streamed = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))

    np.testing.assert_allclose(streamed.signal, dense.signal, equal_nan=True)
    np.testing.assert_allclose(
        streamed.auxiliary_channels["coverage_fraction"].values,
        dense.auxiliary_channels["coverage_fraction"].values,
    )
    np.testing.assert_array_equal(streamed.mask, dense.mask)


def test_streamed_mdhisto_rebin_propagates_progress_cancellation(monkeypatch):
    data = _data()
    config = _config(data)
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", 0)

    def cancel(event):
        if event.get("stage") == "rebin":
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        rebinning._rebin_mdhisto_data(data, config, progress_callback=cancel)


def test_streamed_mdhisto_coverage_propagates_cancellation(monkeypatch):
    data = _data()
    config = _config(data)
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", 0)

    def cancel(event):
        if event.get("stage") == "rebin_coverage" and "iteration" in event:
            raise RuntimeError("cancelled coverage")

    with pytest.raises(RuntimeError, match="cancelled coverage"):
        rebinning._rebin_mdhisto_data(data, config, progress_callback=cancel)


def test_streamed_mdhisto_rebin_matches_dense_explicit_edges(monkeypatch):
    axes = (
        MDHistoAxis("Q", [0.0, 0.5, 1.5, 3.0], "1/angstrom", "momentum"),
        MDHistoAxis("DeltaE", [-2.0, -0.5, 1.0, 4.0], "meV", "energy"),
    )
    data = MDHistoData(
        axes=axes,
        signal=np.arange(9, dtype=float).reshape(3, 3),
        errors=np.ones((3, 3)),
        mask=np.zeros((3, 3), dtype=bool),
        num_events=np.ones((3, 3)),
    )
    config = {
        "fractional": True,
        "minimum_coverage": 0.5,
        "axes": rebinning._default_rebin_axes(data),
    }
    config["axes"][0].update(mode="edges", bin_edges=[0.0, 1.0, 3.0])
    config["axes"][1].update(mode="edges", bin_edges=[-2.0, 0.0, 4.0])
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", data.signal.size + 1)
    dense = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", 0)
    streamed = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))

    np.testing.assert_allclose(streamed.signal, dense.signal, equal_nan=True)
    np.testing.assert_allclose(streamed.errors, dense.errors, equal_nan=True)
    np.testing.assert_allclose(
        streamed.auxiliary_channels["coverage_fraction"].values,
        dense.auxiliary_channels["coverage_fraction"].values,
    )
    np.testing.assert_array_equal(streamed.mask, dense.mask)


def test_streamed_mdhisto_rebin_matches_dense_scaled_axis_permutation(monkeypatch):
    data = MDHistoData(
        axes=(
            MDHistoAxis("x", [0.0, 1.0, 2.0, 3.0], "", "unknown"),
            MDHistoAxis("y", [-1.0, 0.0, 2.0], "", "unknown"),
        ),
        signal=np.arange(6, dtype=float).reshape(3, 2),
        errors=np.ones((3, 2)),
        mask=np.zeros((3, 2), dtype=bool),
        num_events=np.ones((3, 2)),
    )
    config = {"fractional": True, "minimum_coverage": 0.1, "axes": rebinning._default_rebin_axes(data)}
    config["axes"][0].update(mode="bins", num_bins=3, vector=[0.0, 2.0])
    config["axes"][1].update(mode="bins", num_bins=2, vector=[0.5, 0.0])
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", data.signal.size + 1)
    dense = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", 0)
    streamed = rebinning._rebin_mdhisto_data(data, copy.deepcopy(config))

    np.testing.assert_allclose(streamed.signal, dense.signal, equal_nan=True)
    np.testing.assert_allclose(
        streamed.auxiliary_channels["coverage_fraction"].values,
        dense.auxiliary_channels["coverage_fraction"].values,
    )


def test_streamed_mdhisto_rebin_rejects_no_valid_normalized_bins(monkeypatch):
    data = _data().with_updates(
        auxiliary_channels={
            "normalization_denominator": MDHistoChannel(np.full(_data().shape, np.nan))
        }
    )
    config = _config(data)
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", 0)
    with pytest.raises(ValueError, match="no valid data points"):
        rebinning._rebin_mdhisto_data(data, config)


def test_large_dynamic_axis_mode_keeps_dense_compatibility_path(monkeypatch):
    data = _data()
    config = _config(data)
    config.pop("symmetry")
    config["axes"][0].update(mode="tolerance", tolerance=0.1)
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", 0)

    assert not rebinning._mdhisto_streaming_supported(data, config, config["axes"])


def test_small_mdhisto_keeps_dense_path(monkeypatch):
    data = _data()
    config = _config(data)
    monkeypatch.setattr(rebinning, "MDHISTO_STREAM_MIN_POINTS", data.signal.size + 1)

    assert not rebinning._mdhisto_streaming_supported(data, config, config["axes"])
