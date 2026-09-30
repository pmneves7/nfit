"""Scalar calibration after composite binning, independent of Qt."""

from types import SimpleNamespace

import numpy as np

SCALING_KEY = "composite_scaling"
SCALING_DEFAULTS = {"data_scale": 1.0, "result_scale": 1.0, "fit_weight": 1.0}


def composite_scaling(group):
    """Read a collection's scalar recipe without loading numerical data."""
    result = {**SCALING_DEFAULTS, **group.metadata.get(SCALING_KEY, {})}
    for key in SCALING_DEFAULTS:
        value = float(result[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        result[key] = value
    return result


def configure_composite_scaling(
    group, *, node=None, data_scale=None, result_scale=None, fit_weight=None
):
    """Set collection calibration, final-result calibration, or fitting weight.

    Data scale applies before collection backgrounds. Result scale applies
    afterwards. Fit weight affects fitting only. Individual run scales remain
    part of event binning. All coefficients are finite and nonnegative.
    """
    owner = group if node is None else node
    settings = composite_scaling(owner)
    for key, value in (
        ("data_scale", data_scale),
        ("result_scale", result_scale),
        ("fit_weight", fit_weight),
    ):
        if value is not None:
            settings[key] = float(value)
    candidate = SimpleNamespace(metadata={SCALING_KEY: settings})
    owner.metadata[SCALING_KEY] = composite_scaling(candidate)


def scale_composite_data(data, factor, scale_callback):
    """Use the shared dataset-scaling implementation; zero is explicit."""
    if factor == 1:
        return data
    # Keep dataset calibration compatibility while permitting exact zero.
    if factor == 0:
        from .mdhisto import MDHistoData

        if isinstance(data, MDHistoData):
            from .background_channels import scale_background_channels

            result = scale_background_channels(data, 0)
            return result.with_updates(signal=data.signal * 0, errors=data.errors * 0)
        from .dataset import PointData4D, PointListData

        if isinstance(data, PointData4D):
            return data.with_updates(intensity=data.intensity * 0, sigma=data.sigma * 0)
        if isinstance(data, PointListData):
            columns = dict(data.columns)
            for channel in data.channels:
                for name in (channel.get("value"), channel.get("error")):
                    if name in columns:
                        columns[name] = columns[name] * 0
            return data.with_updates(columns=columns)
        raise TypeError("unsupported composite data container")
    return scale_callback(SimpleNamespace(scale_factor=factor), data)
