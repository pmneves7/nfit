from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dataset import PointData4D
from .mdhisto import MDHistoData, hyspec_hhl_point_indices


FIT_COMPARISON_METADATA_KEY = "fit_comparisons"


@dataclass
class FitComparisonResultView:
    """Viewer-ready fit result for one reduced dataset."""

    name: str
    fit: MDHistoData
    data: MDHistoData | None = None
    residual: MDHistoData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FitComparisonModelView:
    """Viewer-ready collection of fit results from one model."""

    name: str
    results: list[FitComparisonResultView]
    metadata: dict[str, Any] = field(default_factory=dict)


def attach_fit_comparisons(
    data: MDHistoData,
    comparisons: list[FitComparisonModelView],
) -> MDHistoData:
    """Attach fit-comparison views to an MDHisto dataset in-place."""

    data.metadata[FIT_COMPARISON_METADATA_KEY] = comparisons
    return data


def fit_comparisons_for_data(data: MDHistoData) -> list[FitComparisonModelView]:
    """Return fit-comparison views attached to an MDHisto dataset."""

    comparisons = data.metadata.get(FIT_COMPARISON_METADATA_KEY, [])
    if comparisons is None:
        return []
    if not isinstance(comparisons, list):
        raise TypeError(
            f"metadata[{FIT_COMPARISON_METADATA_KEY!r}] must be a list of "
            "FitComparisonModelView objects"
        )
    for comparison in comparisons:
        if not isinstance(comparison, FitComparisonModelView):
            raise TypeError("fit comparison entries must be FitComparisonModelView objects")
    return comparisons


def hyspec_hhl_fit_comparison_from_points(
    template: MDHistoData,
    prepared: PointData4D,
    fit_values: np.ndarray,
    *,
    model_name: str,
    result_name: str,
) -> FitComparisonModelView:
    """Build a viewer-ready HYSPEC HHL fit comparison on original MDHisto axes."""

    fit_values = np.asarray(fit_values, dtype=float)
    if fit_values.shape != prepared.intensity.shape:
        raise ValueError(
            f"fit_values shape {fit_values.shape} does not match prepared data "
            f"shape {prepared.intensity.shape}"
        )

    indices = hyspec_hhl_point_indices(template, prepared)
    fit_point_mask = np.ones(template.shape, dtype=bool)
    fit_point_mask[indices] = False

    fit_signal = np.full(template.shape, np.nan, dtype=float)
    fit_signal[indices] = fit_values

    residual_signal = np.full(template.shape, np.nan, dtype=float)
    residual_signal[indices] = (prepared.intensity - fit_values) / prepared.sigma

    data = _mdhisto_like(
        template,
        signal=template.signal,
        errors=template.errors,
        mask=fit_point_mask,
        role="fit data",
    )
    fit = _mdhisto_like(
        template,
        signal=fit_signal,
        errors=np.zeros(template.shape, dtype=float),
        mask=fit_point_mask,
        role="fit",
    )
    residual = _mdhisto_like(
        template,
        signal=residual_signal,
        errors=np.ones(template.shape, dtype=float),
        mask=fit_point_mask,
        role="residual",
    )
    return FitComparisonModelView(
        name=model_name,
        results=[
            FitComparisonResultView(
                name=result_name,
                data=data,
                fit=fit,
                residual=residual,
            )
        ],
    )


def _mdhisto_like(
    template: MDHistoData,
    *,
    signal: np.ndarray,
    errors: np.ndarray,
    mask: np.ndarray,
    role: str,
) -> MDHistoData:
    return MDHistoData(
        axes=template.axes,
        signal=signal,
        errors=errors,
        mask=mask,
        num_events=template.num_events,
        coordinate_system=template.coordinate_system,
        visual_normalization=template.visual_normalization,
        metadata={**template.metadata, "viewer_role": role},
    )
