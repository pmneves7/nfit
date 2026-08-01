"""Angular convergence of the powder orientation average.

A powder dataset supplies :math:`|\\mathbf Q|` rather than a direction, so nfit
evaluates the single-crystal response on ``powder_orientations`` deterministic
approximately equal-area sphere directions and averages them. That count is a
numerical approximation exactly like an integration mesh or a lifetime
broadening, and it is converged the same way: evaluate the observable at a
sequence of orientation counts, compare each against the densest one, and
report the largest absolute and relative deviation over the fitted points.

The scan is model-agnostic. It varies only ``powder_orientations`` on a copy of
the supplied component, so it works for every model that powder-averages
(``heisenberg_rpa``, ``generalized_paramagnon``, and the electronic-response
models) without knowing how any of them evaluates.

Convergence is not monotone in the orientation count. The directions are a
quasi-uniform spiral rather than a nested sequence, so a coarse count can land
favourably by accident; read the trend across several counts rather than a
single pair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .cache_utils import readonly_array
from .dataset import PointData4D

FloatArray = NDArray[np.float64]

POWDER_DATA_TYPES = ("powder_inelastic", "powder_elastic")


@dataclass(frozen=True)
class PowderConvergenceResult:
    """Orientation-count convergence of a powder-averaged model observable."""

    orientation_counts: tuple[int, ...]
    values: FloatArray
    """``(n_counts, n_points)`` model prediction at each orientation count."""

    max_absolute_deviation: FloatArray
    max_relative_deviation: FloatArray
    reference_index: int
    converged_count: int | None
    """Smallest count meeting the requested tolerance, or ``None``."""

    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counts = tuple(int(value) for value in self.orientation_counts)
        values = readonly_array(self.values, float)
        absolute = readonly_array(self.max_absolute_deviation, float)
        relative = readonly_array(self.max_relative_deviation, float)
        if len(counts) < 1 or values.ndim != 2 or values.shape[0] != len(counts):
            raise ValueError("values must hold one row per orientation count")
        if absolute.shape != (len(counts),) or relative.shape != (len(counts),):
            raise ValueError("deviation metrics must hold one value per count")
        if not 0 <= int(self.reference_index) < len(counts):
            raise ValueError("reference_index is out of range")
        if len(set(counts)) != len(counts):
            raise ValueError("orientation counts must be distinct")
        object.__setattr__(self, "orientation_counts", counts)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "max_absolute_deviation", absolute)
        object.__setattr__(self, "max_relative_deviation", relative)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def reference_count(self) -> int:
        """Orientation count every other row is compared against."""

        return self.orientation_counts[int(self.reference_index)]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible convergence record."""

        return {
            "orientation_counts": list(self.orientation_counts),
            "values": self.values.tolist(),
            "max_absolute_deviation": self.max_absolute_deviation.tolist(),
            "max_relative_deviation": self.max_relative_deviation.tolist(),
            "reference_index": int(self.reference_index),
            "converged_count": self.converged_count,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PowderConvergenceResult:
        """Reconstruct a result from :meth:`to_dict` output."""

        return cls(
            orientation_counts=tuple(
                int(value) for value in payload["orientation_counts"]
            ),
            values=np.asarray(payload["values"], dtype=float),
            max_absolute_deviation=payload["max_absolute_deviation"],
            max_relative_deviation=payload["max_relative_deviation"],
            reference_index=int(payload["reference_index"]),
            converged_count=(
                None
                if payload.get("converged_count") is None
                else int(payload["converged_count"])
            ),
            provenance=dict(payload.get("provenance", {})),
        )


def is_powder_dataset(data: PointData4D, data_type: str = "") -> bool:
    """Return whether a dataset is powder-averaged by the model evaluators."""

    metadata = data.metadata if isinstance(data.metadata, Mapping) else {}
    declared = str(data_type or metadata.get("data_type", ""))
    return declared in POWDER_DATA_TYPES or bool(
        metadata.get("powder_q_modulus_axis")
    )


def powder_convergence_scan(
    component: Any,
    data: PointData4D,
    *,
    orientation_counts: Sequence[int] = (12, 24, 50, 100, 200),
    data_type: str = "",
    parameters: Mapping[str, float] | None = None,
    relative_tolerance: float = 1.0e-3,
    relative_floor: float = 1.0e-12,
) -> PowderConvergenceResult:
    """Evaluate a powder model at several orientation counts and compare them.

    ``component`` is a model component that powder-averages; it is copied, so
    the caller's ``powder_orientations`` setting is never modified. ``data``
    must be a powder dataset. Each count is compared against the densest one,
    and ``converged_count`` reports the smallest count whose relative deviation
    is within ``relative_tolerance`` and which every denser count also meets.

    Raises ``ValueError`` for a non-powder dataset, fewer than two counts, or a
    count below the six-direction minimum the evaluators enforce.
    """

    from .fit_config import FitDatasetInput, compile_fit_problem
    from .fitting import evaluate_problem_model

    counts = [int(value) for value in orientation_counts]
    if len(counts) < 2:
        raise ValueError("powder convergence needs at least two orientation counts")
    if len(set(counts)) != len(counts):
        raise ValueError("orientation counts must be distinct")
    if any(value < 6 for value in counts):
        raise ValueError("every orientation count must be at least six")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    floor = float(relative_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("relative_floor must be finite and positive")
    if not is_powder_dataset(data, data_type):
        raise ValueError(
            "powder convergence requires a powder dataset; declare "
            f"data_type as one of {POWDER_DATA_TYPES} or attach a "
            "powder_q_modulus_axis"
        )
    order = np.argsort(counts)
    counts = [counts[index] for index in order]

    predictions = []
    for count in counts:
        trial = deepcopy(component)
        if not isinstance(trial.config, dict):
            raise TypeError("component must provide a mutable config dictionary")
        trial.config["powder_orientations"] = int(count)
        compiled = compile_fit_problem(
            [trial],
            [
                FitDatasetInput(
                    "powder",
                    data,
                    data_type=str(data_type or data.metadata.get("data_type", "")),
                )
            ],
        )
        values = (
            {spec.name: spec.value for spec in compiled.problem.parameter_specs}
            if parameters is None
            else dict(parameters)
        )
        predictions.append(
            np.asarray(
                evaluate_problem_model(compiled.problem, "powder", values),
                dtype=float,
            )
        )
    stacked = np.vstack(predictions)
    reference_index = len(counts) - 1
    reference = stacked[reference_index]
    deviation = np.abs(stacked - reference[None, :])
    absolute = np.max(deviation, axis=1)
    relative = np.max(
        deviation / np.maximum(np.abs(reference)[None, :], floor), axis=1
    )

    # The densest count is the reference and trivially agrees with itself, so a
    # count only certifies convergence when every denser count also passes.
    converged: int | None = None
    passing = relative <= tolerance
    for index in range(len(counts)):
        if bool(np.all(passing[index:])):
            converged = counts[index]
            break

    return PowderConvergenceResult(
        orientation_counts=tuple(counts),
        values=stacked,
        max_absolute_deviation=absolute,
        max_relative_deviation=relative,
        reference_index=reference_index,
        converged_count=converged,
        provenance={
            "component": getattr(component, "name", ""),
            "component_type": getattr(component, "type", ""),
            "data_type": str(data_type or data.metadata.get("data_type", "")),
            "points": int(data.size),
            "relative_tolerance": tolerance,
            "relative_floor": floor,
            "reference_orientation_count": counts[reference_index],
            "comparison": "each orientation count versus the densest count",
            "quadrature": "deterministic approximately equal-area sphere spiral",
            "monotonic": False,
        },
    )
