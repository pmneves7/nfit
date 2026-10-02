"""Serializable statistical targets independent of instrument and presentation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

MEASUREMENT_CONTRACT_VERSION = 1
MEASUREMENT_ESTIMATORS = {
    "counting": ("exposure_pool",),
    "continuous": ("inverse_variance_mean", "uniform_mean"),
    "sampled_function": ("coordinate_mean", "coordinate_integral"),
    "linear_reconstruction": ("linear_sum",),
}


@dataclass(frozen=True, kw_only=True)
class MeasurementContract:
    """Declare a target, units, support policy and statistical assumptions.

    Unit strings are declarations, not a unit-conversion engine. ``value_units``
    describes the observed quantity, or the normalized intensity for counts.
    An interval integral additionally carries ``coordinate_units``. Shared
    sources must be supplied explicitly to the reference estimator.
    """

    kind: str
    estimator: str
    quantity: str
    value_units: str
    exposure_units: str | None = None
    coordinate_units: str | None = None
    dependence: str = "independent"
    normalizer: str = "known"
    interpolation: str | None = None
    missing: str | None = None
    partial_support: str = "reject"

    def __post_init__(self):
        if self.missing is None:
            object.__setattr__(self, "missing", "reject" if self.kind == "linear_reconstruction" else "omit")
        if self.kind not in MEASUREMENT_ESTIMATORS:
            raise ValueError(f"Unknown measurement kind: {self.kind!r}")
        if self.estimator not in MEASUREMENT_ESTIMATORS[self.kind]:
            raise ValueError(f"Estimator {self.estimator!r} does not apply to {self.kind!r}")
        for name in ("quantity", "value_units"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty label; use '1' for dimensionless units")
        if self.dependence not in {"independent", "shared_sources"}:
            raise ValueError("dependence must be 'independent' or 'shared_sources'")
        if self.missing not in {"omit", "reject"}:
            raise ValueError("missing must be 'omit' or 'reject'")
        if self.normalizer not in {"known", "uncertain"}:
            raise ValueError("normalizer must be 'known' or 'uncertain'")
        if self.partial_support not in {"reject", "covered_only"}:
            raise ValueError("partial_support must be 'reject' or 'covered_only'")
        for name in ("exposure_units", "coordinate_units"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty unit label")
        if self.kind == "counting":
            if self.exposure_units is None:
                raise ValueError("counting requires exposure_units")
        elif self.exposure_units is not None or self.normalizer != "known":
            raise ValueError("Exposure and normalizer declarations apply only to counting")
        if self.kind == "sampled_function":
            if self.coordinate_units is None or self.interpolation != "linear":
                raise ValueError("sampled_function requires coordinate_units and explicit linear interpolation")
        elif self.coordinate_units is not None or self.interpolation is not None or self.partial_support != "reject":
            raise ValueError("Coordinate support and interpolation apply only to sampled_function")
        if self.estimator == "inverse_variance_mean" and self.dependence != "independent":
            raise ValueError("Inverse-variance mean requires independent observations; correlated means need GLS")

    @property
    def output_units(self) -> str:
        if self.estimator == "coordinate_integral":
            return f"({self.value_units})*({self.coordinate_units})"
        return self.value_units

    def to_dict(self) -> dict[str, Any]:
        """Return the complete, JSON-compatible scientific declaration."""
        return {"version": MEASUREMENT_CONTRACT_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MeasurementContract:
        values = dict(payload)
        version = values.pop("version", None)
        if type(version) is not int or version != MEASUREMENT_CONTRACT_VERSION:
            raise ValueError("Unsupported measurement contract version")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid measurement contract fields: {error}") from error
