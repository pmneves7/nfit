from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_REPRESENTATIONS = {"measured_intensity", "cross_section", "s_qw", "chi_double_prime"}
_BASES = {"per_magnetic_ion", "per_formula_unit", "per_unit_cell", "unknown"}
_MOMENT_UNITS = {"spin_squared", "mu_B_squared"}
_STATES = {"included", "removed"}


@dataclass(frozen=True)
class SpectralConvention:
    representation: str
    unit: str
    normalization_basis: str
    magnetic_ions_per_basis: float | None
    moment_unit: str
    g_factor: float | None
    form_factor_state: str
    polarization_state: str
    bose_state: str
    kf_ki_state: str
    absolute_scale: bool
    incident_energy_meV: float | None = None
    final_energy_meV: float | None = None

    def __post_init__(self) -> None:
        if self.representation not in _REPRESENTATIONS:
            raise ValueError(f"unknown spectral representation {self.representation!r}")
        if not self.unit.strip():
            raise ValueError("unit must be explicit")
        if self.normalization_basis not in _BASES:
            raise ValueError(f"unknown normalization basis {self.normalization_basis!r}")
        if self.moment_unit not in _MOMENT_UNITS:
            raise ValueError(f"unknown moment unit {self.moment_unit!r}")
        for name in ("form_factor_state", "polarization_state", "bose_state", "kf_ki_state"):
            if getattr(self, name) not in _STATES:
                raise ValueError(f"{name} must be 'included' or 'removed'")
        if self.magnetic_ions_per_basis is not None and self.magnetic_ions_per_basis <= 0:
            raise ValueError("magnetic_ions_per_basis must be positive")
        if self.g_factor is not None and self.g_factor <= 0:
            raise ValueError("g_factor must be positive")
        for name in ("incident_energy_meV", "final_energy_meV"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.incident_energy_meV is not None and self.final_energy_meV is not None:
            raise ValueError("provide incident_energy_meV or final_energy_meV, not both")
        if (
            self.kf_ki_state == "included"
            and self.incident_energy_meV is None
            and self.final_energy_meV is None
        ):
            raise ValueError(
                "kf_ki_state='included' requires incident_energy_meV or final_energy_meV"
            )

    def require_absolute(self, quantity: str) -> None:
        if not self.absolute_scale:
            raise ValueError(f"{quantity} requires an absolute intensity scale")
        if self.normalization_basis == "unknown":
            raise ValueError(f"{quantity} requires a known normalization basis")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SpectralConvention:
        return cls(**payload)
