"""Heat-capacity models and low-temperature thermodynamic relations."""

from __future__ import annotations

import numpy as np

MOLAR_GAS_CONSTANT_MJ = 8.31446261815324e3  # mJ/(mol K)
PPMS_HEAT_CAPACITY_UNITS = (
    "uJ/K",
    "uJ/(mol K)",
    "mJ/(g K)",
    "J/(g K)",
    "cal/(g K)",
    "mJ/(mol K)",
    "J/(mol K)",
    "cal/(mol K)",
    "J/(gat K)",
    "cal/(gat K)",
)

# Fixed Gauss-Legendre quadrature keeps repeated nonlinear fits deterministic
# and vectorized.  The large-y branch uses the exact infinite-limit integral.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def debye_heat_capacity(
    temperature_K: np.ndarray,
    debye_temperature_K: float,
    oscillator_count: float = 1.0,
) -> np.ndarray:
    """Return Debye heat capacity in mJ/(mol K).

    ``oscillator_count`` is the number of atoms represented per formula unit;
    consequently the high-temperature limit is ``3*n*R``.
    """

    temperature = np.asarray(temperature_K, dtype=float)
    theta = float(debye_temperature_K)
    count = float(oscillator_count)
    result = np.full(temperature.shape, np.nan, dtype=float)
    valid = np.isfinite(temperature) & (temperature > 0.0) & np.isfinite(theta) & (theta > 0.0)
    if not np.any(valid):
        return result
    y = theta / temperature[valid]
    integral = np.empty(y.shape, dtype=float)
    large = y >= 80.0
    integral[large] = 4.0 * np.pi**4 / 15.0
    if np.any(~large):
        upper = y[~large]
        x = 0.5 * upper[:, None] * (_GL_X[None, :] + 1.0)
        exp_minus = np.exp(-x)
        denominator = -np.expm1(-x)
        integrand = x**4 * exp_minus / denominator**2
        integral[~large] = 0.5 * upper * np.sum(_GL_W[None, :] * integrand, axis=1)
    result[valid] = 9.0 * count * MOLAR_GAS_CONSTANT_MJ * integral / y**3
    return result


def low_temperature_heat_capacity(
    temperature_K: np.ndarray, sommerfeld_mJ_mol_K2: float, beta_mJ_mol_K4: float
) -> np.ndarray:
    """Return ``C = gamma*T + beta*T^3`` in mJ/(mol K)."""

    temperature = np.asarray(temperature_K, dtype=float)
    return float(sommerfeld_mJ_mol_K2) * temperature + float(beta_mJ_mol_K4) * temperature**3


def debye_temperature_from_beta(beta_mJ_mol_K4: float, atoms_per_formula_unit: float) -> float:
    """Return Debye temperature from ``beta`` for the stated atom count."""

    beta = float(beta_mJ_mol_K4)
    atoms = float(atoms_per_formula_unit)
    if beta <= 0.0 or atoms <= 0.0:
        return float("nan")
    return float((12.0 * np.pi**4 * atoms * MOLAR_GAS_CONSTANT_MJ / (5.0 * beta)) ** (1.0 / 3.0))


def ppms_heat_capacity_to_molar_mJ(
    values: np.ndarray,
    source_unit: str,
    *,
    sample_mass_mg: float | None = None,
    molar_mass_g_mol: float | None = None,
    atoms_per_formula_unit: float | None = None,
) -> np.ndarray:
    """Convert any PPMS Heat Capacity export unit to mJ/(mol K)."""

    array = np.asarray(values, dtype=float)
    unit = str(source_unit)
    molar_mass = float(molar_mass_g_mol or 0.0)
    mass_g = float(sample_mass_mg or 0.0) / 1000.0
    atoms = float(atoms_per_formula_unit or 0.0)
    if unit == "uJ/K":
        if mass_g <= 0.0 or molar_mass <= 0.0:
            raise ValueError("uJ/K heat capacity requires positive sample mass and molar mass")
        return array * 1.0e-3 / (mass_g / molar_mass)
    if unit == "uJ/(mol K)":
        return array * 1.0e-3
    if unit in {"mJ/(g K)", "J/(g K)", "cal/(g K)"}:
        if molar_mass <= 0.0:
            raise ValueError(f"{unit} heat capacity requires a positive molar mass")
        factor = {"mJ/(g K)": 1.0, "J/(g K)": 1.0e3, "cal/(g K)": 4184.0}[unit]
        return array * factor * molar_mass
    if unit == "mJ/(mol K)":
        return array.copy()
    if unit == "J/(mol K)":
        return array * 1.0e3
    if unit == "cal/(mol K)":
        return array * 4184.0
    if unit in {"J/(gat K)", "cal/(gat K)"}:
        if atoms <= 0.0:
            raise ValueError(f"{unit} heat capacity requires a positive atoms-per-formula-unit value")
        factor = 1.0e3 if unit == "J/(gat K)" else 4184.0
        return array * factor * atoms
    raise ValueError(f"unsupported PPMS heat-capacity unit {unit!r}")
