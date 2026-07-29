"""Bulk magnetic-response models shared by fitting and analysis workflows."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def curie_weiss_susceptibility(
    temperature_K: ArrayLike,
    curie_constant_cm3_K_per_mol: float,
    theta_CW_K: float,
) -> FloatArray:
    r"""Return the Curie--Weiss molar susceptibility in cm³/mol.

    .. math:: \chi_{\rm mol}(T)=C/(T-\Theta_{\rm CW}).
    """

    temperature = np.asarray(temperature_K, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(
            float(curie_constant_cm3_K_per_mol)
            / (temperature - float(theta_CW_K)),
            dtype=float,
        )
