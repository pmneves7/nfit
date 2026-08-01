"""Composable complex-susceptibility helpers.

The fitting registry uses :class:`ScalarSusceptibilityResponse` as the small
interchange object between physical response components. It deliberately
contains the complex response before the fluctuation--dissipation factor and
cross-section constant, so response dressings can be composed without
coupling measured intensities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ScalarSusceptibilityResponse:
    """One neutron-visible scalar spin susceptibility per fitted point.

    ``chi`` is one Cartesian component in ``spin^2/meV`` before the isotropic
    neutron polarization factor. ``form_factor`` is the signed magnetic form
    factor amplitude, not its square. Tensor-valued responses are not coerced
    into this scalar interchange type.
    """

    chi: ComplexArray
    form_factor: FloatArray
    provenance: str = ""

    def __post_init__(self) -> None:
        chi = np.asarray(self.chi, dtype=np.complex128)
        form_factor = np.asarray(self.form_factor, dtype=float)
        if chi.ndim != 1:
            raise ValueError("scalar susceptibility chi must be one-dimensional")
        if form_factor.ndim == 0:
            form_factor = np.full(chi.shape, float(form_factor), dtype=float)
        if form_factor.shape != chi.shape:
            raise ValueError("form_factor must be scalar or match chi")
        if np.any(~np.isfinite(chi)) or np.any(~np.isfinite(form_factor)):
            raise ValueError("susceptibility response must be finite")
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "form_factor", form_factor)


def coupled_scalar_susceptibility(
    chi_a: ArrayLike,
    chi_b: ArrayLike,
    *,
    coupling: float,
    amplitude_a: ArrayLike = 1.0,
    amplitude_b: ArrayLike = 1.0,
    singular_tolerance: float = 1.0e-12,
) -> ComplexArray:
    r"""Couple two scalar susceptibilities through a bilinear interaction.

    For bare responses :math:`\chi_A`, :math:`\chi_B` and coupling energy
    :math:`g`, the two-sector Dyson equation is

    .. math::

       \boldsymbol\chi^{-1} =
       \begin{pmatrix}\chi_A^{-1}&-g\\-g&\chi_B^{-1}\end{pmatrix}.

    The returned response of the visible operator
    :math:`O=a_A O_A+a_B O_B` is

    .. math::

       \chi_O = \frac{a_A^2\chi_A+a_B^2\chi_B
       +2a_Aa_Bg\chi_A\chi_B}{1-g^2\chi_A\chi_B}.

    Magnetic form-factor amplitudes can be supplied as ``amplitude_a`` and
    ``amplitude_b``. At zero coupling the result is exactly the incoherent sum
    of the two source responses, including their individual form factors.
    """

    a = np.asarray(chi_a, dtype=np.complex128)
    b = np.asarray(chi_b, dtype=np.complex128)
    weight_a = np.asarray(amplitude_a, dtype=float)
    weight_b = np.asarray(amplitude_b, dtype=float)
    a, b, weight_a, weight_b = np.broadcast_arrays(a, b, weight_a, weight_b)
    g = float(coupling)
    tolerance = float(singular_tolerance)
    if not np.isfinite(g):
        raise ValueError("coupling must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("singular_tolerance must be finite and positive")
    if (
        np.any(~np.isfinite(a))
        or np.any(~np.isfinite(b))
        or np.any(~np.isfinite(weight_a))
        or np.any(~np.isfinite(weight_b))
    ):
        raise ValueError("coupled susceptibility inputs must be finite")
    denominator = 1.0 - g * g * a * b
    scale = np.maximum(1.0, np.abs(g * g * a * b))
    if np.any(np.abs(denominator) <= tolerance * scale):
        raise np.linalg.LinAlgError("coupled susceptibility is at a sampled pole")
    numerator = (
        weight_a * weight_a * a
        + weight_b * weight_b * b
        + 2.0 * weight_a * weight_b * g * a * b
    )
    return np.asarray(numerator / denominator, dtype=np.complex128)
