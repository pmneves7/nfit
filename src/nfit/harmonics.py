"""Shared real/complex spherical-harmonic conventions.

Two subsystems depend on exactly the same ordering and phase of the real
spherical harmonics: :mod:`nfit.electronic_builder`, which derives
symmetry-allowed onsite and hopping terms from the site point group, and
:mod:`nfit.electronic_spin`, which projects orbital angular momentum onto a
manifold for spin-orbit coupling. If the two ever disagreed the resulting
Hamiltonian would be silently wrong rather than raise, so the convention lives
here once.

Convention
----------
Real harmonics are built from the Condon-Shortley complex harmonics as

``Y_{l,m}^{cos} = (Y_l^{-m} + (-1)^m Y_l^{m}) / sqrt(2)``,
``Y_{l,m}^{sin} = i (Y_l^{-m} - (-1)^m Y_l^{m}) / sqrt(2)``,

ordered by :func:`real_harmonic_layout`. The ``l = 1`` and ``l = 2`` orders are
chosen to give the familiar ``(p_x, p_y, p_z)`` and
``(d_xy, d_yz, d_zx, d_x2_y2, d_z2)`` sequences; higher ``l`` uses the generic
``m0, cos1, sin1, cos2, sin2, ...`` order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


def real_harmonic_layout(l: int) -> tuple[tuple[str, int], ...]:
    """Return the ordered ``(kind, m)`` labels of the real harmonics of ``l``."""

    if l == 0:
        return (("m0", 0),)
    if l == 1:
        return (("cos", 1), ("sin", 1), ("m0", 0))
    if l == 2:
        return (("sin", 2), ("sin", 1), ("cos", 1), ("cos", 2), ("m0", 0))
    return (("m0", 0),) + tuple(
        entry for m in range(1, l + 1) for entry in (("cos", m), ("sin", m))
    )


def real_harmonic_names(l: int) -> tuple[str, ...]:
    """Return display names for the real harmonics of ``l``, in layout order."""

    if l == 0:
        return ("s",)
    if l == 1:
        return ("p_x", "p_y", "p_z")
    if l == 2:
        return ("d_xy", "d_yz", "d_zx", "d_x2_y2", "d_z2")
    shell = {3: "f"}.get(l, f"l{l}")
    return tuple(
        f"{shell}_{kind}{m}" if kind != "m0" else f"{shell}_m0"
        for kind, m in real_harmonic_layout(l)
    )


def real_harmonic_transform(l: int) -> ComplexArray:
    """Map nfit's ordered real harmonics into complex ``m = -l..l`` states.

    Column ``j`` holds the complex-harmonic components of the ``j``-th real
    harmonic, so an operator ``O`` given in the complex basis becomes
    ``T^dagger O T`` in the real basis.
    """

    size = 2 * l + 1
    result = np.zeros((size, size), dtype=np.complex128)
    for column, (kind, m) in enumerate(real_harmonic_layout(l)):
        if kind == "m0":
            result[l, column] = 1.0
        elif kind == "cos":
            result[l + m, column] = (-1) ** m / np.sqrt(2.0)
            result[l - m, column] = 1.0 / np.sqrt(2.0)
        else:
            result[l + m, column] = -1j * (-1) ** m / np.sqrt(2.0)
            result[l - m, column] = 1j / np.sqrt(2.0)
    return result


def complex_harmonic_angular_momentum(l: int) -> ComplexArray:
    """Return dimensionless ``Lx, Ly, Lz`` on the complex ``m = -l..l`` basis."""

    angular_momentum = int(l)
    size = 2 * angular_momentum + 1
    m_values = np.arange(-angular_momentum, angular_momentum + 1, dtype=float)
    raising = np.zeros((size, size), dtype=np.complex128)
    for column, m_value in enumerate(m_values[:-1]):
        raising[column + 1, column] = np.sqrt(
            angular_momentum * (angular_momentum + 1.0)
            - m_value * (m_value + 1.0)
        )
    lowering = raising.conj().T
    return np.asarray(
        (
            (raising + lowering) / 2.0,
            (raising - lowering) / (2.0j),
            np.diag(m_values),
        ),
        dtype=np.complex128,
    )
