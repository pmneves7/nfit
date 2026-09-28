"""Experimental fixed-Q Lindhard spectral compression; not a production backend.

Cloud-in-cell deposition preserves signed transition weights and their first
energy moment. Linear (zero-padded) FFT convolution evaluates the *complex*
retarded kernel, retaining both frequency signs and transitions outside the
requested energy window. The exact thermodynamic E=0 replacement is separate.
Only small/moderate operator bases are intended: the transition table is dense.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from nfit.electronic_backends import evaluate_eigensystem
from nfit.electronic_response import _fermi_function, _static_equal_energy_limit


@dataclass
class Transitions:
    gaps: np.ndarray
    strengths: np.ndarray
    equal_gaps: np.ndarray
    equal_products: np.ndarray
    equal_energies: np.ndarray
    equal_occupations: np.ndarray
    equal_differences: np.ndarray
    temperature_K: float
    chemical_potential_meV: float

    @property
    def nbytes(self):
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray))

    def static_correction(self, broadening):
        limit = _static_equal_energy_limit(
            self.equal_energies, self.equal_occupations,
            chemical_potential_meV=self.chemical_potential_meV,
            temperature_K=self.temperature_K, broadening_meV=broadening,
        )
        # At E=0 replace the broadened equal-energy terms, rather than adding
        # their derivative on top of any near-degenerate finite numerator.
        dynamic = self.equal_differences / (self.equal_gaps - 1j * broadening)
        return np.einsum("t,tAB->AB", limit - dynamic, self.equal_products)


def prepare_transitions(model, mesh, Q, operators, *, temperature_K, chemical_potential_meV):
    """Build one exact finite-mesh transition table, including arbitrary probes."""
    k = np.asarray(mesh.reduced_coordinates)
    q = np.mod(np.asarray(Q), 1.0)
    base = evaluate_eigensystem(model, k, eigenvectors=True, backend="numpy", workers=1)
    shifted = evaluate_eigensystem(model, k + q, eigenvectors=True, backend="numpy", workers=1)
    f = _fermi_function(base.eigenvalues, chemical_potential_meV, temperature_K)
    fq = _fermi_function(shifted.eigenvalues, chemical_potential_meV, temperature_K)
    gaps = shifted.eigenvalues[:, None, :] - base.eigenvalues[:, :, None]
    differences = f[:, :, None] - fq[:, None, :]
    matrix = np.einsum(
        "kan,Aab,kbm->knmA", base.eigenvectors.conj(), operators,
        shifted.eigenvectors, optimize=True,
    )
    products = (
        np.asarray(mesh.weights)[:, None, None, None, None]
        * matrix[..., :, None] * matrix[..., None, :].conj()
    )
    equal = np.abs(gaps) <= 1e-10
    # Preserve degenerate thermal/static contributions even when f_n-f_m=0.
    average_energy = 0.5 * (base.eigenvalues[:, :, None] + shifted.eigenvalues[:, None, :])
    average_occupation = 0.5 * (f[:, :, None] + fq[:, None, :])
    return Transitions(
        gaps=gaps.ravel(),
        strengths=(products * differences[..., None, None]).reshape(-1, len(operators), len(operators)),
        equal_gaps=gaps[equal], equal_products=products[equal],
        equal_energies=average_energy[equal], equal_occupations=average_occupation[equal],
        equal_differences=differences[equal], temperature_K=temperature_K,
        chemical_potential_meV=chemical_potential_meV,
    )


def direct_sum(table, energies, broadening):
    """Independent bounded direct sum used by small regression cases."""
    energies = np.asarray(energies)
    result = np.empty((len(energies), *table.strengths.shape[1:]), complex)
    for i, energy in enumerate(energies):
        kernel = 1.0 / (table.gaps - energy - 1j * broadening)
        result[i] = np.einsum("t,tAB->AB", kernel, table.strengths)
    result[np.abs(energies) <= 1e-14] += table.static_correction(broadening)
    return result


def spectral_sum(table, energies, broadening, *, bins_per_width=16, max_bins=1_000_000):
    """Approximate complex response by linear deposition and FFT convolution.

    Grid spacing is eta/bins_per_width. Histogram entries are transition
    *masses*, not densities, so the convolution has no extra grid-step factor.
    No interaction dressing or production interpolation certificates are used.
    """
    energies = np.asarray(energies, dtype=float)
    if not np.isfinite(broadening) or broadening <= 0 or bins_per_width < 1:
        raise ValueError("positive finite broadening and bins_per_width >= 1 required")
    if energies.ndim != 1 or not len(energies) or not np.all(np.isfinite(energies)):
        raise ValueError("energies must be a nonempty finite vector")
    step = broadening / bins_per_width
    low = min(float(table.gaps.min()), float(energies.min()))
    high = max(float(table.gaps.max()), float(energies.max()))
    origin = np.floor(low / step) * step
    bins = max(2, int(np.ceil((high - origin) / step)) + 2)
    if bins > max_bins:
        raise ValueError(f"spectral grid requires {bins} bins, exceeding {max_bins}")
    x = (table.gaps - origin) / step
    left = np.floor(x).astype(np.intp)
    fraction = x - left
    n_ops = table.strengths.shape[1]
    histogram = np.empty((bins, n_ops, n_ops), complex)
    for a in range(n_ops):
        for b in range(n_ops):
            mass = table.strengths[:, a, b]
            histogram[:, a, b] = (
                np.bincount(left, weights=(mass * (1 - fraction)).real, minlength=bins)
                + np.bincount(left + 1, weights=(mass * fraction).real, minlength=bins)
                + 1j * np.bincount(left, weights=(mass * (1 - fraction)).imag, minlength=bins)
                + 1j * np.bincount(left + 1, weights=(mass * fraction).imag, minlength=bins)
            )
    offsets = step * np.arange(-(bins - 1), bins)
    kernel = -1.0 / (offsets + 1j * broadening)
    convolved = fftconvolve(histogram, kernel[:, None, None], axes=0, mode="full")
    values = convolved[bins - 1:2 * bins - 1]
    grid = origin + step * np.arange(bins)
    result = np.empty((len(energies), n_ops, n_ops), complex)
    for a in range(n_ops):
        for b in range(n_ops):
            result[:, a, b] = (
                np.interp(energies, grid, values[:, a, b].real)
                + 1j * np.interp(energies, grid, values[:, a, b].imag)
            )
    result[np.abs(energies) <= 1e-14] += table.static_correction(broadening)
    return result, {"bins": bins, "step_meV": step, "histogram_bytes": histogram.nbytes,
                    "convolution_output_bytes": convolved.nbytes}
