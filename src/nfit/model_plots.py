"""Scriptable calculations and renderers owned by registered response models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .spin_fluctuations import generalized_paramagnon_susceptibility


def generalized_paramagnon_energy_scan(
    energy: ArrayLike,
    *,
    spatial_kernels: ArrayLike = (1.0,),
    chi_peak: float = 1.0,
    gamma0: float = 2.0,
    relaxation_power: float = 1.0,
    inverse_mode_energy_sq: float = 0.0,
) -> dict[str, np.ndarray]:
    """Calculate complex response curves at specified spatial kernels.

    This GUI-independent result is suitable for notebooks, scripts, tests, or
    the registered model-plot renderer.
    """

    energy_array = np.asarray(energy, dtype=float)
    if energy_array.ndim != 1:
        raise ValueError("energy must be one-dimensional")
    kernels = np.atleast_1d(np.asarray(spatial_kernels, dtype=float))
    response = generalized_paramagnon_susceptibility(
        kernels[:, None],
        energy_array[None, :],
        chi_peak=chi_peak,
        gamma0=gamma0,
        relaxation_power=relaxation_power,
        inverse_mode_energy_sq=inverse_mode_energy_sq,
    )
    return {
        "energy": energy_array,
        "spatial_kernels": kernels,
        "susceptibility": response,
    }


def render_generalized_paramagnon_energy_scan(
    result: dict[str, np.ndarray],
    *,
    axes: Sequence[Any] | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """Render real and imaginary susceptibility panels with Matplotlib."""

    import matplotlib.pyplot as plt

    if axes is None:
        figure, created_axes = plt.subplots(2, 1, sharex=True)
        real_axis, imaginary_axis = created_axes
    else:
        if len(axes) != 2:
            raise ValueError("axes must contain real and imaginary axes")
        real_axis, imaginary_axis = axes
        figure = real_axis.figure
    energy = np.asarray(result["energy"], dtype=float)
    kernels = np.asarray(result["spatial_kernels"], dtype=float)
    response = np.asarray(result["susceptibility"], dtype=np.complex128)
    for index, kernel in enumerate(kernels):
        label = f"A = {kernel:g}"
        real_axis.plot(energy, response[index].real, label=label)
        imaginary_axis.plot(energy, response[index].imag, label=label)
    real_axis.set_ylabel(r"$\chi'$ (meV$^{-1}$)")
    imaginary_axis.set_ylabel(r"$\chi''$ (meV$^{-1}$)")
    imaginary_axis.set_xlabel("Energy transfer (meV)")
    real_axis.legend()
    imaginary_axis.legend()
    figure.tight_layout()
    return figure, (real_axis, imaginary_axis)


def generalized_paramagnon_energy_scan_script(
    *,
    energy: Sequence[float],
    spatial_kernels: Sequence[float] = (1.0,),
    chi_peak: float = 1.0,
    gamma0: float = 2.0,
    relaxation_power: float = 1.0,
    inverse_mode_energy_sq: float = 0.0,
) -> str:
    """Return an editable, GUI-free script reproducing an energy-scan plot."""

    arguments = {
        "energy": [float(value) for value in energy],
        "spatial_kernels": [float(value) for value in spatial_kernels],
        "chi_peak": float(chi_peak),
        "gamma0": float(gamma0),
        "relaxation_power": float(relaxation_power),
        "inverse_mode_energy_sq": float(inverse_mode_energy_sq),
    }
    return "\n".join(
        [
            "from nfit.model_plots import (",
            "    generalized_paramagnon_energy_scan,",
            "    render_generalized_paramagnon_energy_scan,",
            ")",
            "",
            f"settings = {arguments!r}",
            "result = generalized_paramagnon_energy_scan(**settings)",
            "figure, axes = render_generalized_paramagnon_energy_scan(result)",
            "figure.show()",
            "",
        ]
    )
