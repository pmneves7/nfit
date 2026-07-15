"""Small script-only Bragg and QFI Data Playground example."""

import numpy as np

from nfit import SpectralConvention, integrate_bragg_peaks, spectral_energy_reduce
from nfit.mdhisto import MDHistoAxis, MDHistoData


def bragg_example():
    edges = np.linspace(-1.0, 1.0, 21)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h, k, l = np.meshgrid(centers, centers, centers, indexing="ij")
    signal = 2.0 + 100.0 * np.exp(-(h * h + k * k + l * l) / (2 * 0.12**2))
    axes = tuple(MDHistoAxis(name, edges, "rlu", "momentum") for name in ("H", "K", "L"))
    data = MDHistoData(
        axes, signal, np.sqrt(signal), np.zeros(signal.shape, bool), np.ones(signal.shape),
        metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}},
    )
    return integrate_bragg_peaks(data, [[0, 0, 0]], method="gaussian_fit", ellipsoid_semiaxes=[0.12] * 3)


def qfi_example():
    axes = (
        MDHistoAxis("Q", np.linspace(0.0, 2.0, 9), "1/angstrom", "momentum"),
        MDHistoAxis("DeltaE", np.linspace(0.25, 5.25, 41), "meV", "energy"),
    )
    q = axes[0].centers[:, None]
    energy = axes[1].centers[None, :]
    chipp = np.exp(-q) * energy / (energy * energy + 1.0)
    data = MDHistoData(axes, chipp, np.full(chipp.shape, 0.01), np.zeros(chipp.shape, bool), np.ones(chipp.shape))
    convention = SpectralConvention(
        "chi_double_prime", "spin^2/meV", "per_magnetic_ion", 1.0,
        "spin_squared", None, "removed", "removed", "removed", "removed", True,
    )
    return spectral_energy_reduce(data, kernel="qfi", convention=convention, temperature_K=10.0, energy_max_meV=5.25)


if __name__ == "__main__":
    print(bragg_example().column("I"))
    print(qfi_example().data.column("Quantum Fisher information"))
