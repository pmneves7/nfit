"""Small immutable datasets shared by plotting and viewer tests."""

import numpy as np

from nfit.mdhisto import MDHistoAxis, MDHistoData


def with_fit_channels(data: MDHistoData) -> MDHistoData:
    metadata = dict(data.metadata)
    metadata["fit"] = np.asarray(data.signal, dtype=float) + 0.5
    metadata["residual"] = np.full(data.shape, -0.5, dtype=float)
    return data.with_updates(metadata=metadata)


def tiny_mdhisto_data() -> MDHistoData:
    signal = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
    return MDHistoData(
        axes=(
            MDHistoAxis("DeltaE", np.array([0.0, 0.5, 1.0]), "meV", "energy"),
            MDHistoAxis("[H,-H,0]", np.array([0.0, 1.0, 2.0, 3.0]), "r.l.u.", "momentum"),
            MDHistoAxis("[0,0,L]", np.linspace(-1.0, 1.0, 5), "r.l.u.", "momentum"),
            MDHistoAxis("[H,H,0]", np.linspace(-2.0, 2.0, 6), "r.l.u.", "momentum"),
        ),
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
    )


def tiny_1d_mdhisto_data() -> MDHistoData:
    signal = np.arange(5, dtype=float).reshape(1, 1, 1, 5)
    return MDHistoData(
        axes=(
            MDHistoAxis("[L,L,-2L]", np.array([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("[H,-H,0]", np.array([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("DeltaE", np.array([2.5, 3.5]), "meV", "energy"),
            MDHistoAxis("[H,H,H]", np.linspace(0.0, 1.0, 6), "r.l.u.", "momentum"),
        ),
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
    )


def tiny_2d_mdhisto_data_with_singletons() -> MDHistoData:
    signal = np.arange(4 * 5, dtype=float).reshape(1, 1, 4, 5)
    return MDHistoData(
        axes=(
            MDHistoAxis("[L,L,-2L]", np.array([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("[H,-H,0]", np.array([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("[0,0,L]", np.linspace(-1.0, 1.0, 5), "r.l.u.", "momentum"),
            MDHistoAxis("[H,H,H]", np.linspace(0.0, 1.0, 6), "r.l.u.", "momentum"),
        ),
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
    )
