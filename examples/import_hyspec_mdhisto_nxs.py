from __future__ import annotations

from pathlib import Path

import numpy as np

from metallix import load_mantid_mdhisto_nxs


FILENAME = "4D_test.nxs"
DATA_PATH = Path("data/hyspec") / FILENAME


def finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.min(finite)), float(np.max(finite)))


def main() -> None:
    path = DATA_PATH
    data = load_mantid_mdhisto_nxs(path)

    print(f"Loaded: {path}")
    print(f"Signal shape: {data.shape}")
    print(f"Coordinate system: {data.coordinate_system}")
    print(f"Visual normalization: {data.visual_normalization}")
    print()
    print("Axes, in array order:")
    for index, axis in enumerate(data.axes):
        low, high = finite_range(axis.values)
        center_low, center_high = finite_range(axis.centers)
        print(
            f"  dim {index}: {axis.name} | kind={axis.kind} | units={axis.units} | "
            f"frame={axis.frame} | values={axis.values.size} | "
            f"edges=[{low:.6g}, {high:.6g}] | centers=[{center_low:.6g}, {center_high:.6g}]"
        )

    print()
    print("Imported arrays:")
    for name, array in (
        ("signal", data.signal),
        ("errors", data.errors),
        ("mask", data.mask),
        ("num_events", data.num_events),
    ):
        print(f"  {name}: shape={array.shape}, dtype={array.dtype}")

    valid_signal = np.isfinite(data.signal)
    unmasked = ~data.mask
    populated = data.num_events > 0
    usable = valid_signal & unmasked & populated
    print()
    print(f"Finite signal bins: {int(np.count_nonzero(valid_signal)):,}")
    print(f"Unmasked bins: {int(np.count_nonzero(unmasked)):,}")
    print(f"Bins with events: {int(np.count_nonzero(populated)):,}")
    print(f"Finite, unmasked bins with events: {int(np.count_nonzero(usable)):,}")


if __name__ == "__main__":
    main()
