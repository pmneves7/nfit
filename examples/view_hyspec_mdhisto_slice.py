from __future__ import annotations

from pathlib import Path

from metallix import load_mantid_mdhisto_nxs, slice_viewer


FILENAME = "4D_test.nxs"
DATA_PATH = Path("data/hyspec") / FILENAME


def main() -> None:
    data = load_mantid_mdhisto_nxs(DATA_PATH, copy_metadata=False)
    viewer = slice_viewer(
        data,
        x_dim="[H,H,0]",
        y_dim="[0,0,L]",
        channel="signal",
        cmap="viridis",
        color_scale="linear",
        auto_limits="min/max",
    )
    print(f"Opened slice viewer for shape {viewer.data.shape}")
    viewer.run()


if __name__ == "__main__":
    main()
