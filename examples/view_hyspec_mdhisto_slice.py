from __future__ import annotations

import os
from pathlib import Path

from metallix import load_mantid_mdhisto_nxs, slice_viewer


FILENAME = "3D_HHL_3meV_1p8K_metallix_m-3m.nxs"


def find_data_file() -> Path:
    candidates = []
    if data_dir := os.environ.get("METALLIX_HYSPEC_DATA_DIR"):
        candidates.append(Path(data_dir) / FILENAME)
    candidates.extend(
        [
            Path("data/hyspec") / FILENAME,
            Path("data/hyspec/raw") / FILENAME,
            Path.home() / "metallix-data/hyspec/raw" / FILENAME,
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {FILENAME}. Tried:\n{tried}")


def main() -> None:
    data = load_mantid_mdhisto_nxs(find_data_file(), copy_metadata=False)
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
