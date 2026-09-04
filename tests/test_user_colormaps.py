import numpy as np
import pytest
from matplotlib import colormaps

from nfit.colormaps import load_colormap_file


def test_rgb_file_roundtrip_and_reverse(tmp_path):
    path = tmp_path / "test_rgb_roundtrip.csv"
    path.write_text("# RGB bytes\n0,0,255\n255,255,255\n255,0,0\n")
    name = load_colormap_file(path)
    try:
        np.testing.assert_allclose(colormaps[name](0.0)[:3], [0, 0, 1])
        np.testing.assert_allclose(colormaps[name + "_r"](0.0)[:3], [1, 0, 0])
        with pytest.raises(ValueError, match="already registered"):
            load_colormap_file(path)
    finally:
        colormaps.unregister(name)
        colormaps.unregister(name + "_r")


@pytest.mark.parametrize("text", ["0 0 0", "0 0\n1 1", "0 0 0\nnan 1 1", "0 0 0\n256 0 0", "0 0 0\n1.5 2 3"])
def test_invalid_rgb_table_is_rejected(tmp_path, text):
    path = tmp_path / "invalid.rgb"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_colormap_file(path)


def test_folder_discovery_in_fresh_script(tmp_path):
    import os
    import subprocess
    import sys

    (tmp_path / "custom.txt").write_text("0.1 0.2 0.3\n0.7 0.8 0.9\n")
    subprocess.run(
        [sys.executable, "-c", "from nfit.colormaps import IMAGE_COLORMAPS, VOLUME_COLORMAPS, WATERFALL_COLORMAPS; from matplotlib import colormaps; assert all('user_custom' in c for c in [IMAGE_COLORMAPS, VOLUME_COLORMAPS, WATERFALL_COLORMAPS]); assert colormaps['user_custom_r'](0.0)[0] == 0.7"],
        env={**os.environ, "NFIT_COLORMAP_DIR": str(tmp_path)}, check=True,
    )
