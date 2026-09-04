import numpy as np
from matplotlib import colormaps

from nfit.colormaps import (
    CARTOCOLORS_COLORMAPS,
    CMOCEAN_SEQUENTIAL_COLORMAPS,
    IMAGE_COLORMAPS,
    MYCARTA_COLORMAPS,
    VOLUME_COLORMAPS,
    WATERFALL_COLORMAPS,
    WATERFALL_DISCRETE_COLORMAPS,
)


def test_scientific_palettes_preserve_source_colors_and_reversal():
    from pathlib import Path

    from cmocean import cm
    from palettable.cartocolors import sequential
    from palettable.mycarta import get_map

    samples = np.linspace(0, 1, 256)
    names = CMOCEAN_SEQUENTIAL_COLORMAPS + MYCARTA_COLORMAPS + CARTOCOLORS_COLORMAPS
    assert len(names) == 20
    for name in names:
        for catalog in (IMAGE_COLORMAPS, VOLUME_COLORMAPS, WATERFALL_COLORMAPS):
            assert catalog.count(name) == 1
        assert name not in WATERFALL_DISCRETE_COLORMAPS
        np.testing.assert_allclose(
            colormaps[name + "_r"](samples), colormaps[name](samples)[::-1], atol=1e-14,
        )
    for name in CMOCEAN_SEQUENTIAL_COLORMAPS:
        rgb = np.loadtxt(Path(cm.datadir) / (name[4:] + "-rgb.txt"))
        np.testing.assert_allclose(colormaps[name](samples)[:, :3], rgb, atol=1e-14)
    for name in MYCARTA_COLORMAPS:
        source = get_map(name.split(".")[1] + "_256")
        np.testing.assert_allclose(colormaps[name](samples)[:, :3], source.mpl_colors, atol=1e-14)
    for name in CARTOCOLORS_COLORMAPS:
        source = getattr(sequential, name.split(".")[1] + "_7")
        np.testing.assert_allclose(colormaps[name](samples), source.mpl_colormap(samples), atol=1e-14)
