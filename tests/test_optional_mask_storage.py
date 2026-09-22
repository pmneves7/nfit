"""Storage contracts for optional MDHisto mask visualization channels."""

from __future__ import annotations

import numpy as np

from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DatasetEntry, MaskSpec
from nfit.project_masks import _mdhisto_with_nfit_masks
from nfit.project_view_data import _mdhisto_without_nfit_masks


def _histogram() -> MDHistoData:
    shape = (2, 3)
    return MDHistoData(
        axes=(
            MDHistoAxis("H", np.arange(3.0), "rlu", "momentum"),
            MDHistoAxis("E", np.arange(4.0), "meV", "energy_transfer"),
        ),
        signal=np.ones(shape),
        errors=np.ones(shape),
        mask=np.asarray([[False, True, False], [False, False, False]]),
        num_events=np.ones(shape),
        metadata={"nfit_mask": np.ones(shape, dtype=bool)},
    )


def test_file_only_view_uses_shared_mask_and_omits_absent_nfit_channel() -> None:
    data = _histogram()

    viewed = _mdhisto_without_nfit_masks(data)

    assert viewed.mask is data.mask
    assert viewed.metadata["file_mask"] is data.mask
    assert "nfit_mask" not in viewed.metadata
    assert viewed.metadata["file_mask_count"] == 1
    assert viewed.metadata["nfit_mask_count"] == 0


def test_mask_service_does_not_allocate_nfit_channel_without_enabled_masks() -> None:
    data = _histogram()
    dataset = DatasetEntry(
        "histogram",
        data,
        masks=[MaskSpec("disabled", "coordinate_range", enabled=False)],
    )

    viewed = _mdhisto_with_nfit_masks(dataset)

    assert viewed.mask is data.mask
    assert viewed.metadata["file_mask"] is data.mask
    assert "nfit_mask" not in viewed.metadata
    assert viewed.metadata["combined_mask_count"] == 1


def test_live_nfit_mask_metadata_shares_the_evaluated_mask() -> None:
    data = _histogram()
    dataset = DatasetEntry(
        "histogram",
        data,
        masks=[MaskSpec("middle", "coordinate_range", parameters={"H": [0.5, 1.5]})],
    )

    viewed = _mdhisto_with_nfit_masks(dataset)

    assert viewed.metadata["file_mask"] is data.mask
    assert viewed.metadata["nfit_mask"].flags.writeable is False
    assert viewed.mask.flags.writeable is False
