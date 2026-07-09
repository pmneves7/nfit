from pathlib import Path

import numpy as np
import pytest

from metallix.dataset import PointListData
from metallix.importers import (
    IMPORTERS,
    import_hb2a_powder,
    import_mpms_dat,
    importers_for_data_type,
    import_with,
    read_delimited_text,
    split_name_and_unit,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MPMS_FILE = DATA_DIR / "MPMS" / "test_MPMS.dat"
HB2A_FILE = DATA_DIR / "HB2A" / "test_powder_diffraction.dat"


def test_split_name_and_unit():
    assert split_name_and_unit("Temperature (K)") == ("Temperature", "K")
    assert split_name_and_unit("AC Susceptibility (emu/Oe)") == ("AC Susceptibility", "emu/Oe")
    assert split_name_and_unit("Comment") == ("Comment", "")


def test_read_delimited_text_headerless_whitespace(tmp_path):
    path = tmp_path / "cols.txt"
    path.write_text("  1.0   2.0   0.1\n  1.5   3.0   0.2\n", encoding="utf-8")
    parsed = read_delimited_text(
        path, has_header_row=False, column_names=["x", "y", "dy"], delimiter="whitespace"
    )
    assert parsed.column_labels == ["x", "y", "dy"]
    assert parsed.columns["x"] == [1.0, 1.5]
    assert parsed.columns["dy"] == [0.1, 0.2]


def test_import_mpms_dat_retains_all_columns_and_roles():
    data = import_mpms_dat(MPMS_FILE)

    assert isinstance(data, PointListData)
    assert data.coordinate_names == ["Temperature", "Magnetic Field"]
    assert data.channels == [{"label": "Moment", "value": "Moment", "error": "M. Std. Err."}]
    assert data.unit("Temperature") == "K"
    assert data.unit("Magnetic Field") == "Oe"
    assert data.unit("Moment") == "emu"
    # Every raw column is retained, including AC channels.
    assert "AC Moment" in data.columns
    assert "AC Susceptibility" in data.columns
    assert data.size > 5000
    # Header metadata is captured.
    assert data.metadata["mpms_info"]["SAMPLE_MATERIAL"].startswith("EXT_Rodriguez")
    np.testing.assert_allclose(data.column("Temperature")[0], 299.499588, rtol=1e-6)


def test_import_hb2a_powder():
    data = import_hb2a_powder(HB2A_FILE)

    assert data.column_names == ["2theta", "I", "dI"]
    assert data.coordinate_names == ["2theta"]
    assert data.channel_labels == ["I"]
    assert data.channel_errors("I") is not None
    np.testing.assert_allclose(data.column("2theta")[0], 2.15)


def test_pointlistdata_rebin_reduces_points():
    data = import_hb2a_powder(HB2A_FILE)
    rebinned = data.rebin_to_histogram(["2theta"], num_bins=[40])

    assert rebinned.size <= 40
    assert rebinned.channel_labels == ["I"]
    assert rebinned.channel_errors("I") is not None
    assert rebinned.metadata["rebin"]["coordinate_names"] == ["2theta"]
    # Bin centers stay within the original coordinate range.
    original = data.column("2theta")
    assert rebinned.column("2theta").min() >= original.min() - 1.0
    assert rebinned.column("2theta").max() <= original.max() + 1.0


def test_importer_registry_lookup():
    assert [spec.name for spec in importers_for_data_type("magnetization")] == ["mpms_dat"]
    assert [spec.name for spec in importers_for_data_type("powder_elastic")] == ["hb2a_powder"]
    assert IMPORTERS["mpms_dat"].can_read(MPMS_FILE)
    data = import_with("hb2a_powder", HB2A_FILE)
    assert data.coordinate_names == ["2theta"]


def test_pointlistdata_validates_column_lengths():
    with pytest.raises(ValueError):
        PointListData(columns={"a": [1.0, 2.0], "b": [1.0]})
    with pytest.raises(ValueError):
        PointListData(columns={"a": [1.0]}, coordinate_names=["missing"])
