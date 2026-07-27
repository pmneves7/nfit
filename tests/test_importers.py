from pathlib import Path

import numpy as np
import pytest

from nfit.dataset import PointListData
from nfit.importers import (
    IMPORTERS,
    import_hb2a_powder,
    import_mpms_dat,
    import_powder_ins_csv,
    import_ppms_heat_capacity_dat,
    import_with,
    importers_for_data_type,
    inspect_powder_ins_csv,
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
    assert data.channels == [{"label": "Moment", "value": "Moment", "error": "M. Std. Err.", "quantity_type": "magnetic_moment", "unit": "emu"}]
    assert data.unit("Temperature") == "K"
    assert data.unit("Magnetic Field") == "Oe"
    assert data.unit("Moment") == "emu"
    # Every raw column is retained, including AC channels.
    assert "AC Moment" in data.columns
    assert "AC Susceptibility" in data.columns
    assert data.size > 5000
    # Header metadata is captured.
    assert data.metadata["mpms_info"]["SAMPLE_MATERIAL"].startswith("EXT_Rodriguez")
    assert data.metadata["sample_mass_mg"] == pytest.approx(9.73)
    assert data.metadata["molar_mass_g_mol"] == pytest.approx(172.8)
    assert data.metadata["sample_volume_mpms"] == pytest.approx(2.16)
    np.testing.assert_allclose(data.column("Temperature")[0], 299.499588, rtol=1e-6)


def test_import_hb2a_powder():
    data = import_hb2a_powder(HB2A_FILE)

    assert data.column_names == ["2theta", "I", "dI"]
    assert data.coordinate_names == ["2theta"]
    assert data.channel_labels == ["I"]
    assert data.channel_errors("I") is not None
    np.testing.assert_allclose(data.column("2theta")[0], 2.15)


def test_import_ppms_heat_capacity_retains_columns_and_normalization(tmp_path):
    path = tmp_path / "heat_capacity.dat"
    path.write_text(
        "[Header]\n"
        "INFO,5.3,MASS:Sample Mass (mg)\n"
        "INFO,172.896,MOLWGHT:Formula Weight (g/mole)\n"
        "[Data]\n"
        "Sample Temp (Kelvin),Samp HC (uJ/K),Samp HC Err (uJ/K),Field (Oersted)\n"
        "2.5,32.4,0.04,0\n3.5,41.5,0.06,0\n",
        encoding="utf-8",
    )
    data = import_ppms_heat_capacity_dat(path)
    assert data.coordinate_names == ["Sample Temp"]
    assert data.channel_labels == ["Sample heat capacity"]
    assert data.unit("Samp HC") == "uJ/K"
    assert data.metadata["sample_mass_mg"] == pytest.approx(5.3)
    assert data.metadata["molar_mass_g_mol"] == pytest.approx(172.896)
    assert "Field" in data.columns
    np.testing.assert_allclose(data.column("Samp HC"), [32.4, 41.5])


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
    assert [spec.name for spec in importers_for_data_type("heat_capacity")] == ["ppms_heat_capacity_dat"]
    assert [spec.name for spec in importers_for_data_type("powder_inelastic")] == [
        "powder_ins_csv"
    ]
    assert IMPORTERS["mpms_dat"].can_read(MPMS_FILE)
    data = import_with("hb2a_powder", HB2A_FILE)
    assert data.coordinate_names == ["2theta"]


def test_import_powder_ins_matrix_sorts_axes_and_masks_nan(tmp_path):
    path = tmp_path / "map.csv"
    path.write_text(
        "y\\x,0.5,1.0\n"
        "2.0,3.0,NaN\n"
        "1.0,1.0,2.0\n",
        encoding="utf-8",
    )
    assert inspect_powder_ins_csv(path)["layout"] == "matrix_q_energy"
    data = import_powder_ins_csv(
        path,
        {
            "temperature_K": 6.0,
            "default_uncertainty": 0.25,
            "source_unit": "mbarn/sr/meV/V",
            "normalization_basis": "per_magnetic_ion",
            "normalization_label": "V",
        },
    )
    assert data.shape == (2, 2)
    np.testing.assert_allclose(data.axes[0].centers, [0.5, 1.0])
    np.testing.assert_allclose(data.axes[1].centers, [1.0, 2.0])
    np.testing.assert_allclose(data.signal[0], [1.0, 3.0])
    assert data.mask[1, 1]
    assert data.metadata["dataset_parameters"]["temperature"] == 6.0
    config = data.metadata["dataset_parameters"]["spectral_channels"]
    assert config["source_unit"] == "mbarn/sr/meV/V"
    assert config["normalization_label"] == "V"


@pytest.mark.parametrize(
    ("cut_type", "fixed_value", "shape", "q", "energy"),
    [
        ("constant_energy", 0.5, (2, 1), [0.4, 0.6], [0.5]),
        ("constant_q", 0.6, (1, 2), [0.6], [0.4, 0.6]),
    ],
)
def test_import_powder_ins_cut_keeps_fixed_axis_and_uncertainty(
    tmp_path, cut_type, fixed_value, shape, q, energy
):
    path = tmp_path / "cut.csv"
    path.write_text("0.6,2.0,0.2\n0.4,1.0,0.1\n", encoding="utf-8")
    data = import_with(
        "powder_ins_csv",
        path,
        {
            "cut_type": cut_type,
            "fixed_value": fixed_value,
            "temperature_K": 10.0,
            "source_representation": "chi_double_prime",
            "source_unit": "mu_B^2/meV/V",
            "fit_representation": "chi_double_prime",
            "normalization_basis": "per_magnetic_ion",
            "normalization_label": "V",
        },
    )
    assert data.shape == shape
    np.testing.assert_allclose(data.axes[0].centers, q)
    np.testing.assert_allclose(data.axes[1].centers, energy)
    np.testing.assert_allclose(data.signal.ravel(), [1.0, 2.0])
    np.testing.assert_allclose(data.errors.ravel(), [0.1, 0.2])


def test_pointlistdata_validates_column_lengths():
    with pytest.raises(ValueError):
        PointListData(columns={"a": [1.0, 2.0], "b": [1.0]})
    with pytest.raises(ValueError):
        PointListData(columns={"a": [1.0]}, coordinate_names=["missing"])
