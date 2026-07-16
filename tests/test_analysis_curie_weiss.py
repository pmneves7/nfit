import numpy as np
import pytest

from nfit.analysis import AnalysisContext, AnalysisInput, run_analysis_operation
from nfit.analysis.curie_weiss import curie_weiss_susceptibility
from nfit.dataset import PointListData
from nfit.plotting import MDHistoSliceViewer
from nfit.quantities import convert_quantity


def _susceptibility_data(unit="cm^3/mol"):
    temperature = np.linspace(40.0, 320.0, 141)
    susceptibility_cgs = curie_weiss_susceptibility(temperature, 0.42, -18.0)
    uncertainty_cgs = np.full(temperature.shape, 2.0e-5)
    susceptibility = convert_quantity(
        susceptibility_cgs, "bulk_susceptibility", "cm^3/mol", unit
    )
    uncertainty = convert_quantity(
        uncertainty_cgs, "bulk_susceptibility", "cm^3/mol", unit
    )
    return PointListData(
        {
            "Temperature": temperature,
            "chi": susceptibility,
            "dchi": uncertainty,
        },
        units={"Temperature": "K", "chi": unit, "dchi": unit},
        coordinate_names=["Temperature"],
        channels=[
            {
                "label": "Susceptibility",
                "value": "chi",
                "error": "dchi",
                "quantity_type": "bulk_susceptibility",
                "unit": unit,
            }
        ],
        quantity_types={
            "Temperature": "temperature",
            "chi": "bulk_susceptibility",
            "dchi": "bulk_susceptibility",
        },
    )


@pytest.mark.parametrize("unit", ["cm^3/mol", "m^3/mol"])
def test_curie_weiss_fit_handles_cgs_and_si_with_uncertainties(unit):
    data = _susceptibility_data(unit)
    analysis_input = AnalysisInput(
        "a" * 32,
        "MPMS",
        data,
        AnalysisContext("group", None, None, None, None),
        "fingerprint",
    )
    result = run_analysis_operation(
        "curie_weiss_fit",
        [analysis_input],
        {"temperature_min_K": 80.0, "temperature_max_K": 280.0},
    )

    assert result.outputs["curie_constant"].value == pytest.approx(0.42, rel=1e-8)
    assert result.outputs["curie_constant"].uncertainty > 0.0
    assert result.outputs["theta_CW"].value == pytest.approx(-18.0, rel=1e-8)
    assert result.outputs["theta_CW"].uncertainty > 0.0
    expected_moment = 2.828 * np.sqrt(0.42)
    assert result.outputs["effective_moment"].value == pytest.approx(
        expected_moment, rel=5e-4
    )
    assert result.outputs["effective_moment"].uncertainty > 0.0
    assert result.diagnostics["fit_points"] < result.diagnostics["total_points"]
    assert len(result.diagnostics["parameter_covariance"]) == 2


def test_curie_weiss_diagnostic_maps_chi_and_inverse_to_fit_overlays():
    data = _susceptibility_data()
    analysis_input = AnalysisInput(
        "a" * 32,
        "MPMS",
        data,
        AnalysisContext("group", None, None, None, None),
        "fingerprint",
    )
    result = run_analysis_operation(
        "curie_weiss_fit",
        [analysis_input],
        {"temperature_min_K": 80.0, "temperature_max_K": 280.0},
    )
    diagnostic = result.outputs["diagnostic"].data
    model = MDHistoSliceViewer(diagnostic)

    assert model.point_channels == ["Susceptibility", "Inverse susceptibility"]
    assert model.point_overlay_channel("fit") == "CW fit susceptibility"
    view = model.slice_arrays()
    np.testing.assert_allclose(view["fit"], diagnostic.column("CW fit susceptibility"))
    model.channel = "Inverse susceptibility"
    assert model.point_overlay_channel("fit") == "CW fit inverse susceptibility"
    assert diagnostic.metadata["vertical_reference_lines"] == [
        {"value": 80.0, "label": "Tmin"},
        {"value": 280.0, "label": "Tmax"},
    ]


def test_curie_weiss_diagnostic_viewer_shows_fit_and_window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _susceptibility_data()
    analysis_input = AnalysisInput(
        "a" * 32,
        "MPMS",
        data,
        AnalysisContext("group", None, None, None, None),
        "fingerprint",
    )
    result = run_analysis_operation(
        "curie_weiss_fit",
        [analysis_input],
        {"temperature_min_K": 80.0, "temperature_max_K": 280.0},
    )
    viewer = QtMDHistoSliceViewer(result.outputs["diagnostic"].data)

    assert viewer.show_fit is True
    assert [
        viewer.channel_combo.itemText(index)
        for index in range(viewer.channel_combo.count())
    ] == ["Susceptibility", "Inverse susceptibility"]
    assert len([line for line in viewer.ax_image.lines if line.get_linestyle() == "--"]) == 2
    assert len([line for line in viewer.ax_image.lines if line.get_label() == "fit"]) == 1
    viewer._set_channel("Inverse susceptibility")
    assert viewer.model._channel_label() == "Inverse susceptibility (mol/cm^3)"
