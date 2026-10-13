"""Registered scientific routes agree with their direct scripting services."""
from dataclasses import replace

import numpy as np
import pytest

from nfit.analysis import AnalysisContext, AnalysisInput, SpectralConvention
from nfit.analysis.bragg import integrate_bragg_peaks
from nfit.analysis.registry import default_analysis_parameters, run_analysis_operation
from nfit.analysis.runner import prepare_analysis_input
from nfit.analysis.spectral import spectral_kernel
from nfit.cross_section import MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ, bose_denominator
from nfit.dataset import PointData4D
from nfit.fit_config import FitDatasetInput, compile_fit_problem
from nfit.fitting import evaluate_problem_model
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.measurement_dependencies import SourceReplayRequired
from nfit.measurement_fit_data import prepare_histogram_fit_points
from nfit.metadata_dimensions import MetadataDimension, discrete_metadata_axis
from nfit.pipeline import DataGroup, DatasetEntry, ModelComponentSpec
from nfit.sample_temperature import data_temperature, dataset_temperature


def energy_histogram(*, metadata=None, channels=None):
    return MDHistoData((MDHistoAxis("DeltaE", [0.,2.,4.], "meV", "energy"),),
        np.array([1.,1.]), np.array([.1,.1]), np.zeros(2,bool), np.ones(2),
        metadata=metadata or {}, auxiliary_channels=channels or {})


def item(data, *, temperature=None, lattice=None):
    return AnalysisInput("a"*32,"input",data,
        AnalysisContext("group",lattice,"P1",None,temperature),"fingerprint")


def convention(representation="chi_double_prime", *, moment="mu_B_squared", states="removed", g=2.):
    unit = "barn/(sr meV)" if representation == "cross_section" else "mu_B^2/meV" if moment == "mu_B_squared" else "spin^2/meV"
    return SpectralConvention(representation,unit,"per_formula_unit",1.,moment,g,
        states,states,states,"removed",True)


def run(operation, data, **updates):
    context_temperature=updates.pop("context_temperature",None)
    lattice=updates.pop("lattice",None)
    parameters={**default_analysis_parameters(operation),**updates}
    return run_analysis_operation(operation,[item(data,temperature=context_temperature,lattice=lattice)],parameters)


def temperature_histogram():
    axis=discrete_metadata_axis(MetadataDimension("Temperature","parameters/temperature","K"),[10.,20.])
    return MDHistoData((axis,MDHistoAxis("DeltaE",[0.,2.,4.],"meV","energy")),
        np.ones((2,2)),np.full((2,2),.1),np.zeros((2,2),bool),np.ones((2,2)),
        metadata={"temperature":99.})


def test_manual_temperature_override_precedes_imported_metadata_and_reaches_analysis():
    data=energy_histogram(metadata={"temperature":10.})
    dataset=DatasetEntry("input",data,data_type="powder_inelastic",parameters={"temperature":30.})
    group=DataGroup("group",datasets=[dataset])
    assert data_temperature(data) == 10.
    assert dataset_temperature(dataset) == 30.
    prepared=prepare_analysis_input(group,dataset)
    assert prepared.context.temperature_K == 30.


def test_temperature_coordinate_precedes_manual_override_and_flattens_for_fitting():
    data=temperature_histogram()
    dataset=DatasetEntry("input",data,parameters={"temperature":30.})
    np.testing.assert_array_equal(data_temperature(data),[[10.],[20.]])
    np.testing.assert_array_equal(dataset_temperature(dataset),[[10.],[20.]])
    np.testing.assert_array_equal(prepare_histogram_fit_points(data).temperature,[10.,10.,20.,20.])


@pytest.mark.parametrize("metadata", [{"temperature_K":15.},{"spectral_observable":{"temperature_K":[10.,20.]}}])
def test_temperature_alias_and_spectral_array_fallback_reach_fit_points(metadata):
    data=energy_histogram(metadata=metadata)
    resolved=data_temperature(data)
    fitted=prepare_histogram_fit_points(data).temperature
    np.testing.assert_array_equal(fitted,resolved)


def test_registered_energy_only_conversion_uses_unity_form_factor_without_q_lookup(monkeypatch):
    import nfit.analysis.input_preparation as preparation

    def unexpected_lookup(*args,**kwargs):
        raise AssertionError("Unity form factor must not require momentum coordinates")
    monkeypatch.setattr(preparation,"q_modulus_for_spectral",unexpected_lookup)
    data=energy_histogram(metadata={"temperature_K":25.,"nfit_data_type":"powder_inelastic"})
    source=convention("cross_section",states="included")
    result=run("spectral_conversion",data,spectral_convention=source.to_dict()).outputs["converted"]
    expected=np.pi*bose_denominator([1.,3.],25.)/MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ
    np.testing.assert_allclose(result.data.signal,expected)
    assert result.data_type == "powder_inelastic"
    assert result.data.metadata["temperature"] == 25.


def test_registered_auxiliary_input_uses_its_own_values_and_uncertainties():
    alternate=MDHistoChannel(np.array([2.,4.]),np.array([.3,.4]),unit="mu_B^2/meV",quantity_type="dynamic_susceptibility")
    data=energy_histogram(metadata={"temperature":25.},channels={"response":alternate})
    result=run("spectral_integration",data,spectral_convention=convention().to_dict(),
        kernel="energy_moment",power=0,energy_max_meV=4.,input_channel="response").outputs["energy_moment"]
    assert result.value == pytest.approx(12.)
    assert result.uncertainty == pytest.approx(1.)


@pytest.mark.parametrize("channel,expected", [("missing","Unknown input channel"),("response","no measurement uncertainties")])
def test_registered_auxiliary_input_rejects_missing_or_unknown_errors(channel,expected):
    data=energy_histogram(metadata={"temperature":25.},channels={"response":MDHistoChannel(np.array([2.,4.]))})
    with pytest.raises(ValueError,match=expected):
        run("spectral_integration",data,spectral_convention=convention().to_dict(),input_channel=channel)


def test_auxiliary_input_does_not_reuse_shared_primary_source_dependencies():
    data=energy_histogram(metadata={"temperature":25.,"measurement_contract":{"dependence":"shared_sources"}},
        channels={"response":MDHistoChannel(np.array([2.,4.]),np.array([.3,.4]))})
    with pytest.raises(SourceReplayRequired,match="own source dependencies"):
        run("spectral_integration",data,spectral_convention=convention().to_dict(),input_channel="response")


def test_registered_scalar_form_factor_and_top_level_g_are_applied_once():
    data=energy_histogram(metadata={"temperature":25.})
    source=convention("cross_section",states="included",moment="spin_squared",g=2.)
    result=run("spectral_integration",data,spectral_convention=source.to_dict(),kernel="energy_moment",power=0,
        energy_max_meV=4.,form_factor_sq=.5,g_factor=3.).outputs["energy_moment"]
    factor=np.pi*bose_denominator([1.,3.],25.)/(MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ*.5*3.**2)
    assert result.value == pytest.approx(np.sum(factor*2.))
    assert result.uncertainty == pytest.approx(np.sqrt(np.sum((.1*factor*2.)**2)))


def test_registered_top_level_g_controls_spin_qfi_from_moment_data():
    data=energy_histogram(metadata={"temperature":25.})
    result=run("spectral_integration",data,spectral_convention=convention(g=None).to_dict(),kernel="qfi",
        energy_max_meV=4.,g_factor=3.).outputs["qfi"]
    assert result.value == pytest.approx(np.sum(spectral_kernel("qfi",np.array([1.,3.]),25.)*2.)/9.)


def test_registered_conversion_flattens_temperature_array_for_downstream_fitting():
    data=temperature_histogram()
    result=run("spectral_conversion",data,spectral_convention=convention("cross_section",states="included").to_dict()).outputs["converted"].data
    np.testing.assert_array_equal(prepare_histogram_fit_points(result).temperature,[10.,10.,20.,20.])


@pytest.mark.parametrize("fit_representation,moment",[("chi_double_prime","mu_B_squared"),("cross_section","mu_B_squared")])
def test_absolute_spin_model_requires_explicit_g_for_moment_mapping(fit_representation,moment):
    points=PointData4D([0.,0.],[0.,0.],[0.,0.],[1.,3.],[1.,1.],[.1,.1],temperature=25.,
        metadata={"spectral_observable":{"fit_representation":fit_representation,"moment_unit":moment,
            "absolute_scale":True,"g_factor":None,"unit":"mu_B^2/meV","polarization_mode":"isotropic_single_component"}})
    component=ModelComponentSpec("loc","local_relaxational",parameters={"chi_loc":1.,"gamma":2.},fit_parameters={})
    with pytest.raises(ValueError,match="explicit g_factor"):
        compiled=compile_fit_problem([component],[FitDatasetInput("data",points)])
        params={spec.name:spec.value for spec in compiled.problem.parameter_specs}
        evaluate_problem_model(compiled.problem,"data",params)


def test_absolute_spin_chipp_can_fit_without_a_moment_unit_conversion():
    points=PointData4D([0.,0.],[0.,0.],[0.,0.],[1.,3.],[1.,1.],[.1,.1],temperature=25.,
        metadata={"spectral_observable":{"fit_representation":"chi_double_prime","moment_unit":"spin_squared",
            "absolute_scale":True,"g_factor":None,"unit":"spin^2/meV","polarization_mode":"isotropic_single_component"}})
    component=ModelComponentSpec("loc","local_relaxational",parameters={"chi_loc":1.,"gamma":2.},fit_parameters={})
    compiled=compile_fit_problem([component],[FitDatasetInput("data",points)])
    params={spec.name:spec.value for spec in compiled.problem.parameter_specs}
    assert np.all(np.isfinite(evaluate_problem_model(compiled.problem,"data",params)))


def test_registered_bragg_supplies_group_lattice_context_to_physical_region():
    axes=tuple(MDHistoAxis(name,[0.,1.,2.],"rlu","momentum") for name in ("H","K","L"))
    data=MDHistoData(axes,np.full((2,2,2),10.),np.ones((2,2,2)),np.zeros((2,2,2),bool),np.ones((2,2,2)),metadata={"signal_semantics":"density"})
    lattice={"a":2*np.pi,"b":2*np.pi,"c":2*np.pi}
    parameters=dict(peak_positions_hkl=[[1.,1.,1.]],method="box_sum",coordinate_frame="q_angstrom_inverse",
        box_half_widths=[.5]*3,background_mode="none")
    actual=run("bragg_integration",data,lattice=lattice,**parameters).outputs["peak_table"].data
    expected=integrate_bragg_peaks(data.with_updates(metadata={**data.metadata,"lattice_parameters":lattice}),
        [[1.,1.,1.]],method="box_sum",coordinate_frame="q_angstrom_inverse",box_half_widths=[.5]*3)
    np.testing.assert_allclose(actual.column("I"),expected.column("I"))
    assert np.isfinite(actual.column("I")[0]) and actual.column("I")[0] > 0.


def test_temperature_and_analysis_routing_services_remain_gui_independent():
    import ast
    from pathlib import Path

    import nfit.analysis.input_preparation as preparation
    import nfit.sample_temperature as temperature

    for module in (preparation,temperature):
        tree=ast.parse(Path(module.__file__).read_text())
        imports=[node.module or "" for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        assert all("gui" not in name.lower() and "qt" not in name.lower() for name in imports)
        assert "project_data" not in imports


@pytest.mark.parametrize("field", ["g_factor", "magnetic_ions_per_basis", "incident_energy_meV", "final_energy_meV"])
@pytest.mark.parametrize("value", [np.nan, np.inf, 0., -1.])
def test_optional_spectral_numbers_require_finite_positive_values(field, value):
    with pytest.raises(ValueError, match=f"{field} must be finite and positive"):
        replace(convention(), **{field:value})


def test_auxiliary_input_rejects_uncontracted_retained_primary_dependencies():
    from nfit.measurement_dependencies import independent_source_dependencies

    data=energy_histogram(metadata={"temperature":25.},
        channels={"response":MDHistoChannel(np.array([2.,4.]),np.array([.3,.4]))})
    data=data.with_updates(source_dependencies=independent_source_dependencies(data.errors**2,"primary"))
    with pytest.raises(SourceReplayRequired,match="own source dependencies"):
        run("spectral_integration",data,spectral_convention=convention().to_dict(),input_channel="response")


def test_paired_temperature_override_is_authoritative_for_fit_preparation():
    from nfit.spectral_channels import (
        default_spectral_channel_config,
        with_paired_spectral_channels,
    )

    data=energy_histogram(metadata={"temperature":5.})
    paired=with_paired_spectral_channels(data,default_spectral_channel_config(),temperature_K=20.)
    assert paired.metadata["temperature"] == 20.
    assert prepare_histogram_fit_points(paired).temperature == 20.


def test_fixed_temperature_cannot_replace_a_varying_experimental_coordinate():
    with pytest.raises(ValueError,match="choose dataset/composite temperature"):
        run("spectral_conversion",temperature_histogram(),
            spectral_convention=convention("cross_section",states="included").to_dict(),
            temperature_source="fixed",temperature_K=20.)


def test_fixed_temperature_may_agree_with_an_existing_perpoint_temperature():
    data=energy_histogram(metadata={"temperature":[20.,20.]})
    result=run("spectral_conversion",data,
        spectral_convention=convention("cross_section",states="included").to_dict(),
        temperature_source="fixed",temperature_K=20.).outputs["converted"].data
    assert prepare_histogram_fit_points(result).temperature == 20.
