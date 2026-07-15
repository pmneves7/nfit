from dataclasses import replace

import pytest

from nfit.analysis import (
    AnalysisContext,
    AnalysisExecution,
    AnalysisInput,
    AnalysisOperationDefinition,
    AnalysisParameterDefinition,
    ScalarOutput,
    SpectralConvention,
    analysis_parameter_tooltip,
    default_analysis_parameters,
    register_analysis_operation,
    run_analysis_operation,
)
from nfit.cross_section import chipp_from_intensity, intensity_from_chipp
from nfit.dataset import PointData4D
from nfit.pipeline import DataGroup, DatasetEntry


def _input() -> AnalysisInput:
    data = PointData4D([0.0], [0.0], [0.0], [1.0], [2.0], [0.1])
    return AnalysisInput(
        dataset_id="a" * 32,
        dataset_name="scan",
        data=data,
        context=AnalysisContext("group", None, None, None, None),
        fingerprint="fingerprint",
    )


def test_dataset_ids_are_stable_for_replace_and_new_for_copy():
    entry = DatasetEntry("scan", _input().data)

    assert len(entry.id) == 32
    int(entry.id, 16)
    assert replace(entry, name="renamed").id == entry.id
    assert entry.copy(name="copied").id != entry.id
    assert DataGroup("group").analyses == []


def test_analysis_registry_defaults_tooltips_validation_and_execution():
    operation = "test_scalar_operation"
    parameter = AnalysisParameterDefinition(
        name="scale",
        label="Scale",
        type="float",
        default=2.0,
        description="Multiply the result.",
        allowed="positive number",
        example="2.0",
        required=True,
    )

    def execute(inputs, parameters, **_callbacks):
        return AnalysisExecution(
            {"value": ScalarOutput(parameters["scale"], None, "a.u.", "Value")}
        )

    register_analysis_operation(
        AnalysisOperationDefinition(
            key=operation,
            label="Test scalar",
            version=1,
            description="Test operation.",
            min_inputs=1,
            max_inputs=1,
            accepted_containers=("PointData4D",),
            parameters=(parameter,),
            validate=lambda _inputs, params: (
                None
                if params["scale"] > 0
                else (_ for _ in ()).throw(ValueError("scale must be positive"))
            ),
            execute=execute,
        )
    )

    assert default_analysis_parameters(operation) == {"scale": 2.0}
    assert "Allowed: positive number" in analysis_parameter_tooltip(operation, "scale")
    result = run_analysis_operation(operation, [_input()], {"scale": 3.0})
    assert result.outputs["value"].value == 3.0

    with pytest.raises(ValueError, match="unknown"):
        run_analysis_operation(operation, [_input()], {"scale": 2.0, "extra": 1})
    with pytest.raises(ValueError, match="positive"):
        run_analysis_operation(operation, [_input()], {"scale": 0.0})


def test_reserved_analysis_types_cannot_be_registered():
    with pytest.raises(ValueError, match="reserved"):
        register_analysis_operation(
            AnalysisOperationDefinition(
                key="raw_tof_reduction",
                label="Raw TOF",
                version=1,
                description="Reserved.",
                min_inputs=1,
                max_inputs=None,
                accepted_containers=(),
                parameters=(),
                validate=lambda _inputs, _parameters: None,
                execute=lambda *_args, **_kwargs: AnalysisExecution({}),
            )
        )


def test_spectral_convention_and_cross_section_inverse():
    convention = SpectralConvention(
        representation="chi_double_prime",
        unit="mu_B^2/meV",
        normalization_basis="per_magnetic_ion",
        magnetic_ions_per_basis=1.0,
        moment_unit="mu_B_squared",
        g_factor=2.1,
        form_factor_state="removed",
        polarization_state="removed",
        bose_state="removed",
        kf_ki_state="removed",
        absolute_scale=True,
    )
    convention.require_absolute("QFI")
    assert SpectralConvention.from_dict(convention.to_dict()) == convention

    chipp = [0.2, 0.8]
    intensity = intensity_from_chipp(
        chipp, [1.0, 4.0], 12.0, scale=2.3, form_factor_sq=[0.9, 0.7], polarization=2.0
    )
    recovered = chipp_from_intensity(
        intensity,
        [1.0, 4.0],
        12.0,
        scale=2.3,
        form_factor_sq=[0.9, 0.7],
        polarization=2.0,
    )
    assert recovered == pytest.approx(chipp)
