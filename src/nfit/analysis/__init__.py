"""Pure analysis domain contracts and operation registry."""

from .bragg import bragg_volume
from .builtins import register_builtin_operations
from .coordinates import (
    bin_edges,
    bin_widths,
    energy_bin_widths,
    measured_mask,
    physical_axis_vectors,
    physical_coordinate_arrays,
    q_bin_volume,
    q_modulus_for_spectral,
    rlu_to_q_matrix,
    signal_semantics,
)
from .core import (
    AnalysisContext,
    AnalysisEntry,
    AnalysisExecution,
    AnalysisInput,
    AnalysisOutput,
    AnalysisOutputRef,
    AnalysisResultRecord,
    DatasetOutput,
    ScalarOutput,
    TableOutput,
    new_analysis_id,
)
from .corrections import SpectralConvention
from .registry import (
    AnalysisOperationDefinition,
    AnalysisParameterDefinition,
    analysis_definition,
    analysis_parameter_tooltip,
    available_analysis_types,
    default_analysis_parameters,
    register_analysis_operation,
    run_analysis_operation,
    validate_analysis,
)
from .runner import (
    prepare_analysis_input,
    prepare_analysis_inputs,
    run_project_analysis,
)

register_builtin_operations()

__all__ = [
    "AnalysisContext",
    "AnalysisEntry",
    "AnalysisExecution",
    "AnalysisInput",
    "AnalysisOperationDefinition",
    "AnalysisOutput",
    "AnalysisOutputRef",
    "AnalysisParameterDefinition",
    "AnalysisResultRecord",
    "DatasetOutput",
    "ScalarOutput",
    "SpectralConvention",
    "bin_edges",
    "bin_widths",
    "bragg_volume",
    "energy_bin_widths",
    "measured_mask",
    "physical_axis_vectors",
    "physical_coordinate_arrays",
    "prepare_analysis_input",
    "prepare_analysis_inputs",
    "q_bin_volume",
    "q_modulus_for_spectral",
    "rlu_to_q_matrix",
    "signal_semantics",
    "TableOutput",
    "analysis_definition",
    "analysis_parameter_tooltip",
    "available_analysis_types",
    "default_analysis_parameters",
    "new_analysis_id",
    "register_analysis_operation",
    "run_analysis_operation",
    "run_project_analysis",
    "validate_analysis",
]
