"""Public MACS settings and migration of ordinary SPEC/DIFF source groups."""

from __future__ import annotations

import copy

MACS_REDUCTION_FORMAT = "macs-step-nexus"
MACS_REDUCTION_VERSION = 1
MACS_DEFAULTS = {
    "monitor_target": 1_000_000.0,
    "monitor_response": "inverse_velocity",
    "monitor_reference_k_inv_angstrom": 1.0,
    "ki_kf_normalization": True,
    "higher_order_monitor_correction": "off",
    "incident_filter": "from_logs",
    "incident_energy_policy": "bragg",
    "final_energy_policy": "aligned_mean",
    "incident_energy_override": None,
    "final_energy_override": None,
    "a3_offset_deg": None,
    "ub_matrix": None,
    "apply_detector_efficiency": False,
    "mask_misaligned_analyzers": True,
    "analyzer_alignment_tolerance_deg": 1.0,
    "detect_dead_analyzers": False,
    "dead_channel_min_nonzero_fraction": 0.2,
    "masked_analyzer_channels": [],
}


def macs_settings_schema():
    """Describe every adjustable importer setting; smoothing is a viewer action."""
    from .reduction_recipes import ReductionSetting as Field

    return (
        Field("monitor_target", "Monitor target", "float", MACS_DEFAULTS["monitor_target"],
              "Intensity scale in monitor counts. This changes units, not the relative exposure of observations.",
              units="counts", scope="normalization", minimum=0.0),
        Field("monitor_response", "Incident monitor response", "choice", "inverse_velocity",
              "A thin 1/v monitor counts more slow neutrons per unit incident flux. Correct its response before applying ki/kf. Constant leaves recorded monitor counts unchanged.",
              scope="normalization", choices=(("inverse_velocity", "1/v monitor"), ("constant", "Constant response"))),
        Field("monitor_reference_k_inv_angstrom", "Monitor reference wavevector", "float", 1.0,
              "Reference incident wavevector for the arbitrary intensity scale of a 1/v monitor. With ki/kf enabled the combined factor is reference k / kf.",
              units="angstrom^-1", scope="normalization", minimum=0.0),
        Field("ki_kf_normalization", "Apply ki/kf correction", "bool", True,
              "Remove the final-to-incident wavevector factor from SPEC scattering. The separate monitor response correction must also match the instrument. DIFF uses an elastic approximation.",
              scope="normalization"),
        Field("higher_order_monitor_correction", "Higher-order monitor correction", "choice", "off",
              "Optional historical DAVE lambda/2 calibration for unfiltered incident beams, restricted to 2–20 meV. It is not a measured calibration of this experiment and is disabled by default.",
              scope="normalization", choices=(("off", "None"), ("dave", "DAVE empirical lambda/2 model"))),
        Field("incident_filter", "Incident CFX filter", "choice", "from_logs",
              "Use recorded Be/HOPG incident-filter states, or override them for a run. Higher-order monitor correction applies only with both filters out. Unknown states require an explicit override.",
              scope="normalization", choices=(("from_logs", "From run logs"), ("in", "Inserted"), ("out", "Removed"))),
        Field("incident_energy_policy", "Incident energy source", "choice", "bragg",
              "Recover Ei from monochromator angle and d-spacing when available, or use the recorded energy log. Missing Bragg metadata falls back to the recorded energy.",
              choices=(("bragg", "Monochromator geometry"), ("recorded", "Recorded energy"))),
        Field("final_energy_policy", "SPEC final energy source", "choice", "aligned_mean",
              "Recover Ef from the mean aligned analyzer angle, use the common analyzer angle, or use the recorded energy. Missing geometry falls back to recorded Ef. DIFF always uses Ef=Ei.",
              choices=(("aligned_mean", "Mean aligned analyzer geometry"), ("common", "Common analyzer geometry"), ("recorded", "Recorded energy"))),
        Field("incident_energy_override", "Ei override", "float", None,
              "Positive incident energy override. Automatic uses the selected incident-energy source.", units="meV", automatic=True, minimum=0.0),
        Field("final_energy_override", "SPEC Ef override", "float", None,
              "Positive analyzed final energy override. Automatic uses the selected analyzer policy; DIFF is unaffected.", units="meV", automatic=True, minimum=0.0),
        Field("a3_offset_deg", "A3 offset", "float", None,
              "Offset added to the recorded sample rotation before projection. Automatic reads sampleState/a3Zero.", units="deg", automatic=True),
        Field("ub_matrix", "UB matrix", "matrix", None,
              "Optional reciprocal transform without 2 pi. An empty matrix uses the run's lattice and orientation metadata.", units="angstrom^-1", scope="coordinates", automatic=True, per_run=False),
        Field("apply_detector_efficiency", "Use recorded detector correction factors", "bool", False,
              "Apply the 20 multiplicative detectorEfficiency correction factors in each stream's NeXus logs. Default assumes equal sensitivity; enable only for a validated calibration.", scope="normalization"),
        Field("mask_misaligned_analyzers", "Mask misset SPEC analyzers", "bool", True,
              "Exclude SPEC analyzer blades outside the alignment tolerance. This does not mask DIFF detectors."),
        Field("analyzer_alignment_tolerance_deg", "Analyzer alignment tolerance", "float", 1.0,
              "Allowed analyzer two-theta deviation beyond the nearest blade to the common angle.", units="deg", minimum=0.0),
        Field("detect_dead_analyzers", "Detect unresponsive SPEC channels", "bool", False,
              "Mask channels with unusually low nonzero-count occupancy over at least 20 scan points. This is a health heuristic, not a sensitivity calibration; disable for sparse data."),
        Field("dead_channel_min_nonzero_fraction", "Dead-channel occupancy limit", "float", 0.2,
              "Upper limit on the occupancy threshold; the actual threshold is the smaller of this value and one quarter of median occupancy.", minimum=0.0, maximum=1.0),
        Field("masked_analyzer_channels", "Additional SPEC channel masks", "channels", [],
              "Comma-separated 1-based analyzer channels (1–20) excluded from SPEC counts and exposure. DIFF is unaffected."),
    )


def macs_group_config(group):
    """Read legacy or current homogeneous MACS groups without loading files."""
    config = group.metadata.get("macs")
    if isinstance(config, dict) and config.get("format") == MACS_REDUCTION_FORMAT:
        return config
    entries = group.datasets
    if group.subgroups or not entries or any(entry.metadata.get("importer") != "macs_nexus" for entry in entries):
        return None
    streams = {entry.metadata.get("import_options", {}).get("stream", "spec") for entry in entries}
    if len(streams) != 1:
        return None
    options = entries[0].metadata.get("import_options", {})
    return {"format": MACS_REDUCTION_FORMAT, "stream": streams.pop(),
            **copy.deepcopy(MACS_DEFAULTS), **copy.deepcopy(options)}


def adopt_macs_reduction(group):
    """Persist settings for a MACS group, preserving legacy per-file choices."""
    config = macs_group_config(group)
    if config is None:
        return False
    if "macs" not in group.metadata:
        group.metadata["macs"] = copy.deepcopy(config)
        for entry in group.datasets:
            options = entry.metadata.get("import_options", {})
            overrides = {key: copy.deepcopy(value) for key, value in options.items()
                         if key in MACS_DEFAULTS and value != config.get(key)}
            overrides.update(entry.metadata.get("reduction_overrides", {}))
            if overrides:
                entry.metadata["reduction_overrides"] = overrides
    return True


def invalidate_macs_source(dataset, config):
    """Make the next source load reduce with edited settings, never old arrays."""
    dataset.metadata["import_options"] = {key: copy.deepcopy(config.get(key, default))
                                          for key, default in MACS_DEFAULTS.items()}
    dataset.metadata["import_options"]["stream"] = config["stream"]
    dataset.metadata.pop("project_artifact_path", None)
    dataset.unload_data()
    dataset.metadata["import_status"] = "pending"
    if isinstance(dataset.metadata.get("resolved_reduction"), dict):
        dataset.metadata["resolved_reduction"]["stale"] = True
