"""Public MACS settings and migration of ordinary SPEC/DIFF source groups."""

from __future__ import annotations

import copy

MACS_REDUCTION_FORMAT = "macs-step-nexus"
MACS_REDUCTION_VERSION = 3
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
    "vanadium_files": [],
    "vanadium_a4_mask_deg": 15.0,
    "background_files": [],
    "background_delta_e_tolerance_meV": 0.05,
    "background_method": "replay",
    "background_scale": 1.0,
    "background_q_bin_width": 0.05,
    "mask_misaligned_analyzers": True,
    "analyzer_alignment_tolerance_deg": 1.0,
    "detect_dead_analyzers": False,
    "dead_channel_min_nonzero_fraction": 0.2,
    "masked_analyzer_channels": [],
}


def macs_settings_schema():
    """Describe source reduction and shared composite background settings."""
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
        Field("vanadium_files", "SPEC vanadium energy scans", "files", [],
              "Select one or more compatible MACS vanadium energy scans at complementary kidney positions. Fit monitor-normalized Gaussian peak areas to calibrate SPEC channels. Uncovered or failed channels are masked. DIFF is unaffected. Clear the list to disable.", scope="normalization"),
        Field("vanadium_a4_mask_deg", "Vanadium A4 exclusion half-angle", "float", 15.0,
              "Exclude vanadium channels within this angle of the direct beam, including both boundaries (-angle through +angle). This only selects calibration observations; sample channel masks are separate.",
              units="deg", scope="normalization", minimum=0.0, maximum=180.0),
        Field("background_files", "Background scans", "files", [],
              "Select background files for this composite. Matching sample runs subtract their background with the same stream and normalization, including vanadium sensitivity. Runs without an energy match remain unsubtracted. Background files do not become selectable datasets. Clear the list to disable.",
              per_run=False, reduced_event_dependency=False),
        Field("background_delta_e_tolerance_meV", "Background energy-transfer tolerance", "float", 0.05,
              "Maximum absolute difference in energy transfer (Ei minus Ef) between a sample run and a background scan. Runs outside this tolerance remain unsubtracted.",
              units="meV", minimum=0.0, per_run=False, reduced_event_dependency=False),
        Field("background_method", "Background subtraction method", "choice", "replay",
              "Replay measured background by detector angle, or pool it as an isotropic radial background. Subtraction is applied when preparing the composite histogram.",
              choices=(("replay", "Detector-angle replay"), ("isotropic", "Isotropic radial background")),
              per_run=False, reduced_event_dependency=False),
        Field("background_scale", "Background scale", "float", 1.0,
              "Dimensionless multiplier applied to the monitor-normalized background before subtraction. One uses the measured background exposure directly.",
              per_run=False, reduced_event_dependency=False),
        Field("background_q_bin_width", "Isotropic background radial bin width", "float", 0.05,
              "Positive width in momentum-transfer magnitude for isotropic background pooling. Used only with the isotropic subtraction method.",
              units="angstrom^-1", minimum=0.0, per_run=False, reduced_event_dependency=False),
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
        per_run = {field.key for field in macs_settings_schema() if field.per_run}
        for entry in group.datasets:
            options = entry.metadata.get("import_options", {})
            overrides = {key: copy.deepcopy(value) for key, value in options.items()
                         if key in per_run and value != config.get(key)}
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
