"""Manual ordinary sequential Shiver/Mantid workflow benchmark.

Run with the existing Shiver interpreter, never with production nfit or pytest.
The installed GenerateDGSMDE is invoked directly: no parallel reduction jobs,
alternate reduction recipe, preexisting MDE shortcut, or smoothing. An initial
resident workflow and later saved-workspace reopening/rebinning are reported
separately. Full diagnostic arrays are written outside the pipeline timings.

Plan without Mantid or scientific I/O::

    python benchmark_dgs_mantid_workflow.py --plan-only --first-runs 16

Scientific products, progress journals and partial receipts must remain within
the configured IPTS/shared/nfit/benchmarks/6A-P-node19 output tree. The benchmark
refuses an existing job directory and never writes input files.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import functools
import hashlib
import importlib
import json
import os
import platform
import resource
import socket
import sys
import time
import traceback
from pathlib import Path

from benchmark_dgs_settings import load_settings, output_directory, select_runs


def _utc():
    return datetime.datetime.now(datetime.UTC).isoformat()


def _memory():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return dict(peak_rss_mib=usage.ru_maxrss / scale, user_seconds=usage.ru_utime,
                system_seconds=usage.ru_stime)


class Recorder:
    def __init__(self, output, settings):
        self.output = output
        self.receipt = dict(schema_version=1, engine="mantid_shiver", status="running",
            started_utc=_utc(), settings=settings, runs=[], stages=[], products={},
            timing_boundaries=dict(primary="imports/setup + actual sequential GenerateDGSMDE + merged SaveMD + resident MDNorm + histogram SaveMD",
                reopen="Delete resident workspaces; LoadMD saved MDE and primary histogram separately",
                rebin="saved MDE resident after one timed reload; each MDNorm and SaveMD separately",
                diagnostics="full parity arrays, digests and preflight validation excluded from pipeline sums",
                instrumentation="journaling overhead remains in outer GenerateDGSMDE wall time",
                filesystem="normal buffered saves; no invented fsync or cold-cache time"))
        self.current_run = None
        self._counter = 0
        self.journal = (output / "progress.jsonl").open("x", buffering=1)
        self.flush()

    def flush(self):
        self.receipt["updated_utc"] = _utc()
        self.receipt["resources"] = _memory()
        temporary = self.output / "receipt.partial.json.new"
        temporary.write_text(json.dumps(self.receipt, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.output / "receipt.partial.json")

    def event(self, kind, **fields):
        record = dict(event=kind, utc=_utc(), **fields)
        self.journal.write(json.dumps(record, sort_keys=True) + "\n")
        self.journal.flush()
        print(json.dumps(record, sort_keys=True), flush=True)

    def call(self, category, name, function, *args, **kwargs):
        self._counter += 1
        identity = self._counter
        run = None if self.current_run is None else self.current_run["index"]
        self.event("stage_started", id=identity, category=category, name=name, run_index=run)
        self.receipt["active_stage"] = dict(id=identity, category=category, name=name, run_index=run)
        if category != "child_algorithm":
            self.flush()
        before = _memory()
        start = time.perf_counter()
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            self.event("stage_failed", id=identity, name=name, error=str(error))
            raise
        seconds = time.perf_counter() - start
        after = _memory()
        record = dict(id=identity, category=category, name=name, run_index=run,
            wall_seconds=seconds, user_seconds=after["user_seconds"] - before["user_seconds"],
            system_seconds=after["system_seconds"] - before["system_seconds"],
            peak_rss_mib=after["peak_rss_mib"])
        self.receipt["stages"].append(record)
        if self.current_run is not None:
            self.current_run.setdefault("stage_ids", []).append(identity)
        self.event("stage_complete", **record)
        if category != "child_algorithm":
            self.receipt.pop("active_stage", None)
            self.flush()
        return result


def _file_stat(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _instrument_installed_shiver(recorder, conversion, generation):
    """Observe installed child calls; only temporary ADS names are made explicit.

    SimpleAPI infers output names from its caller's assignment. Wrapping that
    call would alter name inference, so absent output names are explicitly made
    unique. All scientific properties and installed control flow are preserved.
    """
    restores = []
    counter = 0
    output_algorithms = {"LoadEventNexus", "LoadNexusMonitors", "LoadNexusProcessed",
                         "FilterByLogValue", "FilterBadPulses", "CropWorkspace",
                         "DgsReduction", "CropWorkspaceForMDNorm"}
    names = {"LoadEventNexus", "LoadNexusMonitors", "LoadNexusProcessed", "CloneWorkspace",
             "FilterByLogValue", "FilterBadPulses", "MaskBTP", "MaskDetectors", "SetGoniometer",
             "GetEi", "GetEiT0atSNS", "CropWorkspace", "RotateInstrumentComponent",
             "DgsReduction", "CropWorkspaceForMDNorm", "ConvertToMDMinMaxGlobal", "ConvertToMD",
             "MergeMD", "DeleteWorkspace", "DeleteWorkspaces", "SetUB", "Comment"}
    def wrap(original, name):
        @functools.wraps(original)
        def observed(*args, **kwargs):
            nonlocal counter
            counter += 1
            if name in output_algorithms and "OutputWorkspace" not in kwargs:
                kwargs["OutputWorkspace"] = f"__benchmark_{name}_{counter}"
            result = recorder.call("child_algorithm", name, original, *args, **kwargs)
            run = recorder.current_run
            if run is not None and name == "LoadEventNexus":
                run.setdefault("loaded_raw_files", []).append(str(kwargs.get("Filename", args[0] if args else "")))
                run["raw_events"] = int(result.getNumberEvents())
                run["raw_charge_uah"] = float(result.run().getProtonCharge())
            if run is not None and name in {"FilterByLogValue", "FilterBadPulses", "CropWorkspace"}:
                run[name] = dict(events=int(result.getNumberEvents()), charge_uah=float(result.run().getProtonCharge()))
            if run is not None:
                run.setdefault("stage_call_counts", {})[name] = run.get("stage_call_counts", {}).get(name, 0) + 1
            return result
        return observed
    for module in (conversion, generation):
        for name in names:
            if hasattr(module, name):
                original = getattr(module, name)
                restores.append((module, name, original))
                setattr(module, name, wrap(original, name))
    original_ei = conversion.get_Ei_T0
    def resolved_ei(*args, **kwargs):
        result = original_ei(*args, **kwargs)
        if recorder.current_run is not None:
            recorder.current_run.update(ei_mev=float(result[0]), t0_us=float(result[1]))
        return result
    restores.append((conversion, "get_Ei_T0", original_ei))
    conversion.get_Ei_T0 = resolved_ei
    original_run = generation.ConvertDGSToSingleMDE
    def converted_run(*args, **kwargs):
        index = len(recorder.receipt["runs"])
        record = dict(index=index, filenames=str(kwargs.get("Filenames", "")), started_utc=_utc())
        recorder.current_run = record
        recorder.receipt["active_run"] = record
        recorder.event("run_started", **record)
        recorder.flush()
        start = time.perf_counter()
        try:
            result = original_run(*args, **kwargs)
            record.update(wall_seconds=time.perf_counter() - start, mde_events=int(result.getNEvents()),
                          completed_utc=_utc(), peak_rss_mib=_memory()["peak_rss_mib"])
            recorder.receipt["runs"].append(record)
            recorder.receipt.pop("active_run", None)
            recorder.event("run_complete", **record)
            recorder.flush()
            return result
        finally:
            recorder.current_run = None
    restores.append((generation, "ConvertDGSToSingleMDE", original_run))
    generation.ConvertDGSToSingleMDE = converted_run
    return restores


def _generate_arguments(settings, ub):
    reduction = settings["reduction"]
    result = dict(Filenames=",".join(settings["raw_files"]), Type="Data",
                  OutputWorkspace="benchmark_merged", UBParameters=json.dumps({"UB": ub.ravel().tolist()}),
                  ApplyFilterBadPulses=reduction["bad_pulses_threshold"] > 0,
                  TimeIndependentBackground=reduction["time_independent_background"])
    for key, value in (("NormFilename", settings["normalization_file"]), ("MaskFilename", settings["mask_file"]),
                       ("Ei", reduction["ei_override"]), ("T0", reduction["t0_override"])):
        if value is not None:
            result[key] = value
    if result["ApplyFilterBadPulses"]:
        result["BadPulsesThreshold"] = reduction["bad_pulses_threshold"]
    if reduction["goniometer"] != "Universal":
        result["OmegaMotorName"] = reduction["goniometer"]
    if (reduction["emin_fraction"], reduction["emax_fraction"]) != (-.95, .95):
        if reduction["ei_override"] is None:
            raise ValueError("Custom energy fractions require an explicit Ei for the installed GenerateDGSMDE properties")
        result["EMin"] = reduction["ei_override"] * reduction["emin_fraction"]
        result["EMax"] = reduction["ei_override"] * reduction["emax_fraction"]
    return result


def _bin_arguments(settings, copies, md, normalization):
    arguments = dict(InputWorkspace=md, OutputWorkspace="benchmark_histogram",
        OutputDataWorkspace="benchmark_numerator", OutputNormalizationWorkspace="benchmark_normalization",
        SymmetryOperations=settings["symmetry_operations"][str(copies)])
    if normalization is not None:
        arguments["SolidAngleWorkspace"] = normalization
    for dim, vector in enumerate(settings["grid"]["vectors"]):
        arguments[f"QDimension{dim}"] = ",".join(str(value) for value in vector)
    for dim, edges in enumerate(settings["grid"]["bin_edges"]):
        arguments[f"Dimension{dim}Name"] = f"QDimension{dim}" if dim < 3 else "DeltaE"
        arguments[f"Dimension{dim}Binning"] = ",".join(format(value, ".17g") for value in
            (edges[0], (edges[-1] - edges[0]) / (len(edges) - 1), edges[-1]))
    return arguments


def _dump_arrays(path, result, numerator, normalization, settings, copies, h5py, np):
    """Bounded diagnostic write after actual SaveMD; never part of bin timings."""
    arrays = dict(numerator=numerator.getSignalArray(), variance=numerator.getErrorSquaredArray(),
                  normalization=normalization.getSignalArray(), counts=numerator.getNumEventsArray(),
                  signal=result.getSignalArray(), errors_squared=result.getErrorSquaredArray())
    dimensions = [result.getDimension(index) for index in range(result.getNumDims())]
    with h5py.File(path, "x") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["settings"] = json.dumps(settings, sort_keys=True)
        handle.attrs["symmetry_copies"] = copies
        handle.attrs["axis_order"] = "QDimension0,QDimension1,QDimension2,DeltaE"
        handle.attrs["counts_semantics"] = "Mantid MDNorm data workspace num_events; compare observed copy convention explicitly"
        handle.attrs["timing"] = "diagnostic only, excluded from pipeline times"
        for index, dimension in enumerate(dimensions):
            handle.create_dataset(f"edges/{index}", data=np.linspace(dimension.getMinimum(),
                dimension.getMaximum(), dimension.getNBins() + 1))
        shape = arrays["signal"].shape
        outputs = {name: handle.create_dataset(name, shape=shape, dtype=array.dtype)
                   for name, array in arrays.items()}
        mask = handle.create_dataset("mask", shape=shape, dtype=bool)
        for first in range(shape[0]):
            for name, array in arrays.items():
                outputs[name][first] = array[first]
            mask[first] = ~np.isfinite(arrays["signal"][first]) | (arrays["normalization"][first] <= 0)
    return dict(path=str(path), shape=list(shape), datasets=list(arrays) + ["mask"],
                actual_dimensions=[[dim.getMinimum(), dim.getMaximum(), dim.getNBins()] for dim in dimensions],
                array_axis_order="original dimension order, no transpose",
                edges="edges/0,edges/1,edges/2,edges/3")


def _engine_imports():
    import h5py
    import mantid
    import numpy as np
    from mantid import config
    from mantid.simpleapi import DeleteWorkspaces, LoadMD, MDNorm, SaveMD, mtd

    conversion = importlib.import_module("shiver.models.convert_dgs_to_single_mde")
    generation = importlib.import_module("shiver.models.generate_dgs_mde")
    slicing = importlib.import_module("shiver.models.makeslice")
    shiver_version = importlib.import_module("shiver.version").__version__
    configuration = importlib.import_module("shiver.configuration")
    from mantid.simpleapi import GenerateDGSMDE

    return dict(h5py=h5py, mantid=mantid, np=np, config=config, conversion=conversion,
                generation=generation, slicing=slicing, GenerateDGSMDE=GenerateDGSMDE,
                shiver_version=shiver_version, configuration=configuration,
                DeleteWorkspaces=DeleteWorkspaces, LoadMD=LoadMD, MDNorm=MDNorm, SaveMD=SaveMD, mtd=mtd)


def _read_ub(path, h5py, np):
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["MDEventWorkspace/experiment0/sample/oriented_lattice/orientation_matrix"])


def benchmark(settings, output, *, primary_copies, rebin_copies, input_mde=None):
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    os.environ["TMPDIR"] = str(scratch)
    os.environ["OMP_NUM_THREADS"] = str(settings["threads"])
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    os.chdir(output)
    recorder = Recorder(output, settings)
    wall_start = time.perf_counter()
    restores = []
    try:
        engine = recorder.call("initial_setup", "imports_installed_mantid_shiver", _engine_imports)
        config, mtd = engine["config"], engine["mtd"]
        config["defaultsave.directory"] = str(output)
        config["datasearch.directories"] = str(output)
        config["MultiThreaded.MaxCores"] = str(settings["threads"])
        config["UpdateInstrumentDefinitions.OnStartup"] = "0"
        config["logging.channels.fileChannel.path"] = str(output / "mantid.log")
        recorder.receipt["environment"] = dict(host=socket.gethostname(), platform=platform.platform(),
            python=platform.python_version(), mantid=engine["mantid"].__version__, numpy=engine["np"].__version__,
            shiver=engine["shiver_version"],
            executable=sys.executable, cpu_count=os.cpu_count(),
            cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            internal_mantid_max_cores=config["MultiThreaded.MaxCores"], external_parallel_jobs=1,
            ram_budget_mib=settings["ram_limit_mib"],
            ram_budget_enforcement="reported common budget; Mantid has no matching native allocator limit",
            thread_environment={name: os.environ.get(name) for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        # Provenance and file membership validation are deliberately untimed.
        originals = [*settings["raw_files"], settings["ub_source"],
                     *(path for path in (settings["normalization_file"], settings["mask_file"]) if path)]
        original_stats = {str(path): _file_stat(path) for path in dict.fromkeys(originals)}
        recorder.receipt["inputs"] = original_stats
        recorder.receipt["installed_recipe"] = {name: dict(path=module.__file__,
            sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()) for name, module in
            (("conversion", engine["conversion"]), ("generation", engine["generation"]), ("slicing", engine["slicing"]))}
        configuration = engine["configuration"]
        recorder.receipt["shiver_configuration"] = dict(
            selected_logs=configuration.get_data_logs(),
            file=_file_stat(configuration.CONFIG_PATH_FILE) if Path(configuration.CONFIG_PATH_FILE).exists() else None,
            console_logging="redirect stdout/stderr into this IPTS output tree when launching")
        recorder.receipt["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        recorder.receipt["shared_settings_sha256"] = hashlib.sha256(Path(__file__).with_name("benchmark_dgs_settings.py").read_bytes()).hexdigest()
        ub = recorder.call("initial_setup", "read_reference_UB", _read_ub,
                           settings["ub_source"], engine["h5py"], engine["np"])
        recorder.receipt["UB"] = ub.tolist()
        restores = _instrument_installed_shiver(recorder, engine["conversion"], engine["generation"])
        if input_mde is None:
            arguments = _generate_arguments(settings, ub)
            recorder.receipt["GenerateDGSMDE_parameters"] = arguments
            md = recorder.call("resident_workflow", "GenerateDGSMDE_raw_sequential",
                               engine["GenerateDGSMDE"], **arguments)
            requested = [str(Path(path).resolve()) for path in settings["raw_files"]]
            loaded = [str(Path(path).resolve()) for run in recorder.receipt["runs"] for path in run["loaded_raw_files"]]
            if loaded != requested or len(recorder.receipt["runs"]) != len(requested):
                raise AssertionError("Installed Shiver did not freshly load every requested raw run in order")
            if any(run["stage_call_counts"].get(name, 0) != 1
                   for run in recorder.receipt["runs"] for name in ("DgsReduction", "ConvertToMD")):
                raise AssertionError("Fresh per-run DgsReduction/ConvertToMD coverage is incomplete")
            mde_path = output / "merged_mde.nxs"
            recorder.call("resident_workflow", "SaveMD_merged_MDE", engine["SaveMD"],
                          InputWorkspace=md, Filename=str(mde_path))
            recorder.receipt["products"]["merged_mde"] = _file_stat(mde_path)
        else:
            mde_path = Path(input_mde).resolve()
            md = recorder.call("saved_input_setup", "LoadMD_existing_merged", engine["LoadMD"],
                               Filename=str(mde_path), OutputWorkspace="benchmark_merged")
            recorder.receipt["products"]["input_merged_mde"] = _file_stat(mde_path)
            if settings["normalization_file"]:
                recorder.call("saved_input_setup", "LoadNexusProcessed_normalization",
                    engine["generation"].LoadNexusProcessed, Filename=settings["normalization_file"],
                    OutputWorkspace=Path(settings["normalization_file"]).stem)
        norm_name = None if settings["normalization_file"] is None else Path(settings["normalization_file"]).stem
        normalization = None if norm_name is None else mtd[norm_name]
        recorder.receipt["merged_dimensions"] = [md.getDimension(dim).getName() for dim in range(md.getNumDims())]
        recorder.receipt["merged_events"] = int(md.getNEvents())
        recorder.receipt["merged_experiments"] = int(md.getNumExperimentInfo())
        def bin_and_save(copies, category, label):
            arguments = _bin_arguments(settings, copies, md, normalization)
            recorder.receipt.setdefault("MDNorm_parameters", {})[label] = {
                key: str(value) if key in {"InputWorkspace", "SolidAngleWorkspace"} else value
                for key, value in arguments.items()}
            recorder.call(category, f"MDNorm_{label}_{copies}copies", engine["MDNorm"], **arguments)
            result, numerator, norm = (mtd[name] for name in
                ("benchmark_histogram", "benchmark_numerator", "benchmark_normalization"))
            histogram_path = output / f"histogram_{label}_{copies}copies.nxs"
            recorder.call(category, f"SaveMD_histogram_{label}_{copies}copies", engine["SaveMD"],
                          InputWorkspace=result, Filename=str(histogram_path))
            recorder.receipt["products"][label] = _file_stat(histogram_path)
            arrays = recorder.call("diagnostic_only", f"full_arrays_{label}_{copies}copies", _dump_arrays,
                output / f"arrays_{label}_{copies}copies.h5", result, numerator, norm, settings, copies,
                engine["h5py"], engine["np"])
            recorder.receipt["products"][f"{label}_arrays"] = arrays
            recorder.call("cleanup", f"Delete_histograms_{label}", engine["DeleteWorkspaces"],
                ["benchmark_histogram", "benchmark_numerator", "benchmark_normalization"])
            return histogram_path
        primary_path = bin_and_save(primary_copies, "resident_workflow", "primary")
        # Reopening is a separately reported real user scenario, rather than a
        # forced extra load in the resident reduction-to-histogram headline.
        recorder.call("reopen_cleanup", "Delete_resident_MDE", engine["DeleteWorkspaces"], ["benchmark_merged"])
        del md
        md = recorder.call("reopen", "LoadMD_saved_MDE", engine["LoadMD"],
                           Filename=str(mde_path), OutputWorkspace="benchmark_merged")
        reopened = recorder.call("reopen", "LoadMD_saved_primary_histogram", engine["LoadMD"],
                                 Filename=str(primary_path), OutputWorkspace="benchmark_reopened_histogram")
        recorder.receipt["reopened_histogram_dimensions"] = [reopened.getDimension(dim).getNBins() for dim in range(reopened.getNumDims())]
        recorder.call("reopen_cleanup", "Delete_reopened_histogram", engine["DeleteWorkspaces"], ["benchmark_reopened_histogram"])
        del reopened
        for copies in rebin_copies:
            bin_and_save(copies, "saved_MDE_rebin", f"rebin{copies}")
        # Input identity checks and receipt aggregation are outside the science
        # timers. A completed result must not have modified any original source.
        if original_stats != {str(path): _file_stat(path) for path in dict.fromkeys(originals)}:
            raise AssertionError("A benchmark input changed during execution")
        totals = {}
        for stage in recorder.receipt["stages"]:
            if stage["category"] != "child_algorithm":
                totals[stage["category"]] = totals.get(stage["category"], 0.) + stage["wall_seconds"]
        recorder.receipt["category_seconds"] = totals
        recorder.receipt["primary_resident_workflow_seconds"] = totals.get("initial_setup", 0.) + totals.get("resident_workflow", 0.)
        recorder.receipt["reopen_seconds"] = totals.get("reopen", 0.)
        recorder.receipt["process_wall_seconds_including_diagnostics"] = time.perf_counter() - wall_start
        recorder.receipt["status"] = "complete"
        recorder.receipt["completed_utc"] = _utc()
        recorder.flush()
        (output / "receipt.partial.json").replace(output / "receipt.json")
        recorder.event("benchmark_complete", primary_resident_workflow_seconds=recorder.receipt["primary_resident_workflow_seconds"])
    except BaseException as error:
        recorder.receipt.update(status="failed", error=str(error), traceback=traceback.format_exc())
        recorder.flush()
        raise
    finally:
        for module, name, original in reversed(restores):
            setattr(module, name, original)
        recorder.journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=os.environ.get("NFIT_DGS_BENCHMARK_CONFIG"))
    parser.add_argument("--preset", choices=["sequoia", "hyspec-50k34", "hyspec-50k70"], default="sequoia")
    parser.add_argument("--tag", default="full617")
    parser.add_argument("--first-runs", type=int)
    parser.add_argument("--runs", type=int, nargs="+")
    parser.add_argument("--primary-copies", type=int, choices=[1, 6, 12])
    parser.add_argument("--rebin-copies", type=int, choices=[1, 6, 12], nargs="*")
    parser.add_argument("--input-mde", type=Path, help="Explicit saved-MDE-only benchmark; no raw reduction claim")
    parser.add_argument("--plan-only", action="store_true", help="Print configuration without importing Mantid or reading scientific inputs")
    arguments = parser.parse_args()
    settings = select_runs(load_settings(arguments.config, arguments.preset), arguments.runs, arguments.first_runs)
    primary = settings["primary_copies"] if arguments.primary_copies is None else arguments.primary_copies
    rebins = settings["rebin_copies"] if arguments.rebin_copies is None else arguments.rebin_copies
    output = output_directory(settings, "mantid", arguments.tag)
    if arguments.plan_only:
        print(json.dumps(dict(settings=settings, output=str(output), primary_copies=primary,
                             rebin_copies=rebins, input_mde=None if arguments.input_mde is None else str(arguments.input_mde)), indent=2))
        return
    # The orchestrator redirects Mantid/Python console output into the IPTS tree.
    # This harness also writes bounded JSON progress and partial receipts there.
    benchmark(settings, output, primary_copies=primary, rebin_copies=rebins, input_mde=arguments.input_mde)


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        main()
