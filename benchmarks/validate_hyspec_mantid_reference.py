"""Manual Mantid reference builder for raw HYSPEC DGS parity.

Run only with the existing ORNL Mantid interpreter. This diagnostic is never a
production nfit dependency or a unit test. All generated scientific products
must be inside an IPTS directory; original raw and saved project files are read.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import h5py
import mantid
import numpy as np
from mantid import config
from mantid.simpleapi import (
    AddTimeSeriesLog,
    CloneWorkspace,
    ConvertToMD,
    ConvertToMDMinMaxGlobal,
    CreateWorkspace,
    CropWorkspace,
    CropWorkspaceForMDNorm,
    DgsReduction,
    FilterByLogValue,
    GetEi,
    LoadEventNexus,
    LoadInstrument,
    MaskBTP,
    MDNorm,
    RotateInstrumentComponent,
    SaveMD,
    SaveNexusProcessed,
    SetGoniometer,
    SetUB,
)


def vector(v):
    return [float(v.X()), float(v.Y()), float(v.Z())]


def geometry(ws):
    instrument = ws.getInstrument()
    info = ws.detectorInfo()
    selected = [i for i in range(info.size()) if not info.isMonitor(i)]
    return {"ids": np.asarray(info.detectorIDs())[selected],
            "positions": np.asarray([vector(info.position(i)) for i in selected]),
            "l1": float(ws.spectrumInfo().l1()),
            "source": vector(instrument.getSource().getPos()),
            "sample": vector(instrument.getSample().getPos()),
            "tank_quaternion": [float(instrument.getComponentByName("Tank").getRotation()[i]) for i in range(4)]}


def rotation_matrix(quaternion):
    w, x, y, z = quaternion
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def geometry_reference(output, runs):
    receipt = {"mantid_version": mantid.__version__, "platform": platform.platform(), "runs": []}
    raw = None
    for number in runs:
        started = time.perf_counter()
        source = Path(f"/SNS/HYS/IPTS-36860/nexus/HYS_{number}.nxs.h5")
        raw = LoadEventNexus(Filename=str(source), MetaDataOnly=True,
                            OutputWorkspace=f"geometry_{number}")
        geom = geometry(raw)
        np.savez(output/f"mantid_geometry_{number}.npz", ids=geom.pop("ids"), positions=geom.pop("positions"))
        record = {"run": number, "source": str(source), **geom, "logs": {},
                  "seconds": time.perf_counter()-started}
        for name in ("msd", "s2", "EnergyRequest", "psda", "psr", "omega"):
            log = raw.run()[name]
            statistics = log.getStatistics()
            record["logs"][name] = {"values": np.asarray(log.value).tolist(),
                "mean": float(statistics.mean), "time_mean": float(statistics.time_mean)}
        receipt["runs"].append(record)
    # Alter only an in-memory workspace; deliberately unequal time intervals
    # distinguish arithmetic mean from the time-weighted run-log statistic.
    probe = CloneWorkspace(raw, OutputWorkspace="geometry_log_probe")
    start = str(raw.run().startTime())[:26]
    epoch = datetime.datetime.fromisoformat(start)
    for offset, value in ((0, 0), (10, 90), (100, 0)):
        stamp = (epoch+datetime.timedelta(seconds=offset)).isoformat()
        AddTimeSeriesLog(Workspace=probe, Name="nfit_probe_angle", Time=stamp, Value=float(value))
    original_idf = Path(config["instrumentDefinition.directory"])/"HYSPEC_Definition.xml"
    text = original_idf.read_text()
    probe_idf = output/"hyspec_log_probe.xml"
    probe_idf.write_text(text.replace('id="s2"', 'id="nfit_probe_angle"'))
    LoadInstrument(Workspace=probe, Filename=str(probe_idf), RewriteSpectraMap=False)
    q = geometry(probe)["tank_quaternion"]
    matrix = rotation_matrix(q)
    yaw = float(np.degrees(np.arctan2(matrix[0, 2], matrix[2, 2])))
    stat = probe.run()["nfit_probe_angle"].getStatistics()
    receipt["log_mean_probe"] = {"times_seconds": [0, 10, 100], "values": [0, 90, 0],
        "arithmetic_mean": float(stat.mean), "time_mean": float(stat.time_mean),
        "mantid_resolved_yaw_degrees": yaw, "quaternion": q}
    # A static 45-degree location rotation plus three geometry parameters tests
    # whether parameterized Euler orientation composes with or replaces it.
    probe_idf.write_text(text.replace('id="s2"', 'id="nfit_probe_angle"').replace(
        '<component type="Tank" idlist="Tank">\n    <location>',
        '<component type="Tank" idlist="Tank">\n    <location rot="45" axis-x="0" axis-y="0" axis-z="1">').replace(
        '<parameter name="rotx">\n        <value val="0"/>',
        '<parameter name="rotx">\n        <value val="20"/>').replace(
        '<parameter name="roty">\n        <logfile id="nfit_probe_angle"  eq="0.0+value"/>',
        '<parameter name="roty">\n        <value val="30"/>').replace(
        '<parameter name="rotz">\n        <value val="0"/>',
        '<parameter name="rotz">\n        <value val="40"/>'))
    LoadInstrument(Workspace=probe, Filename=str(probe_idf), RewriteSpectraMap=False)
    q = geometry(probe)["tank_quaternion"]
    receipt["static_dynamic_rotation_probe"] = {"location_z_degrees": 45,
        "parameter_degrees_xyz": [20, 30, 40], "quaternion": q,
        "matrix": rotation_matrix(q).tolist()}
    old_matrix = rotation_matrix(q)
    RotateInstrumentComponent(Workspace=probe, ComponentName="Tank", X=0, Y=1,
                              Z=0, Angle=17, RelativeRotation=True)
    receipt["relative_rotation_probe"] = {"old_matrix": old_matrix.tolist(),
        "added_global_y_degrees": 17,
        "new_matrix": rotation_matrix(geometry(probe)["tank_quaternion"]).tolist()}
    # Distinguish rounding after equation evaluation from rounding input logs.
    for name, values in (("nfit_probe_scalar", [123.456789]*3),
                         ("nfit_probe_order", [12.3456789, 78.9012345, 23.456789])):
        for offset, value in zip((0, 10, 100), values, strict=True):
            stamp = (epoch+datetime.timedelta(seconds=offset)).isoformat()
            AddTimeSeriesLog(Workspace=probe, Name=name, Time=stamp, Value=float(value))
    cases = [("equation_divide", "nfit_probe_scalar", "mean", "value/3"),
             ("equation_cancel", "nfit_probe_scalar", "mean", "value-123.456")]
    cases += [(mode, "nfit_probe_order", mode, "value") for mode in
              ("first_value", "last_value", "median", "minimum", "maximum", "position 2", "mean")]
    results = []
    for label, log_name, mode, equation in cases:
        replaced = text.replace('id="s2"  eq="0.0+value"',
            f'id="{log_name}" extract-single-value-as="{mode}" eq="{equation}"')
        path = output/f"hyspec_log_probe_{label.replace(' ', '_')}_Definition.xml"
        path.write_text(replaced)
        LoadInstrument(Workspace=probe, Filename=str(path), RewriteSpectraMap=False)
        q = geometry(probe)["tank_quaternion"]
        matrix = rotation_matrix(q)
        results.append({"label": label, "log": log_name, "mode": mode, "equation": equation,
            "resolved_yaw_degrees": float(np.degrees(np.arctan2(matrix[0, 2], matrix[2, 2])))})
    receipt["log_extraction_probes"] = results
    receipt["probe_run_start"] = str(raw.run().startTime())
    receipt["probe_run_end"] = str(raw.run().endTime())
    # Mantid's automatic VTK shape-cache files are geometry descriptions; move
    # the files created by these probes beside their IPTS instrument definitions.
    cache_output = output/"geometry_cache"
    cache_output.mkdir(exist_ok=True)
    for path in (Path.home()/".mantid/instrument/geometryCache").glob("hyspec_log_probe*.vtp"):
        shutil.move(str(path), str(cache_output/path.name))
    receipt["idf_sha256"] = hashlib.sha256(original_idf.read_bytes()).hexdigest()
    receipt["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/"mantid_geometry_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def reduction_reference(output, runs):
    """Execute the audited Shiver default MDE chain in Mantid, without importing
    Shiver or its configuration singleton (which can rewrite ~/.shiver settings).
    """
    merged = Path("/SNS/HYS/IPTS-36860/shared/Ei15meV_50K_240Hz_s2_70.nxs")
    with h5py.File(merged, "r") as handle:
        ub = np.asarray(handle["MDEventWorkspace/experiment0/sample/oriented_lattice/orientation_matrix"])
    recipe = dict(Ei=15., Emin=-14.25, Emax=14.25, TIB=False, BadPulsesThreshold=0.,
                  normalization="None", distribution=False, grouping=None,
                  mask="MaskBTP Pixel=1-8,121-128", goniometer="Universal", QFrame="Q_sample")
    receipt = {"mantid_version": mantid.__version__, "recipe": recipe, "UB": ub.tolist(),
               "runs": [], "source": "audited installed Shiver ConvertDGSToSingleMDE default chain"}
    chain = Path("/usr/local/pixi/shiver/.pixi/envs/default/lib/python3.12/site-packages/shiver/models/convert_dgs_to_single_mde.py")
    receipt["shiver_chain_sha256"] = hashlib.sha256(chain.read_bytes()).hexdigest()
    for number in runs:
        started = time.perf_counter()
        source = Path(f"/SNS/HYS/IPTS-36860/nexus/HYS_{number}.nxs.h5")
        raw = LoadEventNexus(Filename=str(source), OutputWorkspace=f"raw_{number}")
        raw_events = raw.getNumberEvents()
        if raw.run().hasProperty("pause"):
            raw = FilterByLogValue(InputWorkspace=raw, LogName="pause", MinimumValue=-1,
                                   MaximumValue=.5, LogBoundary="Left", OutputWorkspace=f"raw_{number}")
        pause_events = raw.getNumberEvents()
        MaskBTP(Workspace=raw, Pixel="1-8,121-128")
        # Persist an explicit matched mask for nfit; no implicit instrument mask
        # or scientific default is introduced by this diagnostic.
        valid = [0. if raw.spectrumInfo().isMasked(i) else 1. for i in range(raw.getNumberHistograms())]
        mask = CreateWorkspace(DataX=[0, 1], DataY=valid, NSpec=len(valid), ParentWorkspace=raw,
                               OutputWorkspace=f"mask_{number}")
        SaveNexusProcessed(InputWorkspace=mask, Filename=str(output/f"hyspec_tip_mask_{number}.nxs"))
        SetGoniometer(Workspace=raw, Goniometers="Universal")
        t0 = float(GetEi(InputWorkspace=raw).Tzero)
        ei = recipe["Ei"]
        msd = float(raw.run()["msd"].getStatistics().mean)
        tel = (39000+msd+4500)*1000/np.sqrt(ei/5.227e-6)
        tof_bounds = [float(tel-1e6/120-470), float(tel+1e6/120+470)]
        raw = CropWorkspace(InputWorkspace=raw, XMin=tof_bounds[0], XMax=tof_bounds[1],
                            OutputWorkspace=f"raw_{number}")
        crop_events = raw.getNumberEvents()
        # Representative unpolarized runs must not silently bypass this branch.
        assert float(raw.run()["psda"].getStatistics().mean) == 0
        dgs, _ = DgsReduction(SampleInputWorkspace=raw, SampleInputMonitorWorkspace=raw,
            TimeZeroGuess=t0, IncidentEnergyGuess=ei, UseIncidentEnergyGuess=True,
            IncidentBeamNormalisation="None", EnergyTransferRange="-14.25,28.5,14.25",
            TimeIndepBackgroundSub=False, SofPhiEIsDistribution=False,
            OutputWorkspace=f"dgs_{number}")
        dgs = CropWorkspaceForMDNorm(InputWorkspace=dgs, XMin=-14.25, XMax=14.25,
                                     OutputWorkspace=f"dgs_{number}")
        limits = ConvertToMDMinMaxGlobal(InputWorkspace=dgs, QDimensions="Q3D",
                                        dEAnalysisMode="Direct", Q3DFrames="Q")
        md = ConvertToMD(InputWorkspace=dgs, QDimensions="Q3D", dEAnalysisMode="Direct",
            Q3DFrames="Q_sample", MinValues=limits.MinValues, MaxValues=limits.MaxValues,
            PreprocDetectorsWS="-", MaxRecursionDepth=2, OutputWorkspace=f"md_{number}")
        SetUB(Workspace=md, UB=ub)
        SaveMD(InputWorkspace=md, Filename=str(output/f"mantid_mde_{number}.nxs"))
        reduction_seconds = time.perf_counter()-started
        # Independent diagnostic: retain requested physical energy support
        # instead of observed event extrema, to identify conversion-bound losses.
        widened_min = np.asarray(limits.MinValues).copy()
        widened_max = np.asarray(limits.MaxValues).copy()
        widened_min[3], widened_max[3] = -14.25, 14.25
        wide = ConvertToMD(InputWorkspace=dgs, QDimensions="Q3D", dEAnalysisMode="Direct",
            Q3DFrames="Q_sample", MinValues=widened_min, MaxValues=widened_max,
            PreprocDetectorsWS="-", MaxRecursionDepth=2, OutputWorkspace=f"wide_{number}")
        SetUB(Workspace=wide, UB=ub)
        SaveMD(InputWorkspace=wide, Filename=str(output/f"mantid_mde_wide_{number}.nxs"))
        started = time.perf_counter()
        MDNorm(InputWorkspace=md, QDimension0="1,1,0", QDimension1="0,0,1", QDimension2="1,-1,0",
            Dimension0Name="QDimension0", Dimension0Binning="-8,0.2,8",
            Dimension1Name="QDimension1", Dimension1Binning="-8,0.2,8",
            Dimension2Name="QDimension2", Dimension2Binning="-0.12,0.12",
            Dimension3Name="DeltaE", Dimension3Binning="-14.25,0.5,14.25",
            OutputWorkspace=f"hist_{number}", OutputDataWorkspace=f"hist_data_{number}",
            OutputNormalizationWorkspace=f"hist_norm_{number}")
        from mantid.simpleapi import mtd
        data, norm = mtd[f"hist_data_{number}"], mtd[f"hist_norm_{number}"]
        with h5py.File(output/f"mantid_hist_{number}.h5", "w") as handle:
            handle.create_dataset("numerator", data=data.getSignalArray())
            handle.create_dataset("variance", data=data.getErrorSquaredArray())
            handle.create_dataset("counts", data=data.getNumEventsArray())
            handle.create_dataset("norm", data=norm.getSignalArray())
            handle.attrs["actual_dimensions"] = json.dumps([[data.getDimension(i).getMinimum(),data.getDimension(i).getMaximum(),data.getDimension(i).getNBins()] for i in range(4)])
            handle.attrs["binning"] = json.dumps(dict(lower=[-8,-8,-.12,-14.25],
                upper=[8,8,.12,14.25], num_bins=[80,80,1,57],
                vectors=[[1,1,0,0],[0,0,1,0],[1,-1,0,0],[0,0,0,1]]))
        original_mdnorm_seconds = time.perf_counter()-started
        MDNorm(InputWorkspace=wide, QDimension0="1,1,0", QDimension1="0,0,1", QDimension2="1,-1,0",
            Dimension0Name="QDimension0", Dimension0Binning="-8,0.2,8",
            Dimension1Name="QDimension1", Dimension1Binning="-8,0.2,8",
            Dimension2Name="QDimension2", Dimension2Binning="-0.12,0.12",
            Dimension3Name="DeltaE", Dimension3Binning="-14.25,0.5,14.25",
            OutputWorkspace=f"wide_hist_{number}", OutputDataWorkspace=f"wide_hist_data_{number}",
            OutputNormalizationWorkspace=f"wide_hist_norm_{number}")
        from mantid.simpleapi import mtd
        data, norm = mtd[f"wide_hist_data_{number}"], mtd[f"wide_hist_norm_{number}"]
        with h5py.File(output/f"mantid_hist_wide_{number}.h5", "w") as handle:
            handle.create_dataset("numerator", data=data.getSignalArray())
            handle.create_dataset("variance", data=data.getErrorSquaredArray())
            handle.create_dataset("counts", data=data.getNumEventsArray())
            handle.create_dataset("norm", data=norm.getSignalArray())
            handle.attrs["actual_dimensions"] = json.dumps([[data.getDimension(i).getMinimum(),data.getDimension(i).getMaximum(),data.getDimension(i).getNBins()] for i in range(4)])
            handle.attrs["binning"] = json.dumps(dict(lower=[-8,-8,-.12,-14.25],
                upper=[8,8,.12,14.25], num_bins=[80,80,1,57],
                vectors=[[1,1,0,0],[0,0,1,0],[1,-1,0,0],[0,0,0,1]]))
        receipt["runs"].append({"run": number, "raw_events": raw_events, "pause_events": pause_events,
            "tof_crop_events": crop_events, "mde_events": md.getNEvents(), "mde_wide_events": wide.getNEvents(),
            "convert_to_md_min_values": np.asarray(limits.MinValues).tolist(), "convert_to_md_max_values": np.asarray(limits.MaxValues).tolist(), "Ei_meV": ei, "T0_us": t0,
            "tof_crop_bounds_us": tof_bounds, "proton_charge_uah": float(raw.run().getProtonCharge()),
            "reduction_seconds": reduction_seconds, "mdnorm_seconds": original_mdnorm_seconds,
            "histogram_shape": list(data.getSignalArray().shape),
            "histogram_numerator_sum": float(np.sum(data.getSignalArray())),
            "histogram_variance_sum": float(np.sum(data.getErrorSquaredArray())),
            "histogram_norm_sum": float(np.sum(norm.getSignalArray()))})
        print("MANTID_REFERENCE_RUN_COMPLETE", number, flush=True)
    receipt["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/"mantid_reduction_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)

def boundary_reference(output, run):
    raw = LoadEventNexus(Filename=f"/SNS/HYS/IPTS-36860/nexus/HYS_{run}.nxs.h5", OutputWorkspace="boundary_raw")
    tofs = np.concatenate([raw.getSpectrum(i).getTofs() for i in range(raw.getNumberHistograms())])
    unique = np.unique(tofs)
    low, high = unique[len(unique)//3], unique[2*len(unique)//3]
    cropped = CropWorkspace(InputWorkspace=raw, XMin=float(low), XMax=float(high), OutputWorkspace="boundary_crop")
    actual = cropped.getNumberEvents()
    hypotheses = {"closed_both": int(np.sum((tofs >= low) & (tofs <= high))),
                  "closed_lower_open_upper": int(np.sum((tofs >= low) & (tofs < high))),
                  "open_lower_closed_upper": int(np.sum((tofs > low) & (tofs <= high))),
                  "open_both": int(np.sum((tofs > low) & (tofs < high)))}
    receipt = {"run": run, "mantid_version": mantid.__version__, "tof_bounds_us": [float(low), float(high)],
               "events_at_lower": int(np.sum(tofs == low)), "events_at_upper": int(np.sum(tofs == high)),
               "actual_retained_events": actual, "hypotheses": hypotheses}
    (output/"mantid_boundary_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, nargs="+", default=[506277, 506278])
    parser.add_argument("--mode", choices=["geometry", "reduction", "boundary"], default="geometry")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if "/IPTS-" not in str(output):
        raise ValueError("Manual ORNL scientific products must be in an IPTS directory")
    output.mkdir(parents=True, exist_ok=True)
    config["defaultsave.directory"] = str(output)
    config["datasearch.directories"] = str(output)
    config["UpdateInstrumentDefinitions.OnStartup"] = "0"
    os.chdir(output)
    if arguments.mode == "geometry":
        geometry_reference(output, arguments.runs)
    elif arguments.mode == "reduction":
        reduction_reference(output, arguments.runs)
    else:
        boundary_reference(output, arguments.runs[0])


if __name__ == "__main__":
    main()
