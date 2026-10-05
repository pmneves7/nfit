"""Run bounded native raw-pipeline trials sequentially on the ORNL runtime.

Usage: python run_dgs_raw_trials.py CONFIG [--rounds 2] [--modes baseline,coalesced]
The configuration, scripts, logs, staging and outputs must be below its IPTS
output root. Neither this scheduler nor the native trials imports Mantid.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from benchmark_dgs_settings import RAW_PIPELINE_TRIAL_MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--modes", default="baseline,coalesced,prefetch,threads2,threads4,threads2-coalesced,threads4-coalesced")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    settings = json.loads(args.config.read_text())
    root = Path(settings["output_root"]).resolve()
    if not any(part.startswith("IPTS-") for part in root.parts):
        raise ValueError("Scientific outputs require an IPTS directory")
    if not args.config.resolve().is_relative_to(root):
        raise ValueError("Store the configuration with its scientific outputs")
    modes = args.modes.split(",")
    if len(set(modes)) != len(modes):
        raise ValueError("Repeated trial modes would overwrite a result")
    if any(mode not in RAW_PIPELINE_TRIAL_MODES for mode in modes):
        raise ValueError("Unknown trial mode")
    scripts = Path(__file__).resolve().parent
    records = []
    for repeat in range(args.rounds):
        order = modes if repeat % 2 == 0 else list(reversed(modes))
        for mode in order:
            tag = f"{settings['preset']}-{mode}-r{repeat}"
            if (root / "nfit" / tag).exists():
                raise FileExistsError(f"Refusing to overwrite {tag}")
            temporary = root / "runtime" / tag
            temporary.mkdir(parents=True, exist_ok=False)
            logs = root / "logs"
            logs.mkdir(exist_ok=True)
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", TMPDIR=str(temporary),
                               MPLCONFIGDIR=str(temporary / "matplotlib"),
                               XDG_CACHE_HOME=str(temporary / "cache"),
                               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               OMP_NUM_THREADS=str(settings.get("threads", 8)),
                               NFIT_DGS_TRIAL_MODE=mode,
                               NFIT_DGS_BENCHMARK_ARGS=json.dumps(["--config", str(args.config), "--tag", tag]))
            # Do not inherit source injection from another benchmark launch.
            environment.pop("NFIT_DGS_BENCHMARK_SOURCE", None)
            environment.pop("NFIT_DGS_TRIAL_PREPRODUCE", None)
            command = ["/usr/bin/time", "-v", "-o", str(logs / f"{tag}-resources.txt"),
                       "/SNS/users/paulneves/bin/nfit", "--run-script", str(scripts / "trial_dgs_raw_pipeline.py")]
            started = time.time()
            with (logs / f"{tag}.log").open("x") as stream:
                result = subprocess.run(command, env=environment, cwd=root,
                                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            record = dict(mode=mode, repeat=repeat, tag=tag, returncode=result.returncode,
                          process_wall_seconds=time.time()-started)
            receipt = root / "nfit" / tag / "receipt.json"
            if receipt.exists():
                data = json.loads(receipt.read_text())
                record.update(status=data.get("status"), initial_seconds=data.get("initial_dataset_workflow_seconds"),
                              cached_seconds=data.get("subsequent_rebin_and_save_seconds"),
                              stages=data.get("stages"), nfit_version=data.get("nfit_version"))
            records.append(record)
            (root / f"{settings['preset']}-trial-summary.json").write_text(json.dumps(records, indent=2)+"\n")
            print(json.dumps(record), flush=True)
            if result.returncode:
                raise RuntimeError(f"Trial failed: {tag}; inspect its log")


if __name__ == "__main__":
    main()
