"""Launch one detached manual workflow with process-level wall/RSS evidence.

Code may live in /tmp; configurations, logs, temporary scientific storage and
outputs must live under the selected experiment's shared/nfit benchmark folder.
The caller coordinates engines sequentially; this helper never schedules a
second job automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("nfit", "mantid"))
    parser.add_argument("config", type=Path)
    parser.add_argument("tag")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    settings = json.loads(config.read_text())
    root = Path(settings["output_root"]).resolve()
    if not any(re.fullmatch(r"IPTS-\d+", part) for part in root.parts):
        raise ValueError("Benchmark storage must be inside an IPTS folder")
    if "shared" not in root.parts or "nfit" not in root.parts:
        raise ValueError("Select an experiment shared/nfit benchmark folder")
    if not config.is_relative_to(root):
        raise ValueError("Store the scientific configuration with the benchmark outputs")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("Use a simple job tag")
    output = root / args.engine / args.tag
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for directory in (root / "logs", root / "runtime" / f"{args.engine}-{args.tag}"):
        directory.mkdir(parents=True, exist_ok=True)
    temporary = root / "runtime" / f"{args.engine}-{args.tag}"
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", TMPDIR=str(temporary),
                       MPLCONFIGDIR=str(temporary / "matplotlib"),
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OMP_NUM_THREADS=str(settings.get("threads", 64)))
    script = Path(__file__).resolve().with_name(f"benchmark_dgs_{args.engine}_workflow.py")
    arguments = ["--config", str(config), "--tag", args.tag]
    if args.profile:
        arguments.append("--profile")
    if args.engine == "nfit":
        environment["NFIT_DGS_BENCHMARK_ARGS"] = json.dumps(arguments)
        command = ["/SNS/users/paulneves/bin/nfit", "--run-script", str(script)]
    else:
        command = ["/usr/local/pixi/shiver/.pixi/envs/default/bin/python", str(script), *arguments]
    log = root / "logs" / f"{args.engine}-{args.tag}.log"
    resource = root / "logs" / f"{args.engine}-{args.tag}-resources.txt"
    with log.open("x") as handle:
        process = subprocess.Popen(["/usr/bin/time", "-v", "-o", str(resource), *command],
                                   env=environment, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(pid=process.pid, engine=args.engine, tag=args.tag, config=str(config),
                  command=command, logfile=str(log), resources=str(resource),
                  expected_output=str(output), threads=settings.get("threads", 64))
    (root / "logs" / f"{args.engine}-{args.tag}-launch.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
