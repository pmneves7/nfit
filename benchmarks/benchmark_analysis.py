"""Descriptive Data Playground numerical benchmarks (no timing assertions)."""

import runpy
from pathlib import Path
from time import perf_counter

_EXAMPLE = runpy.run_path(Path(__file__).parents[1] / "examples" / "data_playground.py")
bragg_example = _EXAMPLE["bragg_example"]
qfi_example = _EXAMPLE["qfi_example"]


def main() -> None:
    for label, operation in (("Bragg Gaussian", bragg_example), ("QFI curve", qfi_example)):
        started = perf_counter()
        operation()
        print(f"{label}: {perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
