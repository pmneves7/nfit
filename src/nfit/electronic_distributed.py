"""Scheduler-neutral response-point chunks and Slurm script rendering."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .electronic_response import SusceptibilityResult

FloatArray = NDArray[np.float64]


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ResponsePointChunk:
    """One deterministic contiguous subset of paired response points."""

    index: int
    start: int
    stop: int
    q_reduced: FloatArray
    energy_meV: FloatArray

    def __post_init__(self) -> None:
        q = _readonly(self.q_reduced)
        energy = _readonly(self.energy_meV)
        if (
            int(self.index) < 0
            or int(self.start) < 0
            or int(self.stop) <= int(self.start)
        ):
            raise ValueError("response chunk indices must define a positive range")
        if q.shape != (self.stop - self.start, 3):
            raise ValueError("chunk wavevectors must match its global point range")
        if energy.shape != (self.stop - self.start,):
            raise ValueError("chunk energies must match its global point range")
        object.__setattr__(self, "q_reduced", q)
        object.__setattr__(self, "energy_meV", energy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "start": int(self.start),
            "stop": int(self.stop),
            "q_reduced": self.q_reduced.tolist(),
            "energy_meV": self.energy_meV.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResponsePointChunk:
        return cls(
            index=int(payload["index"]),
            start=int(payload["start"]),
            stop=int(payload["stop"]),
            q_reduced=payload["q_reduced"],
            energy_meV=payload["energy_meV"],
        )


@dataclass(frozen=True)
class ResponseChunkResult:
    """A calculated chunk paired with its global point range."""

    chunk: ResponsePointChunk
    response: SusceptibilityResult

    def __post_init__(self) -> None:
        if self.response.q_reduced.shape[0] != self.chunk.stop - self.chunk.start:
            raise ValueError("chunk response point count does not match its range")


def partition_response_points(
    q_reduced: ArrayLike,
    energy_meV: ArrayLike,
    *,
    points_per_chunk: int,
) -> tuple[ResponsePointChunk, ...]:
    """Partition paired response points into reproducible contiguous chunks."""

    q = np.asarray(q_reduced, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    energy = np.asarray(energy_meV, dtype=float)
    if energy.ndim == 0:
        energy = np.full(q.shape[0], float(energy))
    size = int(points_per_chunk)
    if size < 1:
        raise ValueError("points_per_chunk must be positive")
    if q.ndim != 2 or q.shape[1] != 3 or energy.shape != (q.shape[0],):
        raise ValueError("q_reduced and energy_meV must define paired points")
    return tuple(
        ResponsePointChunk(
            index=index,
            start=start,
            stop=min(start + size, q.shape[0]),
            q_reduced=q[start : start + size],
            energy_meV=energy[start : start + size],
        )
        for index, start in enumerate(range(0, q.shape[0], size))
    )


def merge_response_chunks(
    chunks: list[ResponseChunkResult] | tuple[ResponseChunkResult, ...],
) -> SusceptibilityResult:
    """Merge complete, nonoverlapping chunk results in global point order."""

    ordered = sorted(chunks, key=lambda item: item.chunk.start)
    if not ordered or ordered[0].chunk.start != 0:
        raise ValueError("response chunks must start at global point zero")
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous.chunk.stop != current.chunk.start:
            raise ValueError("response chunks must be contiguous and nonoverlapping")
    reference = ordered[0].response
    for item in ordered:
        response = item.response
        if (
            response.operator_labels != reference.operator_labels
            or response.conjugate_indices != reference.conjugate_indices
            or response.model_digest != reference.model_digest
            or response.temperature_K != reference.temperature_K
            or response.chemical_potential_meV != reference.chemical_potential_meV
            or response.broadening_meV != reference.broadening_meV
            or response.response_kind != reference.response_kind
            or response.normalization != reference.normalization
        ):
            raise ValueError("response chunks use incompatible scientific settings")
    return SusceptibilityResult(
        q_reduced=np.concatenate(
            [item.response.q_reduced for item in ordered],
            axis=0,
        ),
        Q_reduced=np.concatenate(
            [item.response.Q_reduced for item in ordered],
            axis=0,
        ),
        energy_meV=np.concatenate(
            [item.response.energy_meV for item in ordered],
            axis=0,
        ),
        values_per_meV_cell=np.concatenate(
            [item.response.values_per_meV_cell for item in ordered],
            axis=0,
        ),
        operator_labels=reference.operator_labels,
        conjugate_indices=reference.conjugate_indices,
        model_digest=reference.model_digest,
        temperature_K=reference.temperature_K,
        chemical_potential_meV=reference.chemical_potential_meV,
        broadening_meV=reference.broadening_meV,
        response_kind=reference.response_kind,
        normalization=reference.normalization,
        provenance={
            **dict(reference.provenance),
            "distributed_chunks": [
                {
                    "index": item.chunk.index,
                    "start": item.chunk.start,
                    "stop": item.chunk.stop,
                }
                for item in ordered
            ],
            "merge_order": "global contiguous point index",
        },
    )


def response_slurm_array_script(
    worker_script: str,
    *,
    chunk_count: int,
    job_name: str = "nfit-response",
    python_executable: str = "python",
    cpus_per_task: int = 1,
    memory_gb: float = 4.0,
    walltime: str = "01:00:00",
) -> str:
    """Render an editable Slurm array launcher for a chunk-aware worker."""

    count = int(chunk_count)
    cpus = int(cpus_per_task)
    memory = float(memory_gb)
    if count < 1:
        raise ValueError("chunk_count must be positive")
    if cpus < 1:
        raise ValueError("cpus_per_task must be positive")
    if not np.isfinite(memory) or memory <= 0.0:
        raise ValueError("memory_gb must be finite and positive")
    if not str(job_name).strip() or not str(walltime).strip():
        raise ValueError("job_name and walltime cannot be empty")
    command = (
        f"{shlex.quote(str(python_executable))} "
        f"{shlex.quote(str(worker_script))} "
        '--chunk-index "${SLURM_ARRAY_TASK_ID}"'
    )
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --array=0-{count - 1}",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={memory:g}G",
            f"#SBATCH --time={walltime}",
            "set -euo pipefail",
            "",
            command,
            "",
        ]
    )
