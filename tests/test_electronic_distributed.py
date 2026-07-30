from __future__ import annotations

import numpy as np
import pytest

from nfit import (
    BasisState,
    ResponseChunkResult,
    ResponsePointChunk,
    bare_spin_susceptibility,
    build_electronic_model,
    k_mesh,
    merge_response_chunks,
    partition_response_points,
    response_slurm_array_script,
)


def _model():
    return build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=[BasisState("s", site="A", orbital="s")],
        hoppings={(1, 0, 0): [[-10.0]]},
        orbital_centers=[[0.0, 0.0, 0.0]],
        periodic_axes=(0,),
        energy_unit="meV",
    )


def test_response_point_chunks_round_trip_and_merge_reference_response():
    model = _model()
    mesh = k_mesh(model, (32,))
    Q = np.column_stack(
        (
            np.full(5, 0.35),
            np.zeros(5),
            np.zeros(5),
        )
    )
    energy = np.linspace(-2.0, 2.0, 5)
    settings = {
        "temperature_K": 20.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 0.4,
    }
    reference = bare_spin_susceptibility(model, Q, energy, mesh, **settings)
    chunks = partition_response_points(Q, energy, points_per_chunk=2)
    restored = ResponsePointChunk.from_dict(chunks[0].to_dict())
    np.testing.assert_array_equal(restored.q_reduced, chunks[0].q_reduced)
    results = [
        ResponseChunkResult(
            chunk=chunk,
            response=bare_spin_susceptibility(
                model,
                chunk.q_reduced,
                chunk.energy_meV,
                mesh,
                **settings,
            ),
        )
        for chunk in chunks
    ]
    merged = merge_response_chunks(results)

    np.testing.assert_allclose(
        merged.values_per_meV_cell,
        reference.values_per_meV_cell,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert merged.provenance["merge_order"] == "global contiguous point index"

    with pytest.raises(ValueError, match="contiguous"):
        merge_response_chunks([results[0], results[-1]])


def test_slurm_array_script_is_scheduler_wrapper_not_scientific_state():
    script = response_slurm_array_script(
        "response worker.py",
        chunk_count=4,
        job_name="scan",
        python_executable="/env/bin/python",
        cpus_per_task=8,
        memory_gb=32,
        walltime="02:30:00",
    )

    assert "#SBATCH --array=0-3" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=32G" in script
    assert "'response worker.py'" in script
    assert '"${SLURM_ARRAY_TASK_ID}"' in script

    with pytest.raises(ValueError, match="chunk_count"):
        response_slurm_array_script("worker.py", chunk_count=0)
