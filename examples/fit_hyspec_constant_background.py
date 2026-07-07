from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metallix import (
    FitDataset,
    FitProblem,
    ParameterSpec,
    fit_problem_least_squares,
    load_mantid_mdhisto_nxs,
    make_constant_intensity_model,
    make_energy_q_mask_transform,
    make_phonon_mask_transform,
    point_data_from_hyspec_hhl,
    polynomial_fwhm_energy_resolution,
)


FILENAME = "3D_HHL_3meV_1p8K_metallix_m-3m.nxs"
DATA_PATH = Path("data/hyspec") / FILENAME


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the HYSPEC example data to a constant background with energy resolution."
    )
    parser.add_argument("--path", type=Path, default=DATA_PATH, help="NeXus path to fit.")
    parser.add_argument(
        "--max-q-trajectories",
        type=int,
        default=5000,
        help="Maximum complete Q trajectories to fit; use 0 for all trajectories.",
    )
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument(
        "--phonon-velocity",
        type=float,
        default=35.0,
        help="Phonon cone velocity dE/d|Q| in meV per inverse angstrom.",
    )
    parser.add_argument("--resolution-oversampling", type=int, default=5)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    path = args.path
    max_q_trajectories = None if args.max_q_trajectories == 0 else args.max_q_trajectories

    mdhisto = load_mantid_mdhisto_nxs(path, copy_metadata=False)
    point_data, data_summary = point_data_from_hyspec_hhl(
        mdhisto,
        max_q_trajectories=max_q_trajectories,
        random_seed=args.random_seed,
        temperature=1.8,
    )

    transforms = [
        make_energy_q_mask_transform(energy=(-2.0, 1.0)),
        make_phonon_mask_transform(
            center=[2.0, 2.0, 2.0],
            slope=args.phonon_velocity,
            center_units="rlu",
        ),
    ]
    dataset = FitDataset(
        name="hyspec_3meV_1p8K",
        data=point_data,
        transforms=transforms,
        resolution=polynomial_fwhm_energy_resolution(
            [0.6, 0.0, 0.0],
            oversampling=args.resolution_oversampling,
        ),
    )
    prepared = dataset.prepared().valid()

    initial_constant = float(np.nanmedian(prepared.intensity))
    problem = FitProblem(
        datasets=[dataset],
        model=make_constant_intensity_model("constant"),
        parameter_specs=[ParameterSpec("constant", initial_constant, description="flat background")],
        description="HYSPEC constant background example",
    )
    result = fit_problem_least_squares(problem)

    stderr = None if result.stderr is None else result.stderr.get("constant")
    summary = {
        "source_file": str(path),
        "model": "constant measured intensity",
        "resolution": {
            "type": "polynomial Gaussian FWHM",
            "coefficients_meV": [0.6, 0.0, 0.0],
            "oversampling": args.resolution_oversampling,
        },
        "masks": {
            "elastic_energy_meV": [-2.0, 1.0],
            "phonon_center_rlu": [2.0, 2.0, 2.0],
            "phonon_velocity_meV_per_inv_angstrom": args.phonon_velocity,
        },
        "data": {
            **data_summary,
            "points_after_masks": prepared.size,
            "used_subset": max_q_trajectories is not None,
        },
        "fit": {
            "success": result.success,
            "message": result.message,
            "constant": result.params["constant"],
            "constant_stderr": stderr,
            "chi2": result.chi2,
            "reduced_chi2": result.reduced_chi2,
            "dof": prepared.size - len(result.variable_names),
            "cost": result.cost,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
