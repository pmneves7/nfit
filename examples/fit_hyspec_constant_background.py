from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metallix import (
    attach_fit_comparisons,
    DataGroup,
    DatasetEntry,
    FitModelSession,
    ParameterSpec,
    hyspec_hhl_fit_comparison_from_points,
    load_mantid_mdhisto_nxs,
    make_constant_intensity_model,
    make_energy_q_mask_transform,
    make_phonon_mask_transform,
    point_data_from_hyspec_hhl,
    polynomial_fwhm_energy_resolution,
    slice_viewer,
)


FILENAME = "4D_test.nxs"
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
    parser.add_argument(
        "--view-fit",
        action="store_true",
        help="Open the slice viewer with masked data, fit, and residual datasets.",
    )
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
    data_group = DataGroup(
        name="hyspec_3meV_1p8K_group",
        datasets=[
            DatasetEntry(
                name="hyspec_3meV_1p8K",
                data=point_data,
                kind="inelastic_neutron_point_data",
                parameters={"temperature": 1.8},
                transforms=transforms,
            )
        ],
        metadata={"source_file": str(path)},
    )
    prepared = data_group.get_dataset("hyspec_3meV_1p8K").prepared().valid()

    initial_constant = float(np.nanmedian(prepared.intensity))
    model_session = FitModelSession(
        name="constant_background",
        model=make_constant_intensity_model("constant"),
        parameter_specs=[ParameterSpec("constant", initial_constant, description="flat background")],
        resolution_by_dataset={
            "hyspec_3meV_1p8K": polynomial_fwhm_energy_resolution(
                [0.6, 0.0, 0.0],
                oversampling=args.resolution_oversampling,
            )
        },
    )
    data_group.add_model(model_session)
    result = model_session.fit(data_group)

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
            "history_length": len(model_session.history),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.view_fit:
        fit_values = result.dataset_model_values["hyspec_3meV_1p8K"]
        attach_fit_comparisons(
            mdhisto,
            [
                hyspec_hhl_fit_comparison_from_points(
                    mdhisto,
                    prepared,
                    fit_values,
                    model_name=model_session.name,
                    result_name=f"fit {len(model_session.history) - 1}",
                )
            ],
        )
        viewer = slice_viewer(
            mdhisto,
            x_dim="[H,H,0]",
            y_dim="[0,0,L]",
            channel="signal",
            cmap="viridis",
            color_scale="linear",
            auto_limits="min/max",
        )
        viewer.run()


if __name__ == "__main__":
    main()
