from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metallix import PointData4D, ParameterSpec, fit_least_squares
from metallix.cross_section import intensity_from_chipp
from metallix.models import paramagnon_chipp


def measured_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
    chipp = paramagnon_chipp(
        data.H,
        data.K,
        data.L,
        data.E,
        amplitude=params["amplitude"],
        q0=(params["q0_h"], params["q0_k"], params["q0_l"]),
        kappa=params["kappa"],
        omega_sf=params["omega_sf"],
    )
    return intensity_from_chipp(
        chipp,
        data.E,
        params["temperature"],
        scale=params["scale"],
        background=params["background"],
    )


def make_dataset(seed: int = 7) -> PointData4D:
    rng = np.random.default_rng(seed)
    H = np.linspace(0.25, 0.75, 30)
    E = np.linspace(1.0, 12.0, 25)
    HH, EE = np.meshgrid(H, E, indexing="ij")
    H_flat = HH.ravel()
    E_flat = EE.ravel()
    K_flat = np.full_like(H_flat, 0.5)
    L_flat = np.zeros_like(H_flat)

    true = {
        "amplitude": 5.0,
        "q0_h": 0.5,
        "q0_k": 0.5,
        "q0_l": 0.0,
        "kappa": 0.18,
        "omega_sf": 4.0,
        "scale": 2.0,
        "background": 0.15,
        "temperature": 50.0,
    }
    clean = measured_model(
        PointData4D(H_flat, K_flat, L_flat, E_flat, np.zeros_like(H_flat), np.ones_like(H_flat)),
        true,
    )
    sigma = np.full_like(clean, 0.05)
    intensity = clean + rng.normal(0.0, sigma)
    return PointData4D(
        H_flat,
        K_flat,
        L_flat,
        E_flat,
        intensity,
        sigma,
        temperature=true["temperature"],
        metadata={
            "example": "synthetic_single_q_paramagnon",
            "energy_units": "meV",
            "coordinate_units": "RLU",
            "temperature_units": "K",
        },
    )


def main() -> None:
    data = make_dataset()
    specs = [
        ParameterSpec("amplitude", 4.0, min=0.0, unit="arb."),
        ParameterSpec("q0_h", 0.48, min=0.0, max=1.0, unit="RLU"),
        ParameterSpec("q0_k", 0.5, vary=False, unit="RLU"),
        ParameterSpec("q0_l", 0.0, vary=False, unit="RLU"),
        ParameterSpec("kappa", 0.22, min=0.02, max=1.0, unit="RLU"),
        ParameterSpec("omega_sf", 3.0, min=0.2, max=20.0, unit="meV"),
        ParameterSpec("scale", 2.0, vary=False),
        ParameterSpec("background", 0.1),
        ParameterSpec("temperature", 50.0, vary=False, unit="K"),
    ]
    result = fit_least_squares(data, measured_model, specs)
    summary = {
        "success": result.success,
        "reduced_chi2": result.reduced_chi2,
        "params": result.params,
        "stderr": result.stderr,
    }
    out = Path("examples/reference_cases/single_q_paramagnon/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
