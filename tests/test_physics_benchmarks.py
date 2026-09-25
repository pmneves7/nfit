"""Analytical physics benchmarks independent of the response implementations."""

import numpy as np
import pytest
from scipy.integrate import quad

from nfit import (
    BasisState,
    bare_spin_susceptibility,
    build_electronic_model,
    isotropic_spin_component,
    k_mesh,
)
from nfit.cross_section import KB_MEV_PER_K
from nfit.spin_fluctuations import (
    build_rpa_geometry,
    heisenberg_rpa_susceptibility,
)


def _chain_geometry(hkl):
    hkl = np.asarray(hkl, dtype=float)
    zeros = np.zeros_like(hkl)
    return build_rpa_geometry(
        hkl,
        zeros,
        zeros,
        [[0.0, 0.0, 0.0]],
        [
            {
                "label": "J1",
                "bonds": [
                    {"site_i": 0, "site_j": 0, "offset": [1, 0, 0]},
                ],
            }
        ],
    )


def test_scalar_spin_curie_weiss_chain_and_antiferromagnetic_peak():
    """The high-temperature chain response has theta=z J S(S+1)/3kB."""
    spin = 1.5
    exchange_mev = -0.8  # Negative exchange is antiferromagnetic here.
    coordination = 2
    curie_numerator = spin * (spin + 1.0) / 3.0
    theta_mev = coordination * exchange_mev * curie_numerator
    temperatures_mev = np.array([30.0, 45.0, 75.0])
    q = np.array([0.0, 0.5])
    geometry = _chain_geometry(q)

    uniform = []
    staggered = []
    for temperature in temperatures_mev:
        response = heisenberg_rpa_susceptibility(
            geometry,
            np.zeros(q.shape),
            chi0=curie_numerator / temperature,
            gamma0=3.0,
            j_values={"J1": exchange_mev},
        ).real
        uniform.append(response[0])
        staggered.append(response[1])

    # Curie-Weiss law with theta < 0 for AF exchange.
    np.testing.assert_allclose(
        np.reciprocal(uniform),
        (temperatures_mev - theta_mev) / curie_numerator,
        rtol=1.0e-12,
    )
    # AF exchange suppresses uniform response and enhances the ordering wavevector.
    assert np.all(np.asarray(staggered) > np.asarray(uniform))
    np.testing.assert_allclose(
        staggered,
        curie_numerator / (temperatures_mev - abs(theta_mev)),
        rtol=1.0e-12,
    )


@pytest.mark.parametrize("inverse_mode_energy_sq", [0.0, 0.06])
def test_heisenberg_spectrum_kramers_kronig_equals_static_response(
    inverse_mode_energy_sq,
):
    """Causality fixes chi(0) from the positive-energy spectrum, with or without inertia."""
    q = 0.23
    chi0 = 0.7
    gamma0 = 1.4
    exchange = 0.22
    geometry = _chain_geometry([q])
    kwargs = {
        "chi0": chi0,
        "gamma0": gamma0,
        "j_values": {"J1": exchange},
        "inverse_mode_energy_sq": inverse_mode_energy_sq,
    }
    static = float(
        heisenberg_rpa_susceptibility(geometry, [0.0], **kwargs).real[0]
    )

    def kk_integrand(energy):
        dynamic = heisenberg_rpa_susceptibility(geometry, [energy], **kwargs)[0]
        return 2.0 * dynamic.imag / (np.pi * energy)

    integrated, _ = quad(kk_integrand, 1.0e-8, np.inf, epsabs=2.0e-9, limit=300)
    assert integrated == pytest.approx(static, rel=2.0e-7, abs=2.0e-9)


def test_one_dimensional_tight_binding_uniform_spin_dos():
    """At low temperature, chi_xx(q=0) tends to one half the 1D per-spin DOS."""
    hopping_mev = 20.0
    chemical_potential_mev = 5.0
    # The thermal width spans several points of this finite k mesh while
    # remaining negligible compared with the band scale.
    temperature_K = 0.5
    model = build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=[BasisState("s", site="A", orbital="s")],
        hoppings={(1, 0, 0): [[-hopping_mev]]},
        orbital_centers=[[0.25, 0.0, 0.0]],
        periodic_axes=(0,),
        energy_unit="meV",
    )
    response = bare_spin_susceptibility(
        model,
        np.zeros((3, 3)),
        [0.0, -1.0, 1.0],
        k_mesh(model, (8192,)),
        temperature_K=temperature_K,
        chemical_potential_meV=chemical_potential_mev,
        broadening_meV=0.3,
    )

    # For epsilon(k)=-2t cos(k), the per-spin DOS is
    # 1/(pi sqrt(4t^2-mu^2)); the Cartesian spin trace contributes 1/2.
    expected = 1.0 / (
        2.0
        * np.pi
        * np.sqrt(4.0 * hopping_mev**2 - chemical_potential_mev**2)
    )
    actual = float(isotropic_spin_component(response).real[0])
    assert actual == pytest.approx(expected, rel=2.0e-3)
    assert KB_MEV_PER_K * temperature_K < 0.05
    # Conserved total spin has no finite-energy uniform response. The finite
    # static value above is an equilibrium limit, not its dynamic continuation.
    np.testing.assert_allclose(response.values_per_meV_cell[1:], 0.0, atol=1e-14)
