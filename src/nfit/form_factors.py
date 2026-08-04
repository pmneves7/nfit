"""Magnetic form factors for neutron scattering.

The magnetic neutron scattering cross section from unpolarized neutrons is
proportional to ``|f(Q)|^2`` where ``f(Q)`` is the magnetic form factor of the
scattering ion. The radial integrals are tabulated as three-Gaussian analytic
approximations in ``s = |Q| / (4 pi) = sin(theta) / lambda`` (inverse Angstrom):

``<j0>(s) = A exp(-a s^2) + B exp(-b s^2) + C exp(-c s^2) + D``,
``<j2>(s) = s^2 [A exp(-a s^2) + B exp(-b s^2) + C exp(-c s^2) + D]``.

The extra ``s^2`` on ``<j2>`` is part of the tabulation, and makes
``<j2>(0) = 0`` as a spherical Bessel function of order two requires.

Spin-only and dipole approximation
----------------------------------
For a spin-only moment ``f(Q) = <j0(Q)>``. When the moment carries orbital
angular momentum, the dipole approximation is

``f(Q) = <j0(Q)> + (2 / g_J - 1) <j2(Q)>``,

with ``g_J`` the Lande factor of the ion. Writing the moment as
``mu = (<L> + 2<S>) mu_B`` and projecting onto ``J`` gives ``<L> = (2 - g_J) J``
and ``2<S> = 2 (g_J - 1) J``, so the orbital part weights ``<j0> + <j2>`` and
the spin part weights ``<j0>``; the ratio is the coefficient above. ``g_J = 2``
returns ``<j0>`` exactly, which is the default everywhere in nfit.

The correction is not small for rare earths: Yb(3+) has ``g_J = 8/7``, giving a
``<j2>`` coefficient of ``0.75``.

Coverage gaps: the tables have no ``5d`` ion (Re, Os, Ir, Pt, W, Ta), no
``Ru2``/``Ru3`` or ``Rh3``, and no ``Ce3``. This is a gap in Section 4.4.5
itself rather than a transcription omission, so pass explicit coefficients from
the literature for those ions. ``Pr3`` and ``O1`` have ``<j0>`` but no tabulated
``<j2>`` and therefore support only the spin-only form.

References
----------
- P. J. Brown, "Magnetic form factors", International Tables for
  Crystallography, Vol. C, Section 4.4.5.
- ILL form factor tables: https://www.ill.eu/sites/ccsl/ffacts/
- Coefficients transcribed from the public-domain ``periodictable`` package
  (P. Kienzle), which encodes the CrysFML / International Tables values.
  ``tests/test_form_factors.py`` re-derives both tables from it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

FORM_FACTOR_MODES = ("none", "single_ion", "custom", "mixture")

# <j0> coefficients (A, a, B, b, C, c, D), keyed by ion label "<Element><charge>"
# (e.g. "Mn2" for Mn2+, "Fe0" for neutral Fe). Source: see module docstring.
J0_COEFFICIENTS: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "Sc0": (0.251200, 90.029602, 0.329000, 39.402100, 0.423500, 14.322200, -0.004300),
    "Sc1": (0.488900, 51.160301, 0.520300, 14.076400, -0.028600, 0.179200, 0.018500),
    "Sc2": (0.504800, 31.403500, 0.518600, 10.989700, -0.024100, 1.183100, 0.000000),
    "Ti0": (0.465700, 33.589802, 0.549000, 9.879100, -0.029100, 0.323200, 0.012300),
    "Ti1": (0.509300, 36.703300, 0.503200, 10.371300, -0.026300, 0.310600, 0.011600),
    "Ti2": (0.509100, 24.976299, 0.516200, 8.756900, -0.028100, 0.916000, 0.001500),
    "Ti3": (0.357100, 22.841299, 0.668800, 8.930600, -0.035400, 0.483300, 0.009900),
    "V0": (0.408600, 28.810900, 0.607700, 8.543700, -0.029500, 0.276800, 0.012300),
    "V1": (0.444400, 32.647900, 0.568300, 9.097100, -0.228500, 0.021800, 0.215000),
    "V2": (0.408500, 23.852600, 0.609100, 8.245600, -0.167600, 0.041500, 0.149600),
    "V3": (0.359800, 19.336399, 0.663200, 7.617200, -0.306400, 0.029600, 0.283500),
    "V4": (0.310600, 16.816000, 0.719800, 7.048700, -0.052100, 0.302000, 0.022100),
    "Cr0": (0.113500, 45.199001, 0.348100, 19.493099, 0.547700, 7.354200, -0.009200),
    "Cr1": (-0.097700, 0.047000, 0.454400, 26.005400, 0.557900, 7.489200, 0.083100),
    "Cr2": (1.202400, -0.005500, 0.415800, 20.547501, 0.603200, 6.956000, -1.221800),
    "Cr3": (-0.309400, 0.027400, 0.368000, 17.035500, 0.655900, 6.523600, 0.285600),
    "Cr4": (-0.232000, 0.043300, 0.310100, 14.951800, 0.718200, 6.172600, 0.204200),
    "Mn0": (0.243800, 24.962900, 0.147200, 15.672800, 0.618900, 6.540300, -0.010500),
    "Mn1": (-0.013800, 0.421300, 0.423100, 24.667999, 0.590500, 6.654500, -0.001000),
    "Mn2": (0.422000, 17.684000, 0.594800, 6.005000, 0.004300, -0.609000, -0.021900),
    "Mn3": (0.419800, 14.282900, 0.605400, 5.468900, 0.924100, -0.008800, -0.949800),
    "Mn4": (0.376000, 12.566100, 0.660200, 5.132900, -0.037200, 0.563000, 0.001100),
    "Fe0": (0.070600, 35.008499, 0.358900, 15.358300, 0.581900, 5.560600, -0.011400),
    "Fe1": (0.125100, 34.963299, 0.362900, 15.514400, 0.522300, 5.591400, -0.010500),
    "Fe2": (0.026300, 34.959702, 0.366800, 15.943500, 0.618800, 5.593500, -0.011900),
    "Fe3": (0.397200, 13.244200, 0.629500, 4.903400, -0.031400, 0.349600, 0.004400),
    "Fe4": (0.378200, 11.380000, 0.655600, 4.592000, -0.034600, 0.483300, 0.000500),
    "Co0": (0.413900, 16.161600, 0.601300, 4.780500, -0.151800, 0.021000, 0.134500),
    "Co1": (0.099000, 33.125198, 0.364500, 15.176800, 0.547000, 5.008100, -0.010900),
    "Co2": (0.433200, 14.355300, 0.585700, 4.607700, -0.038200, 0.133800, 0.017900),
    "Co3": (0.390200, 12.507800, 0.632400, 4.457400, -0.150000, 0.034300, 0.127200),
    "Co4": (0.351500, 10.778500, 0.677800, 4.234300, -0.038900, 0.240900, 0.009800),
    "Ni0": (-0.017200, 35.739201, 0.317400, 14.268900, 0.713600, 4.566100, -0.014300),
    "Ni1": (0.070500, 35.856098, 0.398400, 13.804200, 0.542700, 4.396500, -0.011800),
    "Ni2": (0.016300, 35.882599, 0.391600, 13.223300, 0.605200, 4.338800, -0.013300),
    "Ni3": (-0.013400, 35.867699, 0.267800, 12.332600, 0.761400, 4.236900, -0.016200),
    "Ni4": (-0.009000, 35.861401, 0.277600, 11.790400, 0.747400, 4.201100, -0.016300),
    "Cu0": (0.090900, 34.983799, 0.408800, 11.443200, 0.512800, 3.824800, -0.012400),
    "Cu1": (0.074900, 34.965599, 0.414700, 11.764200, 0.523800, 3.849700, -0.012700),
    "Cu2": (0.023200, 34.968601, 0.402300, 11.564000, 0.588200, 3.842800, -0.013700),
    "Cu3": (0.003100, 34.907398, 0.358200, 10.913800, 0.653100, 3.827900, -0.014700),
    "Cu4": (-0.013200, 30.681700, 0.280100, 11.162600, 0.749000, 3.817200, -0.016500),
    "Y0": (0.591500, 67.608101, 1.512300, 17.900400, -1.113000, 14.135900, 0.008000),
    "Zr0": (0.410600, 59.996101, 1.054300, 18.647600, -0.475100, 10.540000, 0.010600),
    "Zr1": (0.453200, 59.594799, 0.783400, 21.435699, -0.245100, 9.036000, 0.009800),
    "Nb0": (0.394600, 49.229698, 1.319700, 14.821600, -0.726900, 9.615600, 0.012900),
    "Nb1": (0.457200, 49.918201, 1.027400, 15.725600, -0.496200, 9.157300, 0.011800),
    "Mo0": (0.180600, 49.056801, 1.230600, 14.785900, -0.426800, 6.986600, 0.017100),
    "Mo1": (0.350000, 48.035400, 1.030500, 15.060400, -0.392900, 7.479000, 0.013900),
    "Tc0": (0.129800, 49.661098, 1.165600, 14.130700, -0.313400, 5.512900, 0.019500),
    "Tc1": (0.267400, 48.956600, 0.956900, 15.141300, -0.238700, 5.457800, 0.016000),
    "Ru0": (0.106900, 49.423801, 1.191200, 12.741700, -0.317600, 4.912500, 0.021300),
    "Ru1": (0.441000, 33.308601, 1.477500, 9.553100, -0.936100, 6.722000, 0.017600),
    "Rh0": (0.097600, 49.882500, 1.160100, 11.830700, -0.278900, 4.126600, 0.023400),
    "Rh1": (0.334200, 29.756399, 1.220900, 9.438400, -0.575500, 5.332000, 0.021000),
    "Pd0": (0.200300, 29.363300, 1.144600, 9.599300, -0.368900, 4.042300, 0.025100),
    "Pd1": (0.503300, 24.503700, 1.998200, 6.908200, -1.524000, 5.513300, 0.021300),
    "Ce2": (0.295300, 17.684601, 0.292300, 6.732900, 0.431300, 5.382700, -0.019400),
    "Pr3": (0.050400, 24.998900, 0.257200, 12.037700, 0.714200, 5.003900, -0.021900),
    "Nd2": (0.164500, 25.045300, 0.252200, 11.978200, 0.601200, 4.946100, -0.018000),
    "Nd3": (0.054000, 25.029301, 0.310100, 12.102000, 0.657500, 4.722300, -0.021600),
    "Sm2": (0.090900, 25.203199, 0.303700, 11.856200, 0.625000, 4.236600, -0.020000),
    "Sm3": (0.028800, 25.206800, 0.297300, 11.831100, 0.695400, 4.211700, -0.021300),
    "Eu2": (0.075500, 25.296000, 0.300100, 11.599300, 0.643800, 4.025200, -0.019600),
    "Eu3": (0.020400, 25.307800, 0.301000, 11.474400, 0.700500, 3.942000, -0.022000),
    "Gd2": (0.063600, 25.382299, 0.303300, 11.212500, 0.652800, 3.787700, -0.019900),
    "Gd3": (0.018600, 25.386700, 0.289500, 11.142100, 0.713500, 3.752000, -0.021700),
    "Tb2": (0.054700, 25.508600, 0.317100, 10.591100, 0.649000, 3.517100, -0.021200),
    "Tb3": (0.017700, 25.509501, 0.292100, 10.576900, 0.713300, 3.512200, -0.023100),
    "Dy2": (0.130800, 18.315500, 0.311800, 7.664500, 0.579500, 3.146900, -0.022600),
    "Dy3": (0.115700, 15.073200, 0.327000, 6.799100, 0.582100, 3.020200, -0.024900),
    "Ho2": (0.099500, 18.176100, 0.330500, 7.855600, 0.592100, 2.979900, -0.023000),
    "Ho3": (0.056600, 18.317600, 0.336500, 7.688000, 0.631700, 2.942700, -0.024800),
    "Er2": (0.112200, 18.122299, 0.346200, 6.910600, 0.564900, 2.761400, -0.023500),
    "Er3": (0.058600, 17.980200, 0.354000, 7.096400, 0.612600, 2.748200, -0.025100),
    "Tm2": (0.098300, 18.323601, 0.338000, 6.917800, 0.587500, 2.662200, -0.024100),
    "Tm3": (0.058100, 15.092200, 0.278700, 7.801500, 0.685400, 2.793100, -0.022400),
    "Yb2": (0.085500, 18.512300, 0.294300, 7.373400, 0.641200, 2.677700, -0.021300),
    "Yb3": (0.041600, 16.094900, 0.284900, 7.834100, 0.696100, 2.672500, -0.022900),
    "U3": (0.505800, 23.288200, 1.346400, 7.002800, -0.872400, 4.868300, 0.019200),
    "U4": (0.329100, 23.547501, 1.083600, 8.454000, -0.434000, 4.119600, 0.021400),
    "U5": (0.365000, 19.803801, 3.219900, 6.281800, -2.607700, 5.301000, 0.023300),
    "Np3": (0.515700, 20.865400, 2.278400, 5.893000, -1.816300, 4.845700, 0.021100),
    "Np4": (0.420600, 19.804600, 2.800400, 5.978300, -2.243600, 4.984800, 0.022800),
    "Np5": (0.369200, 18.190001, 3.151000, 5.850000, -2.544600, 4.916400, 0.024800),
    "Np6": (0.292900, 17.561100, 3.486600, 5.784700, -2.806600, 4.870700, 0.026700),
    "Pu3": (0.384000, 16.679300, 3.104900, 5.421000, -2.514800, 4.551200, 0.026300),
    "Pu4": (0.493400, 16.835501, 1.639400, 5.638400, -1.158100, 4.139900, 0.024800),
    "Pu5": (0.388800, 16.559200, 2.036200, 5.656700, -1.451500, 4.255200, 0.026700),
    "Pu6": (0.317200, 16.050699, 3.465400, 5.350700, -2.810200, 4.513300, 0.028100),
    "Am2": (0.474300, 21.776100, 1.580000, 5.690200, -1.077900, 4.145100, 0.021800),
    "Am3": (0.423900, 19.573900, 1.457300, 5.872200, -0.905200, 3.968200, 0.023800),
    "Am4": (0.373700, 17.862499, 1.352100, 6.042600, -0.751400, 3.719900, 0.025800),
    "Am5": (0.295600, 17.372499, 1.452500, 6.073400, -0.775500, 3.661900, 0.027700),
    "Am6": (0.230200, 16.953300, 1.486400, 6.115900, -0.745700, 3.542600, 0.029400),
    "Am7": (0.360100, 12.729900, 1.964000, 5.120300, -1.356000, 3.714200, 0.031600),
    "O1": (0.115285, 85.197300, 0.556229, 25.252200, 0.332476, 6.362070, -0.004607),
}


# <j2> coefficients (A, a, B, b, C, c, D) in the same ion-label keying and
# from the same source as J0_COEFFICIENTS. <j2> carries an extra s^2 factor
# (see the module docstring). "Pr3" and "O1" have <j0> but no tabulated <j2>.
J2_COEFFICIENTS: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "Sc0": (10.817200, 54.327000, 4.735300, 14.847000, 0.607100, 4.218000, 0.001100),
    "Sc1": (8.502100, 34.285000, 3.211600, 10.994000, 0.424400, 3.605000, 0.000900),
    "Sc2": (4.368300, 28.654000, 3.723100, 10.823000, 0.607400, 3.668000, 0.001400),
    "Ti0": (4.358300, 36.056000, 3.823000, 11.133000, 0.685500, 3.469000, 0.002000),
    "Ti1": (6.156700, 27.275000, 2.683300, 8.983000, 0.407000, 3.052000, 0.001100),
    "Ti2": (4.310700, 18.348000, 2.096000, 6.797000, 0.298400, 2.548000, 0.000700),
    "Ti3": (3.371700, 14.444000, 1.825800, 5.713000, 0.247000, 2.265000, 0.000500),
    "V0": (3.809900, 21.347000, 2.329500, 7.409000, 0.433300, 2.632000, 0.001500),
    "V1": (4.747400, 23.323000, 2.360900, 7.808000, 0.410500, 2.706000, 0.001400),
    "V2": (3.438600, 16.530000, 1.963800, 6.141000, 0.299700, 2.267000, 0.000900),
    "V3": (2.300500, 14.682000, 2.036400, 6.130000, 0.409900, 2.382000, 0.001400),
    "V4": (1.837700, 12.267000, 1.824700, 5.458000, 0.397900, 2.248000, 0.001200),
    "Cr0": (3.408500, 20.127000, 2.100600, 6.802000, 0.426600, 2.394000, 0.001900),
    "Cr1": (3.776800, 20.346000, 2.102800, 6.893000, 0.401000, 2.411000, 0.001700),
    "Cr2": (2.642200, 16.060000, 1.919800, 6.253000, 0.444600, 2.372000, 0.002000),
    "Cr3": (1.626200, 15.066000, 2.061800, 6.284000, 0.528100, 2.368000, 0.002300),
    "Cr4": (1.029300, 13.950000, 1.993300, 6.059000, 0.597400, 2.346000, 0.002700),
    "Mn0": (2.668100, 16.060000, 1.756100, 5.640000, 0.367500, 2.049000, 0.001700),
    "Mn1": (3.295300, 18.695000, 1.879200, 6.240000, 0.392700, 2.201000, 0.002200),
    "Mn2": (2.051500, 15.556000, 1.884100, 6.063000, 0.478700, 2.232000, 0.002700),
    "Mn3": (1.242700, 14.997000, 1.956700, 6.118000, 0.573200, 2.258000, 0.003100),
    "Mn4": (0.787900, 13.886000, 1.871700, 5.743000, 0.598100, 2.182000, 0.003400),
    "Fe0": (1.940500, 18.473000, 1.956600, 6.323000, 0.516600, 2.161000, 0.003600),
    "Fe1": (2.629000, 18.660000, 1.870400, 6.331000, 0.469000, 2.163000, 0.003100),
    "Fe2": (1.649000, 16.559000, 1.906400, 6.133000, 0.520600, 2.137000, 0.003500),
    "Fe3": (1.360200, 11.998000, 1.518800, 5.003000, 0.470500, 1.991000, 0.003800),
    "Fe4": (1.558200, 8.275000, 1.186300, 3.279000, 0.136600, 1.107000, -0.002200),
    "Co0": (1.967800, 14.170000, 1.491100, 4.948000, 0.384400, 1.797000, 0.002700),
    "Co1": (2.409700, 16.161000, 1.578000, 5.460000, 0.409500, 1.914000, 0.003100),
    "Co2": (1.904900, 11.644000, 1.315900, 4.357000, 0.314600, 1.645000, 0.001700),
    "Co3": (1.705800, 8.859000, 1.140900, 3.309000, 0.147400, 1.090000, -0.002500),
    "Co4": (1.311000, 8.025000, 1.155100, 3.179000, 0.160800, 1.130000, -0.001100),
    "Ni0": (1.030200, 12.252000, 1.466900, 4.745000, 0.452100, 1.744000, 0.003600),
    "Ni1": (2.104000, 14.866000, 1.430200, 5.071000, 0.403100, 1.778000, 0.003400),
    "Ni2": (1.708000, 11.016000, 1.214700, 4.103000, 0.315000, 1.533000, 0.001800),
    "Ni3": (1.161200, 7.700000, 1.002700, 3.263000, 0.271900, 1.378000, 0.002500),
    "Ni4": (1.161200, 7.700000, 1.002700, 3.263000, 0.271900, 1.378000, 0.002500),
    "Cu0": (1.918200, 14.490000, 1.332900, 4.730000, 0.384200, 1.639000, 0.003500),
    "Cu1": (1.881400, 13.433000, 1.280900, 4.545000, 0.364600, 1.602000, 0.003300),
    "Cu2": (1.518900, 10.478000, 1.151200, 3.813000, 0.291800, 1.398000, 0.001700),
    "Cu3": (1.279700, 8.450000, 1.031500, 3.280000, 0.240100, 1.250000, 0.001500),
    "Cu4": (0.956800, 7.448000, 0.909900, 3.396000, 0.372900, 1.494000, 0.004900),
    "Y0": (14.408400, 44.658000, 5.104500, 14.904000, -0.053500, 3.319000, 0.002800),
    "Zr0": (10.137800, 35.337000, 4.773400, 12.545000, -0.048900, 2.672000, 0.003600),
    "Zr1": (11.872200, 34.920000, 4.050200, 12.127000, -0.063200, 2.828000, 0.003400),
    "Nb0": (7.479600, 33.179000, 5.088400, 11.571000, -0.028100, 1.564000, 0.004700),
    "Nb1": (8.773500, 33.285000, 4.655600, 11.605000, -0.026800, 1.539000, 0.004400),
    "Mo0": (5.118000, 23.422000, 4.180900, 9.208000, -0.050500, 1.743000, 0.005300),
    "Mo1": (7.236700, 28.128000, 4.070500, 9.923000, -0.031700, 1.455000, 0.004900),
    "Tc0": (4.244100, 21.397000, 3.943900, 8.375000, -0.037100, 1.187000, 0.006600),
    "Tc1": (6.405600, 24.824000, 3.540000, 8.611000, -0.036600, 1.485000, 0.004400),
    "Ru0": (3.744500, 18.613000, 3.474900, 7.420000, -0.036300, 1.007000, 0.007300),
    "Ru1": (5.282600, 23.683000, 3.581300, 8.152000, -0.025700, 0.426000, 0.013100),
    "Rh0": (3.365100, 17.344000, 3.212100, 6.804000, -0.035000, 0.503000, 0.014600),
    "Rh1": (4.026000, 18.950000, 3.166300, 7.000000, -0.029600, 0.486000, 0.012700),
    "Pd0": (3.310500, 14.726000, 2.633200, 5.862000, -0.043700, 1.130000, 0.005300),
    "Pd1": (4.274900, 17.900000, 2.702100, 6.354000, -0.025800, 0.700000, 0.007100),
    "Ce2": (0.980900, 18.063000, 1.841300, 7.769000, 0.990500, 2.845000, 0.012000),
    "Nd2": (1.453000, 18.340000, 1.619600, 7.285000, 0.875200, 2.622000, 0.012600),
    "Nd3": (0.675100, 18.342000, 1.627200, 7.260000, 0.964400, 2.602000, 0.015000),
    "Sm2": (1.036000, 18.425000, 1.476900, 7.032000, 0.881000, 2.437000, 0.015200),
    "Sm3": (0.470700, 18.430000, 1.426100, 7.034000, 0.957400, 2.439000, 0.018200),
    "Eu2": (0.897000, 18.443000, 1.376900, 7.005000, 0.906000, 2.421000, 0.019000),
    "Eu3": (0.398500, 18.451000, 1.330700, 6.956000, 0.960300, 2.378000, 0.019700),
    "Gd2": (0.775600, 18.469000, 1.312400, 6.899000, 0.895600, 2.338000, 0.019900),
    "Gd3": (0.334700, 18.476000, 1.246500, 6.877000, 0.953700, 2.318000, 0.021700),
    "Tb2": (0.668800, 18.491000, 1.248700, 6.822000, 0.888800, 2.275000, 0.021500),
    "Tb3": (0.289200, 18.497000, 1.167800, 6.797000, 0.943700, 2.257000, 0.023200),
    "Dy2": (0.591700, 18.511000, 1.182800, 6.747000, 0.880100, 2.214000, 0.022900),
    "Dy3": (0.252300, 18.517000, 1.091400, 6.736000, 0.934500, 2.208000, 0.025000),
    "Ho2": (0.509400, 18.515000, 1.123400, 6.706000, 0.872700, 2.159000, 0.024200),
    "Ho3": (0.218800, 18.516000, 1.024000, 6.707000, 0.925100, 2.161000, 0.026800),
    "Er2": (0.469300, 18.528000, 1.054500, 6.649000, 0.867900, 2.120000, 0.026100),
    "Er3": (0.171000, 18.534000, 0.987900, 6.625000, 0.904400, 2.100000, 0.027800),
    "Tm2": (0.419800, 18.542000, 0.995900, 6.600000, 0.859300, 2.082000, 0.028400),
    "Tm3": (0.176000, 18.542000, 0.910500, 6.579000, 0.897000, 2.062000, 0.029400),
    "Yb2": (0.385200, 18.550000, 0.941500, 6.551000, 0.849200, 2.043000, 0.030100),
    "Yb3": (0.157000, 18.555000, 0.848400, 6.540000, 0.888000, 2.037000, 0.031800),
    "U3": (4.158200, 16.534000, 2.467500, 5.952000, -0.025200, 0.765000, 0.005700),
    "U4": (3.744900, 13.894000, 2.645300, 4.863000, -0.521800, 3.192000, 0.000900),
    "U5": (3.072400, 12.546000, 2.307600, 5.231000, -0.064400, 1.474000, 0.003500),
    "Np3": (3.717000, 15.133000, 2.321600, 5.503000, -0.027500, 0.800000, 0.005200),
    "Np4": (2.920300, 14.646000, 2.597900, 5.559000, -0.030100, 0.367000, 0.014100),
    "Np5": (2.330800, 13.654000, 2.721900, 5.494000, -0.135700, 0.049000, 0.122400),
    "Np6": (1.824500, 13.180000, 2.850800, 5.407000, -0.157900, 0.044000, 0.143800),
    "Pu3": (2.088500, 12.871000, 2.596100, 5.190000, -0.146500, 0.039000, 0.134300),
    "Pu4": (2.724400, 12.926000, 2.338700, 5.163000, -0.130000, 0.046000, 0.117700),
    "Pu5": (2.140900, 12.832000, 2.566400, 5.152000, -0.133800, 0.046000, 0.121000),
    "Pu6": (1.726200, 12.324000, 2.665200, 5.066000, -0.169500, 0.041000, 0.155000),
    "Am2": (3.523700, 15.955000, 2.285500, 5.195000, -0.014200, 0.585000, 0.003300),
    "Am3": (2.862200, 14.733000, 2.409900, 5.144000, -0.132600, 0.031000, 0.123300),
    "Am4": (2.414100, 12.948000, 2.368700, 4.945000, -0.249000, 0.022000, 0.237100),
    "Am5": (2.010900, 12.053000, 2.415500, 4.836000, -0.226400, 0.027000, 0.212800),
    "Am6": (1.677800, 11.337000, 2.453100, 4.725000, -0.204300, 0.034000, 0.189200),
    "Am7": (1.884500, 9.161000, 2.074600, 4.042000, -0.131800, 1.723000, 0.002000),
}


def available_ions() -> list[str]:
    """Return the ion labels with tabulated ``<j0>`` coefficients."""

    return sorted(J0_COEFFICIENTS)


def available_dipole_ions() -> list[str]:
    """Return the ion labels usable with the ``g_J`` dipole approximation."""

    return sorted(J2_COEFFICIENTS)


def dipole_j2_weight(g_J: float) -> float:
    """Return the ``<j2>`` weight ``2 / g_J - 1`` of the dipole approximation.

    ``g_J = 2`` gives zero, i.e. the spin-only ``<j0>`` form factor.
    """

    value = float(g_J)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("g_J must be finite and positive")
    return 2.0 / value - 1.0


def _resolve_coefficients(
    ion: str | None,
    coefficients: ArrayLike | None,
    table: dict[str, tuple[float, float, float, float, float, float, float]] = None,
    label: str = "<j0>",
) -> tuple[float, float, float, float, float, float, float]:
    lookup = J0_COEFFICIENTS if table is None else table
    values = _coerce_coefficients(coefficients)
    if values is not None:
        return values
    if not ion:
        raise ValueError("provide either an ion label or explicit coefficients")
    key = str(ion).strip()
    try:
        return lookup[key]
    except KeyError:
        known = ", ".join(sorted(lookup))
        raise KeyError(
            f"magnetic ion {key!r} has no tabulated {label} coefficients; "
            f"ions with {label}: {known}"
        ) from None


def _coerce_coefficients(
    coefficients: ArrayLike | None,
) -> tuple[float, float, float, float, float, float, float] | None:
    if coefficients is None:
        return None
    if isinstance(coefficients, str):
        stripped = coefficients.strip()
        if not stripped:
            return None
        pieces = [piece for piece in stripped.replace(",", " ").split() if piece]
        try:
            values = tuple(float(piece) for piece in pieces)
        except ValueError as exc:
            raise ValueError(
                "form factor coefficients must be seven comma-separated numbers "
                "(A, a, B, b, C, c, D); "
                f"got {coefficients!r}"
            ) from exc
    else:
        try:
            values = tuple(float(v) for v in np.asarray(coefficients, dtype=float).ravel())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "form factor coefficients must be seven numbers "
                "(A, a, B, b, C, c, D)"
            ) from exc
    if len(values) == 0:
        return None
    if len(values) != 7:
        raise ValueError(
            "form factor coefficients must be seven numbers (A, a, B, b, C, c, D); "
            f"got {len(values)}"
        )
    return values


def magnetic_form_factor_j0(
    q_modulus_inv_angstrom: ArrayLike,
    *,
    ion: str | None = None,
    coefficients: ArrayLike | None = None,
) -> FloatArray:
    """Return the ``<j0>`` magnetic form factor ``f(|Q|)``.

    ``f(s) = A exp(-a s^2) + B exp(-b s^2) + C exp(-c s^2) + D`` with
    ``s = |Q|/(4 pi)`` in inverse Angstrom. Pass either a tabulated ``ion``
    label (see :func:`available_ions`) or explicit ``coefficients``
    ``(A, a, B, b, C, c, D)`` from the ILL tables; explicit coefficients take
    precedence.
    """

    A, a, B, b, C, c, D = _resolve_coefficients(ion, coefficients)
    q = np.asarray(q_modulus_inv_angstrom, dtype=float)
    s_sq = (q / (4.0 * np.pi)) ** 2
    return A * np.exp(-a * s_sq) + B * np.exp(-b * s_sq) + C * np.exp(-c * s_sq) + D


def magnetic_form_factor_j2(
    q_modulus_inv_angstrom: ArrayLike,
    *,
    ion: str | None = None,
    coefficients: ArrayLike | None = None,
) -> FloatArray:
    """Return the ``<j2>`` radial integral.

    ``<j2>(s) = s^2 [A exp(-a s^2) + B exp(-b s^2) + C exp(-c s^2) + D]`` with
    ``s = |Q|/(4 pi)`` in inverse Angstrom. The leading ``s^2`` is part of the
    tabulation, so ``<j2>(0) = 0``. Pass a tabulated ``ion`` label (see
    :func:`available_dipole_ions`) or explicit ``coefficients``.
    """

    A, a, B, b, C, c, D = _resolve_coefficients(
        ion, coefficients, J2_COEFFICIENTS, "<j2>"
    )
    q = np.asarray(q_modulus_inv_angstrom, dtype=float)
    s_sq = (q / (4.0 * np.pi)) ** 2
    return s_sq * (
        A * np.exp(-a * s_sq) + B * np.exp(-b * s_sq) + C * np.exp(-c * s_sq) + D
    )


def magnetic_form_factor(
    q_modulus_inv_angstrom: ArrayLike,
    *,
    ion: str | None = None,
    coefficients: ArrayLike | None = None,
    j2_coefficients: ArrayLike | None = None,
    g_J: float = 2.0,
) -> FloatArray:
    """Return ``f(|Q|)`` in the spin-only or ``g_J`` dipole approximation.

    ``g_J = 2`` (the default) returns ``<j0>`` unchanged and never consults the
    ``<j2>`` table, so a spin-only ion needs no ``<j2>`` entry. Any other
    ``g_J`` adds ``(2 / g_J - 1) <j2>``; see the module docstring for the
    derivation and :func:`dipole_j2_weight` for the coefficient alone.
    """

    j0 = magnetic_form_factor_j0(
        q_modulus_inv_angstrom, ion=ion, coefficients=coefficients
    )
    weight = dipole_j2_weight(g_J)
    if weight == 0.0 and j2_coefficients is None:
        return j0
    j2 = magnetic_form_factor_j2(
        q_modulus_inv_angstrom, ion=ion, coefficients=j2_coefficients
    )
    return j0 + weight * j2


def form_factor_sq(
    q_modulus_inv_angstrom: ArrayLike,
    *,
    ion: str | None = None,
    coefficients: ArrayLike | None = None,
    j2_coefficients: ArrayLike | None = None,
    g_J: float = 2.0,
) -> FloatArray:
    """Return ``|f(|Q|)|^2`` in the spin-only or ``g_J`` dipole approximation."""

    f = magnetic_form_factor(
        q_modulus_inv_angstrom,
        ion=ion,
        coefficients=coefficients,
        j2_coefficients=j2_coefficients,
        g_J=g_J,
    )
    return f * f


def normalized_form_factor_mode(config: Mapping[str, Any]) -> str:
    """Return the explicit or backward-compatible form-factor profile mode.

    Projects written before form-factor profiles used only ``ion`` and
    ``form_factor_coefficients``. Their effective mode is inferred when the
    new ``form_factor_mode`` field is absent.
    """

    if "form_factor_mode" in config:
        mode = str(config.get("form_factor_mode", "none")).strip().lower()
    elif config.get("form_factor_coefficients"):
        mode = "custom"
    elif str(config.get("ion", "") or "").strip() not in {"", "__custom__"}:
        mode = "single_ion"
    else:
        mode = "none"
    if mode not in FORM_FACTOR_MODES:
        choices = ", ".join(FORM_FACTOR_MODES)
        raise ValueError(f"form_factor_mode must be one of {choices}")
    return mode


def normalize_form_factor_mixture(
    mixture: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    """Validate a coherent normalized mixture of magnetic amplitudes.

    Each term supplies a nonnegative ``weight`` and either a tabulated ``ion``
    or explicit seven-value ``coefficients``. Optional ``g_J`` and
    ``j2_coefficients`` select the same dipole approximation as
    :func:`magnetic_form_factor`. Weights are amplitudes, not intensity
    fractions, and must sum to one.
    """

    if mixture is None or isinstance(mixture, (str, bytes)):
        raise ValueError("form_factor_mixture must be a nonempty list of mappings")
    terms: list[dict[str, Any]] = []
    for index, raw in enumerate(mixture):
        if not isinstance(raw, Mapping):
            raise ValueError(f"form_factor_mixture term {index + 1} must be a mapping")
        weight = float(raw.get("weight", np.nan))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"form_factor_mixture term {index + 1} weight must be finite and nonnegative"
            )
        ion = str(raw.get("ion", "") or "").strip()
        coefficients = _coerce_coefficients(raw.get("coefficients"))
        if bool(ion) == (coefficients is not None):
            raise ValueError(
                f"form_factor_mixture term {index + 1} must provide exactly one "
                "of ion or coefficients"
            )
        g_J = float(raw.get("g_J", 2.0))
        dipole_j2_weight(g_J)
        j2 = _coerce_coefficients(raw.get("j2_coefficients"))
        # Validate table coverage now rather than during a long fit.
        magnetic_form_factor(
            np.asarray([0.0]),
            ion=ion or None,
            coefficients=coefficients,
            j2_coefficients=j2,
            g_J=g_J,
        )
        term: dict[str, Any] = {"weight": weight, "g_J": g_J}
        if ion:
            term["ion"] = ion
        else:
            term["coefficients"] = list(coefficients or ())
        if j2 is not None:
            term["j2_coefficients"] = list(j2)
        terms.append(term)
    if not terms:
        raise ValueError("form_factor_mixture must contain at least one term")
    total = float(sum(term["weight"] for term in terms))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"form_factor_mixture amplitude weights must sum to one; got {total:g}"
        )
    return tuple(terms)


def magnetic_form_factor_profile(
    q_modulus_inv_angstrom: ArrayLike,
    config: Mapping[str, Any],
) -> FloatArray:
    """Evaluate a serialized shared/effective magnetic form-factor profile.

    ``none`` returns one, ``single_ion`` and ``custom`` use the existing
    dipole implementation, and ``mixture`` forms a coherent weighted sum of
    amplitudes before squaring. This shared profile is appropriate when every
    spin-carrying orbital uses the same effective radial magnetization density.
    """

    q = np.asarray(q_modulus_inv_angstrom, dtype=float)
    mode = normalized_form_factor_mode(config)
    if mode == "none":
        return np.ones_like(q, dtype=float)
    if mode in {"single_ion", "custom"}:
        ion = str(config.get("ion", "") or "").strip()
        if ion == "__custom__":
            ion = ""
        coefficients = config.get("form_factor_coefficients")
        if mode == "single_ion" and not ion:
            raise ValueError("single_ion form-factor mode requires an ion")
        if mode == "custom" and not coefficients:
            raise ValueError("custom form-factor mode requires coefficients")
        return magnetic_form_factor(
            q,
            ion=ion or None,
            coefficients=coefficients,
            j2_coefficients=config.get("form_factor_j2_coefficients"),
            g_J=float(config.get("form_factor_g_J", 2.0)),
        )
    terms = normalize_form_factor_mixture(config.get("form_factor_mixture"))
    amplitude = np.zeros_like(q, dtype=float)
    for term in terms:
        amplitude += float(term["weight"]) * magnetic_form_factor(
            q,
            ion=term.get("ion"),
            coefficients=term.get("coefficients"),
            j2_coefficients=term.get("j2_coefficients"),
            g_J=float(term.get("g_J", 2.0)),
        )
    return amplitude


def form_factor_profile_sq(
    q_modulus_inv_angstrom: ArrayLike,
    config: Mapping[str, Any],
) -> FloatArray:
    """Return ``|f(Q)|^2`` for a serialized shared/effective profile."""

    amplitude = magnetic_form_factor_profile(q_modulus_inv_angstrom, config)
    return np.asarray(amplitude, dtype=float) ** 2
