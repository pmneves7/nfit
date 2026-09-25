# Validating magnetic response calculations

The [Heisenberg RPA](heisenberg_rpa.md) and [Lindhard](lindhard.md) models
describe different approximations. Heisenberg RPA feeds exchange fields back
into a chosen local response; Lindhard evaluates independent-electron
particle--hole transitions. Agreement with their defining equations does not
establish that either describes a particular material.

## Analytical checks

The regression suite checks the following independently of fitted data.
Energies are in meV and susceptibilities use dimensionless spin operators;
see [Physics conventions](physics_conventions.md).

| Check | Required result | Regression coverage |
| --- | --- | --- |
| Heisenberg exchange sign and bond counting | A nearest-neighbor chain has $J(q)=2J\cos(2\pi q)$; $J>0$ favors $q=0$, $J<0$ favors $q=1/2$ | `test_spin_fluctuations.py`, `test_physics_benchmarks.py` |
| Free-spin molecular-field limit | $\Theta_{\rm CW}=zJS(S+1)/(3k_B)$ and $\chi(0,0)=S(S+1)/[3k_B(T-\Theta_{\rm CW})]$ | `test_physics_benchmarks.py` |
| Causal scalar dynamics | Positive-energy Kramers--Kronig integral recovers the static response for both relaxational and inertial Heisenberg modes | `test_physics_benchmarks.py` |
| Cell and phase convention | Equivalent enlarged and primitive magnetic cells give the same per-site observable; mode weights sum to one | `test_spin_fluctuations.py`, `test_tensor_rpa.py` |
| Tensor response | Isotropic neutron contraction gives twice one Cartesian component; global frame rotation preserves intensity | `test_tensor_rpa.py` |
| Field response | Correct static tensor, circular precession sign, nonnegative absorption for the local relaxor, and static dispersion integral | `test_tensor_rpa.py` |
| Local moment integral | Site trace agrees with direct quadrature and the classical equipartition limit; closures satisfy their stated scalar equations | `test_sum_rules.py`, `test_closures.py` |
| Lindhard operator algebra | Complex two-level transition matrix elements agree with an independent spectral formula | `test_electronic_response.py` |
| Pauli spin normalization | Uniform static response equals one half the thermally smeared per-spin DOS; implicit and explicit spin agree | `test_electronic_response.py`, `test_physics_benchmarks.py` |
| Electronic cell normalization | Conversion to formula-unit and magnetic-center bases is independent of orbital radial form factors | `test_electronic_normalization.py`, `test_electronic_response.py` |
| Numerical execution | NumPy, Numba, and factorized orbital-pair contractions agree; CuPy equivalence runs when available | `test_electronic_response.py` |

Here $z$ is the coordination number, $J$ a nearest-neighbor exchange in meV,
$S$ the spin quantum number, $k_B$ in meV/K, $T$ and $\Theta_{\rm CW}$ in K,
and $q$ reduced momentum along the chain. The tight-binding DOS benchmark
uses $\epsilon(k)=-2t\cos(2\pi k)$, with hopping magnitude $t$ in meV.
Inside the band, the per-spin density of states is
$D_\uparrow(\mu)=1/[\pi\sqrt{4t^2-\mu^2}]$ per meV per model cell,
where $\mu$ is chemical potential. Its Cartesian spin response tends to
$D_\uparrow(\mu)/2$ at low temperature. Tests use a temperature resolved by
the numerical mesh rather than an unresolved delta function.

Two small field-on tensor cases guard against using an isotropic local
response accidentally. Set the field along $z$ and
$C=\operatorname{diag}(2,2,1)$ meV$^{-1}$. For
$J=\operatorname{diag}(0.6,0,0)$ meV, the largest feedback eigenvalue is
$1.2$, so the stability margin is $-0.2$ and the state must be rejected;
using only the longitudinal local susceptibility would incorrectly give
$+0.4$. For the stable case $J_{xz}=J_{zx}=0.2$ meV with all other entries
zero, the full static inverse gives
$\chi_{zz}=1/(1-2\times0.2^2)=1.0869565$ meV$^{-1}$.
Replacing $C$ by the scalar longitudinal response instead gives
$1.0416667$ meV$^{-1}$. These checks cover the spectral evaluator,
bulk evaluator, closure stability, and diagnostic margin.

## Literature correspondence

- Wysin's [Onsager reaction-field paper](https://arxiv.org/abs/cond-mat/9909266),
  Eq. (12), has the same $1-\chi_0(J-\lambda)$ denominator and reaction-field
  sign. Its classical equilibrium construction is a reference for the
  feedback, not a derivation of nfit's frequency dependence.
- Graser *et al.*, [multiorbital response](https://arxiv.org/abs/0812.0343),
  Eqs. (12)--(14), give the orbital bubble and band-eigenvector factors.
  Comparing individual tensor components requires matching the ordered
  probe/adjoint indices, momentum signs, and spin normalization.
- [Mermin's relaxation-time analysis](https://doi.org/10.1103/PhysRevB.1.2362)
  explains why constant denominator broadening alone is not a conserving
  treatment of collisions. nfit's Lindhard calculation uses that broadening
  approximation and a separately declared static prescription.

## Interpreting a calculation

1. Declare the Hamiltonian sign, operator normalization, electronic cell,
   energy unit, and experimental normalization before comparing amplitudes.
   An electronic response per model cell is not automatically per magnetic ion.
2. Converge the integration mesh at fixed temperature and broadening. Then
   vary broadening on a sufficiently dense mesh. These are independent tests;
   a smoother spectrum alone does not demonstrate convergence.
3. Check stability over the full relevant zone, including likely ordering
   vectors. Bare evaluations guard only sampled momenta. In an anisotropic
   field response use the full static local tensor, not a scalar `chi0` test.
4. Distinguish the equilibrium uniform Lindhard susceptibility from the
   dynamic limit of a conserved total spin. Its substituted zero-energy
   value cannot be reconstructed from the finite-energy uniform spectrum.
5. Inspect the physical absorptive tensor, the anti-Hermitian part divided
   by $i$, for complex probes. At finite Lindhard broadening, negative tails
   can be approximation artifacts; they must not be interpreted as gain.
6. Report closure cutoffs and convergence. These closures constrain the
   connected fluctuating moment; a field-induced mean moment is not included
   in the fixed target. A fitted effective onsite anisotropy is not an exact
   quantum single-ion calculation.

The focused scientific checks can be run with:

```bash
python -m pytest -q tests/test_physics_benchmarks.py tests/test_spin_fluctuations.py tests/test_tensor_rpa.py tests/test_sum_rules.py tests/test_closures.py tests/test_electronic_response.py tests/test_electronic_normalization.py
```

Use the configured project environment. GPU tests may be skipped when the
optional backend or device is unavailable; a skipped test is not a numerical
validation of that backend. These checks do not establish exact ordered-state
physics, critical exponents, saturation, or material-specific accuracy.
