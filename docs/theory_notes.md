# Theory: sum rules and self-consistency

This page distinguishes exact identities from model closures and maps each
implemented closure onto nfit parameters. The practical goal is to describe a
temperature series with the fewest defensible parameters. Users may fit
$\chi_0(T)$ and $\Gamma_0(T)$ independently or test whether an Onsager, SCR,
or Takahashi closure explains their temperature dependence.

## Bare RPA baseline

Without an active closure, `heisenberg_rpa` (see
[Spin-fluctuation models](spin_fluctuation_models.md)) evaluates

$$
\chi(\mathbf{Q},E)
  = \bigl[\mathbb 1 - \chi_0(E)\,J(\mathbf{Q})\bigr]^{-1}\chi_0(E),
\qquad
\chi_0(E) = \frac{\chi_0}{1 - iE/\Gamma_0}.
$$

This bare path is **linear response around a fixed local propagator** —
RPA/mean-field,
not a self-consistent theory. Nothing the dressed $\chi$ describes ever flows
back into $\chi_0$, $\Gamma_0$, or the couplings. Three consequences:

1. The temperature dependence of $\chi_0(T)$, $\Gamma_0(T)$ is *input* (fitted
   per dataset), not predicted.
2. The instability criterion $\lambda_{\max}(\mathbf{Q})\,\chi_0 \to 1$ is a
   Stoner-like mean-field boundary: real systems suppress it (mode coupling
   depletes the soft mode), so RPA overestimates ordering tendencies, and in
   two dimensions predicts finite-$T$ order that the Mermin–Wagner theorem
   [9] forbids.
3. The total fluctuating moment is unconstrained: as the fit approaches the
   instability, spectral weight piles into the soft mode without being
   depleted elsewhere.

The closures below address these limitations in different ways. The bare model
remains useful: fitted $\chi_0(T)$ and $\Gamma_0(T)$ describe an empirical
renormalized local response, and their trajectories provide a baseline against
which to test a closure.

## Sum rules

### Total-moment (zeroth-moment) sum rule

Via the fluctuation–dissipation theorem the scattering function is

$$
S^{\alpha\beta}(\mathbf{Q},E)
 = \frac{1}{\pi}\,
   \frac{\chi''_{\alpha\beta}(\mathbf{Q},E)}{1 - e^{-E/k_BT}} ,
$$

and integrating the diagonal over the Brillouin zone and *all* energies
(positive and negative, i.e. including the anti-Stokes side) gives an exact
equal-time identity:

$$
\sum_\alpha
\frac{1}{V_{BZ}}\int_{BZ} d\mathbf{Q} \int_{-\infty}^{\infty} dE\,
S^{\alpha\alpha}(\mathbf{Q},E)
 \;=\; \langle \mathbf{S}_i^2 \rangle .
$$

For a **rigid local spin** the right-hand side is the Casimir
$S(S+1)$ — a fixed, temperature-independent number, because the local Hilbert
space is a single spin multiplet. This is the celebrated total-moment sum rule
of magnetic neutron scattering [10]. Plain RPA violates it (point 3 above).

For an **itinerant or valence-fluctuating system** the identity still holds,
but the right-hand side $\langle m^2\rangle(T)$ is a thermodynamic quantity,
not a constraint: the site fluctuates between configurations with different
moments (e.g. $f^1 \leftrightarrow f^0$), so $\mathbf{S}_i^2$ is no longer
proportional to the identity. Neutron practice reads the identity in reverse —
the energy-integrated local response *defines* an effective fluctuating moment
$\mu_{\text{eff}}^2(T)$, which in mixed-valence compounds collapses as the
system cools through the valence-fluctuation/Kondo scale [11, 12].

Two practical caveats for using this quantitatively:

- The experimental window never covers the full $(\mathbf{Q},E)$ range;
  the *model*, integrated over the full zone and up to a cutoff, is the object
  that satisfies (or fails) the sum rule — not the windowed data.
- The relaxational form itself needs a high-energy cutoff: the local
  amplitude integral $\int^\Lambda dE\,\chi''_{\text{loc}}(E)
  \propto \chi_0\Gamma_0 \ln(\Lambda/\Gamma_0)$ diverges logarithmically. The
  cutoff is physics (bandwidth, crystal-field splitting), and any quantitative
  sum-rule implementation must carry it as an explicit scale.

### First-moment (f-) sum rule

$$
\int_{-\infty}^{\infty} dE\; E\,\chi''(\mathbf{Q},E)
 = \pi\,\bigl\langle\,[\,[S_\mathbf{Q},H\,],\,S_{-\mathbf{Q}}\,]\,\bigr\rangle
$$

(Hohenberg–Brinkman [8]). For a Heisenberg $H$ the double commutator evaluates
to exchange constants times equal-time correlators
$\langle S_i\cdot S_j\rangle$, so the rule is exact but again carries unknowns.
Its value is as a *consistency relation*: given a fitted model, both sides are
computable, and a large mismatch flags missing spectral weight (e.g. an
unmodelled high-energy branch).

### Kramers–Kronig / static susceptibility

$$
\chi(\mathbf{Q}, 0) = \frac{2}{\pi}\int_0^\infty
 \frac{\chi''(\mathbf{Q},E)}{E}\, dE .
$$

For the relaxational form this integral closes exactly to $\chi_\mathbf{Q}$ —
i.e. the fitted RPA static susceptibility at the ordering vector is directly
comparable to bulk susceptibility data (after the form-factor and $g$-factor
bookkeeping). This is the cheapest cross-check between an inelastic fit and a
magnetometry curve, and unlike the moment sum rule it converges without a
cutoff.

## The closures: one exact identity + one extra equation

Every self-consistent spin-fluctuation theory in this family has the same
skeleton: keep the RPA-like form of $\chi(\mathbf{Q},E)$, then add one
scalar equation per temperature that determines the local parameters. They
differ in *which* quantity is conserved or fed back.

| closure | conserved / fed-back quantity | moment length | classic reference |
| --- | --- | --- | --- |
| Spherical / Onsager reaction field / SCGA | thermal $\langle m^2\rangle$ fixed to a target | fixed (or targeted) | [1, 2, 3, 4] |
| Moriya SCR | $u\,\langle m^2\rangle(T)$ renormalizes $\chi_0^{-1}$ | soft (amplitude fluctuates) | [5, 6] |
| Takahashi TAC | zero-point **+** thermal amplitude constant | soft, with quantum budget | [7] |
| TPSC | $\int\chi_{\text{sp}} = n - 2\langle n_\uparrow n_\downarrow\rangle$ | set by double occupancy | [13] |
| Anderson-lattice phenomenology | $\langle m^2\rangle \sim n_f(T)\,\mu_{CF}^2$; $\Gamma_0 \sim$ charge-fluctuation rate | valence-dependent | [11, 12] |

### Spherical model / Onsager reaction field

Replace the rigid per-site constraint by its average: demand only that the
*mean* fluctuation amplitude equal the target,
$\langle m^2\rangle = m_0^2$. Operationally this is a $\mathbf{Q}$-independent
shift of the exchange,

$$
J(\mathbf{Q}) \;\to\; J(\mathbf{Q}) - \lambda(T),
$$

with $\lambda(T)$ (the Onsager reaction field / spherical Lagrange multiplier)
fixed per temperature by the sum rule. Berlin–Kac solved the classical version
exactly [1]; Brout–Thomas connected it to Onsager's reaction-field idea [2];
its modern incarnation is the self-consistent Gaussian approximation (SCGA)
and large-$N$ treatments that describe frustrated paramagnets (including
pyrochlores) remarkably well [3, 4]. The divergence at
$\lambda_{\max}\chi_0 \to 1$ is converted into a smooth approach — the sum
rule *cannot* be satisfied with a diverging soft mode, so $\lambda$ backs the
system away from it, suppressing $T_c$ toward (in 2D exactly to) the
Mermin–Wagner answer.

**Implementation.** The shift $-\lambda\,\mathbb 1$ has the matrix shape of an
isotropic on-site term (the trace part omitted by the SIA projection). The
closure uses an outer scalar root-find per temperature and a Brillouin-zone
quadrature with an energy cutoff to evaluate
$\langle m^2\rangle[\lambda]$ — that integral is over the *full zone*, not the
measured window, so it needs its own $\mathbf{Q}$-grid independent of the
data.

### Moriya's self-consistent renormalization (SCR)

Start from a soft-spin (Ginzburg–Landau) functional
$\delta\,m^2 + u\,m^4 + \dots$ instead of a length constraint. Mode–mode
coupling feeds the fluctuation amplitude back into the Gaussian mass:

$$
\chi_0^{-1}(T) = \chi_0^{-1}(0) + u\,\langle m^2\rangle(T),
\qquad
\langle m^2\rangle(T)
 = \frac{1}{V_{BZ}}\int_{BZ} d\mathbf{Q}\int dE\,
   \coth\!\Bigl(\frac{E}{2k_BT}\Bigr)\,
   \frac{\chi''(\mathbf{Q},E)}{\pi},
$$

solved self-consistently since $\chi''$ itself depends on $\chi_0$. This is
the theory that *derives* the Curie–Weiss law of weak itinerant magnets — the
linear-in-$T$ inverse susceptibility comes from the thermal growth of
$\langle m^2\rangle$, not from a local moment — and gives the canonical
$T$-dependences of $\Gamma$ near ferro- and antiferromagnetic instabilities
[5]. Lonzarich–Taillefer [6] recast it as a phenomenology with a handful of
measurable parameters, which is precisely the form most useful for fitting.
Amplitude (longitudinal) fluctuations are part of the spectrum here — the
moment is a fluctuating field, not a fixed-length vector.

### Takahashi's total-amplitude conservation (TAC)

Takahashi's refinement [7]: what is conserved is the **zero-point plus
thermal** amplitude,

$$
\langle m^2\rangle_{\text{ZP}}(T) + \langle m^2\rangle_{T}(T) = \text{const},
$$

not the thermal part alone. Because the zero-point part is itself an integral
over the fitted spectrum, this one condition ties together the ground state
and the finite-$T$ response, and reproduces the Rhodes–Wohlfarth systematics
of weak itinerant ferromagnets and the $T^{3/2}$–$T^2$ crossovers with few
parameters.

### TPSC (Vilk–Tremblay)

For Hubbard-like models the exact identity
$\int \chi_{\text{sp}} = n - 2\langle n_\uparrow n_\downarrow\rangle$ ties the
spin sum rule to the double occupancy, itself determined self-consistently
alongside the matching charge sum rule [13]. This is the cleanest example of a
sum rule *for a non-fixed spin length*: charge (valence) fluctuations reduce
the moment budget through $\langle n_\uparrow n_\downarrow\rangle$, and the
effective vertex adjusts so both channels are satisfied at once. Included here
as a reference point; a TPSC-level implementation is out of scope for the
current phenomenological family.

### Valence fluctuations / Anderson lattice

In a mixed-valence material the local moment budget scales with the
$f$-occupancy, $\langle m^2\rangle \sim n_f(T)\,\mu_{CF}^2$ (modulo
crystal-field projection), and the quasielastic width $\Gamma_0$ *is* the
valence-fluctuation/Kondo rate [11, 12]. A fixed-moment sum rule is therefore
generally inappropriate. The useful diagnostics are the fitted
$\mu_{\text{eff}}^2(T)$, obtained from the zone- and energy-integrated
$\chi''$, and its correlation with $\Gamma_0(T)$. Shrinking amplitude together
with a saturating width is the characteristic valence-fluctuation pattern.

## Dictionary: closures ↔ nfit parameters

| theory object | unconstrained fit | closure result or control |
| --- | --- | --- |
| local static susceptibility $\chi_{\text{loc}}(T)$ | fitted `chi0` per dataset | derived from $\chi_0^{-1}(0) + u\langle m^2\rangle(T)$ (SCR) or from $\lambda(T)$ (spherical) |
| local relaxation rate $\Gamma(T)$ | fitted `gamma0` per dataset | fitted globally or per dataset; not presently fixed by the closures |
| Onsager shift $\lambda(T)$ | absent | solved internally per temperature and reported in diagnostics |
| moment budget $\langle m^2\rangle$ | diagnostic integral | fixed `moment_target` or fitted `m2_total`/`total_amplitude` |
| high-energy cutoff $\Lambda$ | diagnostic choice | global `energy_cutoff_mev`, shared across temperatures |
| static $\chi(\mathbf{Q}_{\text{pk}}, T)$ | Kramers–Kronig of the fit | cross-check against bulk magnetometry |

Useful identities when relating fitted values across temperatures:

- **Classical regime** ($k_BT \gg \Gamma_0$): equipartition gives
  $\langle m^2\rangle_T \approx k_BT\,\chi_{\text{loc}}(T)$ per component —
  so a Curie-like $\chi_0 \propto 1/T$ corresponds to a *constant* thermal
  amplitude, and deviations measure amplitude growth or collapse.
- **Quantum regime** ($k_BT \ll \Gamma_0$): the amplitude is dominated by
  zero-point weight $\propto \chi_0\Gamma_0\ln(\Lambda/\Gamma_0)$ — this is
  the piece Takahashi's TAC budget trades against the thermal part.

## Choosing a temperature-dependent model

The options below increase in theoretical commitment. Compare each closure
against the unconstrained baseline using fit quality, parameter count, and the
diagnostic trajectories.

1. **Unconstrained.** `chi0`, `gamma0` per dataset
   (`per_dataset` sharing); exchanges and scale global. Parameter count
   $\sim 2N_T + N_J + 1$. This provides the comparison baseline.
2. **Diagnostics.** Every fit records per-dataset
   $\mu_{\text{eff}}^2(T)$ (zone/energy integral with cutoff $\Lambda$),
   Kramers–Kronig $\chi(\mathbf{Q}=0,T)$ and $\chi(\mathbf{Q}_{\text{pk}},T)$,
   the $\chi_0\Gamma_0$ product, and the distance to instability
   $1-\lambda_{\max}\chi_0$ into the fit result (shown in the *Physics
   diagnostics* table, persisted for plotting vs $T$). The *shapes* of these
   trajectories select the closure: flat $\mu_{\text{eff}}^2$ →
   spherical/local-moment; linear $\chi_0^{-1}(T)$ with growing amplitude →
   SCR; collapsing amplitude with saturating $\Gamma_0$ → valence fluctuations.
3. **Spherical/Onsager closure.** Replace $N_T$ values of `chi0`
   with one moment target $m_0^2$ (fixed `moment_target`, or fitted
   `m2_total`) and the cutoff $\Lambda$: $\lambda(T)$ solved internally. Exact
   in form (a $\mathbf{Q}$-independent eigenvalue shift), needs only an outer
   scalar root-find, and directly cures the RPA sum-rule violation.
4. **SCR/TAC dynamic closure.** SCR renormalizes `chi0` through
   the mode coupling `mode_coupling_u` (with `chi0` reinterpreted as the $T=0$
   bare value); TAC conserves the total zero-point + thermal amplitude
   (`total_amplitude`). Both reduce a temperature series to a handful of global
   parameters, so the temperature dependence becomes a *prediction* checked
   against the data rather than fitted per point. (`gamma0` remains a directly
   fitted, usually global, parameter; a fully derived $\Gamma_0(T)$ à la
   Lonzarich–Taillefer is not implemented.)
5. **Interplay with tensor terms.** Anisotropy gaps, Zeeman
   fields, and dipolar terms all shift where the sum-rule weight sits (a gap
   moves weight up in energy; a field splits it between Larmor channels). The
   closures act on the isotropic local propagator and inherit these effects
   through the full $\chi''$ entering $\langle m^2\rangle$: Tier A (field off)
   integrates the mode Lorentzians in closed form; Tier B (field on) integrates
   the gyrotropic $\chi''$ numerically on a $(\mathbf{Q},E)$ grid.
6. **Bulk-susceptibility co-fit.** The uniform static
   susceptibility $\chi(\mathbf{Q}=0,0)$ — with the active closure — predicts
   the bulk moment $M(T,B)$, and MPMS magnetization datasets are fit jointly
   with the inelastic data sharing $J$/`chi0`/`gamma0`. See the closure and
   bulk-$\chi$ sections of
   [spin_fluctuation_models](spin_fluctuation_models.md).

## Implementation notes

The closures (`nfit.closures`) wrap the existing evaluator with no kernel
change. The moment integrals (`nfit.sum_rules`) run on a standalone
$N^3$ Monkhorst–Pack Brillouin-zone grid (`config["closure"]["bz_grid"]`,
midpoint-shifted so no point lands on $\Gamma$/zone boundary), independent of
the measured points and using the same primitive-reduced site network as the
evaluator. Each closure adds one derived report per temperature ($\lambda(T)$,
$\chi_{0,\text{eff}}(T)$, $\langle m^2\rangle$) surfaced in the diagnostics.

**Energy integrals.** The per-mode moment
$(1/\pi)\int_0^\Lambda \coth(E/2k_BT)\,\chi''_{\text{Lor}}(E)\,dE$
has two exact closed forms (both locked to adaptive quadrature): for
$\Lambda \gg k_BT$ the digamma form (zero-point
$(\chi\Gamma/2\pi)\ln(1+\Lambda^2/\Gamma^2)$ plus infinite-cutoff thermal part
$(\chi\Gamma/\pi)[\ln z - 1/2z - \psi(z)]$, $z=\Gamma/2\pi k_BT$); otherwise a
finite-cutoff Matsubara sum completed with a Hurwitz-zeta tail. Tier B uses a
log-spaced energy quadrature of $\mathrm{Tr}\,\chi''$.

**Bulk-$\chi$ units.** The model susceptibility is $\chi_{\text{spin}}$ in
1/meV per magnetic site; the molar magnetic susceptibility is
$\chi_{\text{mol}}[\mathrm{emu/mol}] = N_A(g\mu_B)^2\chi_{\text{spin}}
= C\,g^2\,\chi_{\text{model}}$ per site with
$C = N_A\mu_{B,\text{cgs}}^2/(\mathrm{meV\ in\ erg}) \approx 0.0323$
emu·meV/mol (`nfit.sum_rules.EMU_PER_MOL_PER_MODEL_CHI`). By default a free
per-dataset scale absorbs this constant and the sample amount; the optional
absolute mode pins it from the sample mass and molar mass and fits in emu.

**Cost.** Tier A amortizes one BZ-grid eigendecomposition across every solver
probe (the Onsager $\lambda$ is a rigid eigenvalue shift). Dense Tier-A
Onsager, SCR, and TAC temperature sweeps use continuation from neighboring
temperatures, a shared fused Numba finite-cutoff Matsubara kernel, and exact
closure-result caching. SCR's `u = 0` limit bypasses root finding entirely.
TAC's stored `chi0` is only a root-search seed, so it is excluded from the
cache key and cannot be varied by the optimizer; the conserved total amplitude
replaces it physically. Tier B has no such shortcut: each closure iteration is
a batched LU over the $(\text{BZ}\times E)$ grid per distinct $(T,B)$. The
default $16^3\times200$ grid requires approximately $8\times10^5$ solves per
iteration. Smaller exploratory grids require convergence checks before
quantitative use. Solves are cached per $(T,B,\text{params})$.

## References

1. T. H. Berlin and M. Kac, "The spherical model of a ferromagnet",
   Phys. Rev. **86**, 821 (1952). <https://doi.org/10.1103/PhysRev.86.821>
2. R. Brout and H. Thomas, "Molecular field theory, the Onsager reaction
   field and the spherical model", Physics Physique Fizika **3**, 317 (1967).
   <https://doi.org/10.1103/PhysicsPhysiqueFizika.3.317>
3. D. A. Garanin and B. Canals, "Classical spin liquid: Exact solution for
   the infinite-component antiferromagnet on the kagomé lattice",
   Phys. Rev. B **59**, 443 (1999).
   <https://doi.org/10.1103/PhysRevB.59.443>
4. P. H. Conlon and J. T. Chalker, "Absent pinch points and emergent clusters:
   Further neighbor interactions in the pyrochlore Heisenberg
   antiferromagnet", Phys. Rev. B **81**, 224413 (2010).
   <https://doi.org/10.1103/PhysRevB.81.224413>
5. T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*, Springer
   Series in Solid-State Sciences 56 (Springer, Berlin, 1985).
6. G. G. Lonzarich and L. Taillefer, "Effect of spin fluctuations on the
   magnetic equation of state of ferromagnetic or nearly ferromagnetic
   metals", J. Phys. C: Solid State Phys. **18**, 4339 (1985).
   <https://doi.org/10.1088/0022-3719/18/22/017>
7. Y. Takahashi, "On the origin of the Curie–Weiss law of the magnetic
   susceptibility in itinerant electron ferromagnetism",
   J. Phys. Soc. Jpn. **55**, 3553 (1986); *Spin Fluctuation Theory of
   Itinerant Electron Magnetism* (Springer, Berlin, 2013).
   <https://doi.org/10.1143/JPSJ.55.3553>
8. P. C. Hohenberg and W. F. Brinkman, "Sum rules for the frequency spectrum
   of linear magnetic chains", Phys. Rev. B **10**, 128 (1974).
   <https://doi.org/10.1103/PhysRevB.10.128>
9. N. D. Mermin and H. Wagner, "Absence of ferromagnetism or
   antiferromagnetism in one- or two-dimensional isotropic Heisenberg
   models", Phys. Rev. Lett. **17**, 1133 (1966).
   <https://doi.org/10.1103/PhysRevLett.17.1133>
10. I. A. Zaliznyak and S.-H. Lee, "Magnetic neutron scattering", in *Modern
    Techniques for Characterizing Magnetic Materials*, edited by Y. Zhu
    (Springer, Boston, 2005) — total-moment sum rule in neutron practice.
11. J. M. Lawrence, P. S. Riseborough, and R. D. Parks, "Valence fluctuation
    phenomena", Rep. Prog. Phys. **44**, 1 (1981).
    <https://doi.org/10.1088/0034-4885/44/1/001>
12. E. Holland-Moritz, D. Wohlleben, and M. Loewenhaupt, "Neutron
    spectroscopy of intermediate-valence compounds", Phys. Rev. B **25**,
    7482 (1982). <https://doi.org/10.1103/PhysRevB.25.7482>
13. Y. M. Vilk and A.-M. S. Tremblay, "Non-perturbative many-body approach to
    the Hubbard model and single-particle pseudogap",
    J. Phys. I France **7**, 1309 (1997).
    <https://doi.org/10.1051/jp1:1997135>
