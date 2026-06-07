# Physics Notes: Itinerant Magnetic Fluctuations, SCR, Paramagnons, and QFI

## 1. Context

The target systems are nearly magnetic, often frustrated metals. Unlike localized-spin insulators, the spin amplitude is not fixed. In metals, the relevant fluctuating variable is often a spin-density field rather than a rigid local spin. Longitudinal amplitude fluctuations, damping into particle-hole excitations, and broad spectral weight are central.

Bulk susceptibility can look nearly Pauli-like even when neutron scattering sees strong finite-Q magnetic fluctuations. This is because bulk magnetometry measures the uniform static response chi(q=0, omega=0), while frustrated or nearly antiferromagnetic metals may have their dominant spectral weight at finite q, along broad ridges, rings, shells, or several competing Q positions.

## 2. Bulk susceptibility expectations

A useful phenomenological decomposition is:

chi_bulk(T) = chi_core + chi_orb + chi_Pauli*(T) + delta_chi_SF(T) + C_imp/(T - theta_imp).

The Pauli-like term may be exchange enhanced. The impurity/defect tail can dominate low-T magnetometry. The finite-Q magnetic correlations measured by neutron scattering need not produce a strong Curie-Weiss-like bulk susceptibility.

Expected cases:

- Nearly ferromagnetic metal: chi_bulk(T) can be strongly enhanced and Curie-Weiss-like over some range.
- Nearly antiferromagnetic metal: chi(Q_AF,T) can be strong while chi_bulk(T) remains weakly T-dependent.
- Frustrated nearly magnetic metal: chi(q,T) can be broad in momentum; chi_bulk(T) may show a weak maximum, crossover, or nearly Pauli-like behavior.

## 3. Moriya SCR picture

Moriya self-consistent renormalization theory treats spin fluctuations in itinerant magnets beyond simple Stoner/RPA mean field. The schematic RPA susceptibility is:

chi(q,omega) = chi0(q,omega) / [1 - I chi0(q,omega)].

SCR adds the idea that spin fluctuations renormalize the distance to magnetic order. Schematically:

chi_Q^{-1}(T) = chi_Q^{-1}(0) + u <m^2>_T,

where

<m^2>_T ~ sum_q integral d omega coth(hbar omega / 2 kB T) chi''(q,omega).

The same spectrum measured by neutrons contributes to the renormalization of the susceptibility. This makes the theory self-consistent.

A common nearly antiferromagnetic overdamped form is:

chi^{-1}(Q + q, omega) = r(T) + A q^2 - i C omega.

Then:

chi(Q + q, 0) = 1 / [r(T) + A q^2],

and the relaxation rate is approximately:

Gamma_q(T) = [r(T) + A q^2] / C.

As r(T) softens, chi increases and Gamma decreases near the incipient ordering wave vector: critical slowing down.

## 4. Takahashi spin-fluctuation theory

Takahashi's extension emphasizes spin-amplitude conservation including zero-point spin fluctuations. In weak itinerant magnets, the ordered moment can be small while the total fluctuating moment is large. A schematic constraint is:

M0^2 + <m^2>_thermal + <m^2>_zero-point ~ constrained total spin amplitude.

For neutron scattering this motivates absolute normalization and integration of spectral weight, not just low-energy peak fitting. The zero-temperature spectral weight can remain finite because quantum spin-density fluctuations persist.

## 5. Low-energy paramagnon expansion

A common phenomenological form for fluctuations near one or more soft wave vectors Q_delta is:

chi(Q,omega) = sum_delta chi_delta [1 + |Q - Q_delta|^2/kappa0^2 - i omega/omega_SF]^{-1}.

The imaginary part is:

chi''(Q,omega) = sum_delta chi_delta * (omega/omega_SF) /
{ [1 + |Q - Q_delta|^2/kappa0^2]^2 + (omega/omega_SF)^2 }.

Define:

a(Q) = 1 + |Q - Q_delta|^2/kappa0^2.

Then:

chi(Q,0) = chi_delta / a(Q),
Gamma_Q = a(Q) omega_SF.

At the soft wave vector Q_delta, Gamma = omega_SF. Away from Q_delta, the fluctuations broaden and become faster.

This low-energy form is related to Moriya/SCR in spirit and often serves as the dynamical susceptibility inserted into SCR-like fluctuation integrals. By itself it is not full SCR unless the mass/width parameters are determined self-consistently from the fluctuation spectrum.

## 6. Symmetry-constrained J(Q) or Lambda(Q) expansions

For broad, complex momentum dependence, fit the inverse susceptibility with symmetry-allowed lattice harmonics:

chi^{-1}(q,omega) = r(T) + Lambda(q) - i C(q) omega,

where

Lambda(q) = sum_n K_n gamma_n(q).

The gamma_n(q) can be Fourier transforms of neighbor shells, equivalent to a Heisenberg-like J(q) expansion. In itinerant systems, the coefficients K_n should be interpreted phenomenologically as effective inverse-susceptibility harmonics, not literal microscopic exchange constants unless separately justified.

A useful static form is:

chi(q) = chi0 / [r(T) + Lambda(q)].

A useful relaxational dynamical form is:

chi''(q,E) = chi(q) * [E Gamma_q] / [E^2 + Gamma_q^2],

with E = hbar omega. Another equivalent convention uses:

chi^{-1}(q,E) = r(T) + Lambda(q) - i E / gamma,

which implies:

Gamma_q = gamma [r(T) + Lambda(q)].

Be explicit about the convention used in code and documentation.

## 7. Local susceptibility and moment sum rules

Neutron scattering measures S(Q,E), related to chi''(Q,E) by a fluctuation-dissipation factor:

S(Q,E) = [1/pi] * chi''(Q,E) / [1 - exp(-E/kB T)]

up to convention-dependent constants and unit choices.

The local susceptibility is a momentum integral:

chi''_local(E) = integral_BZ d^3Q chi''(Q,E).

The fluctuating moment estimate is based on an integral over S(Q,E):

<m^2> ~ integral_BZ d^3Q integral dE S(Q,E),

with magnetic form factor, polarization, absolute-unit normalization, and finite measurement-window corrections handled carefully.

In itinerant metals, finite experimental energy windows may miss high-energy Stoner continuum weight. Observed spectral weight is often a lower bound on the total fluctuating moment.

## 8. Quantum Fisher information from neutron data

For a spin-density operator O_q, QFI can be related to the dynamical susceptibility:

F_Q(q,T) = (4/pi) integral_0^infty dE tanh[E/(2 kB T)] chi''(q,E),

up to operator normalization conventions.

The tanh factor suppresses classical low-energy thermal fluctuations and emphasizes quantum fluctuations. This differs strongly from equal-time moment integrals, which involve Bose/coth-like thermal weighting.

For itinerant systems, entanglement-depth normalization is subtle because the local Hilbert space and effective spin length are not uniquely defined. Sensible reporting options:

1. Absolute QFI density in physical units.
2. Conservative ionic/electron-count normalization using S_max.
3. Effective-moment normalization using measured or estimated fluctuating moment.
4. DFT/DMFT/Wannier-based orbital normalization, if available.
5. A range of normalizations with clear caveats.

A measured-window QFI is a lower bound on the numerator if spectral weight is missing. However, using a measured-window fluctuating moment as the denominator can artificially inflate normalized QFI if unobserved moment exists. Be explicit about which conclusions are rigorous and which are model-dependent.

## 9. SCR and Onsager reaction-field analogy

SCR is conceptually related to Onsager reaction-field ideas in that both correct naive mean field using self-consistent fluctuation effects. Onsager subtracts or compensates for the self-field/reaction field in local-moment systems. Moriya SCR renormalizes an itinerant dynamical susceptibility using the spin fluctuations generated by that susceptibility.

They are not the same formalism. SCR is dynamical, quantum, and itinerant; Onsager reaction-field theory is often static and local-moment based. But the philosophy is similar: improve mean-field/RPA by accounting for the feedback of fluctuations.

