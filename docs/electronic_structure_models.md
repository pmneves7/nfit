# Electronic-structure and electronic-response models

nfit separates an electronic Hamiltonian from the magnetic response calculated
from it. This keeps the same band model reusable for band plots, density of
states, Fermi surfaces, a bare particle--hole susceptibility, and different
interaction dressings.

The implemented calculation has three layers:

1. `tight_binding` defines an orthonormal one-electron Hamiltonian
   $H(\mathbf k)$.
2. `lindhard` evaluates the complex bare spin susceptibility
   $\boldsymbol\chi^0(\mathbf Q,E)$.
3. An optional `stoner_rpa`, `matrix_rpa`, or `hubbard_hund_rpa` component
   maps $\boldsymbol\chi^0$ to an interaction-dressed susceptibility.

Only the response component is compared with neutron or bulk datasets. The
tight-binding component remains a parameter-providing dependency, so its
onsite, hopping, and spin--orbit coefficients can still be fitted through the
measured response.

## Choosing a construction route

| Starting information | Recommended route | Main limitation |
| --- | --- | --- |
| Crystal structure and a small physically motivated basis | Build the lattice, orbital manifolds, onsite terms, and selected hopping terms in nfit | The chosen orbital basis and hopping range are model assumptions |
| Explicit real-space matrices | Use `build_electronic_model` | The script author is responsible for the basis meaning and matrix convention |
| Wannier90 Hamiltonian | Import `*_hr.dat` or `*_tb.dat` | nfit does not infer missing orbital character or a symmetry representation |

The nfit builder is material independent: it does not prescribe a number of
orbitals, a lattice type, or a specific compound. A site may carry a single
effective orbital, complete spherical-harmonic shells, symmetry-selected
subspaces, or a declared custom basis.

## Guide to this section

- [Tight-binding model](tight_binding.md) defines $H(\mathbf R)$,
  $H(\mathbf k)$, the bands, and the physical scope of the model.
- [Lattice, sites, and orbitals](tight_binding_lattice_orbitals.md) explains
  the structure-first builder and onsite terms.
- [Hoppings](tight_binding_hoppings.md) defines directed hopping matrices,
  spatial bond orbits, Slater--Koster terms, and the general symmetry basis.
- [Spin and spin--orbit coupling](tight_binding_spin.md) explains implicit,
  collinear, and spinor representations.
- [Manual models and external interfaces](tight_binding_imports.md) covers the
  low-level constructor, Wannier90, units, and provenance.
- [Configuration and performance](tight_binding_configuration.md) defines
  every registered `tight_binding` setting and gives practical execution
  guidance.
- [Electronic-structure visualization](electronic_structure_visualization.md)
  covers the geometry, Brillouin-zone, band, DOS, Fermi-surface, and matrix
  viewers.
- [From bands to magnetic response](electronic_response_models.md) follows the
  calculation from a band Hamiltonian to a neutron or bulk prediction and
  routes to the detailed response-model pages.

The [electronic-response design contract](electronic_response_contract.md)
collects invariants for developers and external adapters. Future extensions
are listed only in [Planned features](planned_features.md).

```{toctree}
:maxdepth: 1
:caption: Electronic models

tight_binding
tight_binding_lattice_orbitals
tight_binding_hoppings
tight_binding_spin
tight_binding_imports
tight_binding_configuration
electronic_structure_visualization
electronic_response_models
```
