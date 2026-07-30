# Manual models and external interfaces

nfit supports two complete Hamiltonian inputs: explicit real-space matrices
and Wannier90 files. CIF import is a geometry input for the structure-first
builder; a CIF alone does not contain an electronic Hamiltonian.

## Electronic energy units

Electronic-structure literature and Wannier90 normally use eV, while neutron
energy transfer is normally expressed in meV. nfit uses a single explicit
boundary:

- user-facing electronic inputs and plots default to eV;
- `ElectronicModel` and numerical response kernels store energies in meV; and
- low-level argument names ending in `_meV` always mean meV.

Conversion occurs once when a model is built or imported and is recorded in
provenance. nfit never guesses a unit from numerical magnitude.

The GUI **Electronic energy unit** changes electronic entry fields and plot
labels without changing stored physical values. A script should pass
`energy_unit` explicitly when the source convention might be ambiguous.

## Direct matrix construction

`build_electronic_model` accepts arbitrary complex onsite and hopping
matrices.

| Argument | Meaning | Example |
| --- | --- | --- |
| `direct_lattice` | direct-lattice vectors as columns, in Å | `np.diag([4.0,4.0,12.0])` |
| `basis` | ordered `BasisState`, dictionary, or label sequence | `[BasisState("M1_d"), BasisState("X1_p")]` |
| `hoppings` | map $\mathbf R\mapsto H(\mathbf R)$ in `energy_unit` | `{(0,0,0): onsite, (1,0,0): hop}` |
| `orbital_centers` | fractional center of each basis state | `[[0,0,0],[0.5,0.5,0]]` |
| `periodic_axes` | one to three periodic direct-lattice axes | `(0,1)` |
| `interpolation_weights` | optional map $\mathbf R\mapsto w_{\mathbf R}$ | `{(1,0,0): 0.5}` |
| `parameter_values` | named energy coefficients in `energy_unit` | `{"t_nn": -0.08}` |
| `parameter_hoppings` | dimensionless $\mathbf R\mapsto P_p(\mathbf R)$ maps | `{"t_nn": {(1,0,0): [[1.0]]}}` |
| `spin_operators` | optional Hermitian $S_x,S_y,S_z$ | array of shape `(3,N,N)` |
| `energy_unit` | unit of hoppings, parameter values, and energy zero | `"eV"` or `"meV"` |
| `energy_zero` | recorded reference energy in `energy_unit` | `0.0` |
| `add_hermitian_conjugates` | generate a missing $-\mathbf R$ partner | `true` |
| `provenance` | JSON-compatible source metadata | `{"source":"custom-script"}` |

`parameter_values` and `parameter_hoppings` must have identical keys. A
parameter matrix is dimensionless under the recommended convention, so the
coefficient carries the energy unit.

```python
import numpy as np

from nfit import BasisState, build_electronic_model

model = build_electronic_model(
    direct_lattice=np.diag([4.0, 4.0, 12.0]),
    basis=[BasisState("M1_s", site="M1", orbital="s")],
    hoppings={
        (0, 0, 0): [[0.0]],
        (1, 0, 0): [[-0.08]],
    },
    orbital_centers=[[0.0, 0.0, 0.0]],
    periodic_axes=(0, 1),
    energy_unit="eV",
)
```

When `add_hermitian_conjugates=True`, the example receives the missing
$(-1,0,0)$ block automatically. An explicitly supplied inconsistent partner
or a non-Hermitian $\mathbf R=0$ block raises an error.

## Wannier90

`import_wannier90` reads `seedname_hr.dat` and `seedname_tb.dat` without
running Wannier90.

| Argument | Meaning | Example |
| --- | --- | --- |
| `path` | Hamiltonian file | `"seedname_hr.dat"` or `"seedname_tb.dat"` |
| `direct_lattice` | lattice override when an `hr.dat` has no associated `.win` | `np.diag([4.0,4.0,4.0])` |
| `orbital_centers` | fractional centers when an `hr.dat` has no associated centers file | `[[0,0,0],[0.5,0.5,0.5]]` |
| `basis` | optional basis metadata replacing `w1`, `w2`, ... | `[BasisState("M1_d_xy")]` |
| `periodic_axes` | optional periodic directions | `(0,1,2)` |
| `wsvec_path` | optional pair-dependent replica file | `"seedname_wsvec.dat"` |

A `tb.dat` contains the direct lattice and position matrix. An `hr.dat` does
not; nfit searches for `seedname.win` and `seedname_centres.xyz` unless the
lattice and centers are supplied explicitly. If present,
`seedname_wsvec.dat` supplies pair-dependent Wigner--Seitz replicas.

The importer:

- converts the documented Wannier90 eV energies to meV;
- applies Wigner--Seitz degeneracies and optional replica corrections;
- stores the result in the Wannier Fourier gauge;
- records paths, file sizes, SHA-256 digests, format, conversion, and importer
  version; and
- validates Hermiticity rather than repairing the data.

Wannier functions are general numerical basis states. Unless `basis` metadata
is supplied, nfit labels them only as `w1`, `w2`, and so on. It does not infer
atomic orbitals, local frames, correlated shells, or symmetry
representations from the Hamiltonian file. A Wannier90 model therefore uses
the imported matrices directly rather than converting automatically into the
symmetry-aware builder representation.

SCDM-generated Wannier functions are supported through the same Wannier90
files; nfit does not implement the SCDM localization algorithm.

## Portability and integrity

`ElectronicModel.to_dict()` is a portable canonical representation.
`save_electronic_model` and `load_electronic_model` write and validate it as
JSON. The model digest covers the resolved lattice, basis, matrices,
parameters, spin operators, and conventions.

A component may store either:

- `source_path` plus the expected `model_digest`, causing the original
  Wannier90 source to be reloaded and checked; or
- embedded `model_data`, used for manual and structure-built models.

Copied scripts reproduce the same distinction. Source-backed scripts reload
and verify the original files; portable manual scripts reconstruct the stored
scientific state without Qt.

## Other electronic-structure packages

nfit uses ASE for Setyawan--Curtarolo paths and linear-tetrahedron DOS, but
does not currently import ASE calculator or `Atoms` electronic models.
Pymatgen, PythTB, and sisl model adapters are also not implemented. These
packages remain useful references and can be used in a script to prepare
explicit matrices, but the conversion must state its lattice, basis order,
energy unit, Fourier gauge, and orbital centers. Wannier90 is the only
implemented external Hamiltonian interface.

## References

- [Wannier90 file-format documentation](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/).
- V. Vitale *et al.*, *npj Comput. Mater.* **6**, 66 (2020),
  [doi:10.1038/s41524-020-0312-y](https://doi.org/10.1038/s41524-020-0312-y).
- [ASE unit conventions](https://docs.ase-lib.org/ase/units.html).
- [pymatgen electronic-structure API](https://pymatgen.org/pymatgen.electronic_structure.html).
- [PythTB hopping API](https://pythtb.readthedocs.io/en/latest/generated/pythtb/TBModel/pythtb.TBModel.set_hop.html).
