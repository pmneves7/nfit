# Electronic-structure models

nfit's electronic-structure layer supplies material-independent Hamiltonians
for band calculations and itinerant magnetic-response models. These
models describe the one-electron basis and propagator; they are not themselves
spin-fluctuation susceptibilities.

| Model | Main equation | Typical use | Calculable data |
| --- | --- | --- | --- |
| `tight_binding` | $H(\mathbf k)=\sum_{\mathbf R}w_{\mathbf R}H(\mathbf R)e^{2\pi i\mathbf k\cdot\mathbf R}$ | Manual, symmetry-built, or Wannier90 electronic structure | Bands, orbital projections, density of states, Fermi surfaces, and model geometry |

The [bare Lindhard model](lindhard.md) consumes a tight-binding component
without merging the electronic Hamiltonian into the response model. The
[electronic-response design contract](electronic_response_contract.md)
defines this implemented boundary and the later interaction dressings,
superconducting extensions, and more advanced correlated-electron layers.
Once a model produces
$\chi(\mathbf Q,E)$, its magnetic-response formalism is documented under
[Spin-fluctuation models](spin_fluctuation_models.md).

```{toctree}
:maxdepth: 1
:caption: Model details

tight_binding
```
