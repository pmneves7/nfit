# Lindhard acceleration experiments

Measured on 2026-09-28 with the local `nfit` environment on arm64 macOS,
Python 3.14.7, NumPy 2.5.3, SciPy 1.18.0, and Numba 0.67.0.
These are synthetic response-kernel benchmarks, not complete project or fit
runtimes. JSON files retain the workloads, individual timings, and errors.
Imports and JIT startup are excluded. Native BLAS uses one thread; the fastest
exact baseline below uses four Numba workers for transition contraction;
eigensystem evaluation uses NumPy. Timing scripts run separately.

## Spectral FFT: worthwhile for long fixed-Q scans

At each transferred wavevector Q, write the finite-mesh bubble as

\[
\chi^0_{AB}(Q,E)=\sum_t \frac{C_{t,AB}}{\Delta_t-E-i\eta},\qquad
C_{t,AB}=w_k(f_{kn}-f_{k+q,m})M^A_{knm}(M^B_{knm})^*.
\]

Here t labels (k,n,m); w_k is the dimensionless normalized integration weight,
f the dimensionless Fermi occupation, M a dimensionless operator matrix
element, and Delta the final minus initial band energy in meV. E and the
positive broadening eta are in meV, so chi is in inverse meV per cell.
The physical spin prefactor is applied identically on both paths.

`lindhard_spectral_candidate.py` deposits the signed, possibly complex
transition masses C onto an energy grid using linear weights. This conserves
the total mass and first gap moment. Linear FFT convolution with
`-1/(x + i*eta)` evaluates both real and imaginary susceptibility; linear
interpolation returns the requested energies. Histogram entries are masses,
so there is no extra grid-spacing multiplier. Zero padding prevents circular
wraparound. Both gap signs and gaps outside the output window are retained.
The production equal-energy static replacement is added separately at E=0;
it must not be inferred from the dynamic broadened spectrum.

The grid spacing in this table is eta/16. Timings include fresh transition
preparation and FFT evaluation, and are medians of three trials. The reference
is the fastest measured NumPy/Numba-1/Numba-4 implementation for each case,
with completed-response caching disabled on both sides.

| Bands / operators | k mesh | Energies | eta (meV) | Exact | FFT including preparation | Speedup | Peak-normalized complex error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 / 1 | 48 x 48 | 3 | 0.5 | 71.54 ms | 74.23 ms | 0.96x | 0.00064% |
| 6 / 1 | 48 x 48 | 501 | 0.5 | 51.27 ms | 19.04 ms | 2.69x | 0.00980% |
| 12 / 1 | 48 x 48 | 2001 | 0.5 | 622.56 ms | 74.68 ms | 8.34x | 0.00438% |
| 12 / 1 | 32 x 32 | 1001 | 0.05 | 147.60 ms | 34.96 ms | 4.22x | 0.01507% |
| 6 / 3 | 32 x 32 | 501 | 0.5 | 140.21 ms | 16.42 ms | 8.54x | 0.01006% |

Peak-normalized error means `max(abs(candidate-reference))/max(abs(reference))`
over all sampled energies and tensor entries. It is not a pointwise relative
bound near response zeros. Separate real/imaginary errors and results at 4,
8, 16, and 32 bins per eta are in `lindhard-spectral.json`; errors decrease
with refinement in these cases.

For the four long scans, retaining the transition table across eta changes
increased the measured speedup to 14-78x against the exact path with its
normal eigensystem cache. This reuse assumes fixed Hamiltonian, Q, probe
operators, temperature, chemical potential, and mesh. Changing any of those
requires rebuilding this prototype's table. The total preparation results
above are the more conservative comparison.

The retained transition tables occupy 1.9-7.6 MiB for these cases. Their
storage scales as `Nk * Nb**2 * Nop**2` complex numbers, plus gaps and static
terms; this is unsuitable for a large full orbital-pair tensor. FFT grid
size grows with the full gap span divided by eta, so very small broadening
can lose both time and memory advantages. The prototype rejects more than
one million grid bins, but is not a production memory-budget implementation.
The reported process peak RSS (~2.3 GiB) includes every baseline, prototype,
import, and warmup; it cannot attribute peak memory to the FFT alone.

Regression checks compare the independent transition sum with the public
API, test complex/circular probes and extended-zone orbital phases, retain
negative spectral tails, check finite- and zero-temperature static terms,
and check grid convergence and absence of FFT wraparound. Scalar RPA stress
checks use vertices giving static denominator margins 0.1 and 0.01 at the
sampled Q. At eta/16 their peak-normalized dressed errors were below 0.015%
in these examples. These checks do not certify stability throughout the
Brillouin zone or give an error guarantee arbitrarily close to an RPA pole.
A production approximation needs tolerance validation of the dressed result,
refinement/fallback, bounded storage, and full tensor RPA checks.

Decision: retain this isolated prototype and its tests. The measured gains
justify further work for long scans, but it is not enabled in GUI or public
response dispatch, and short requests should stay exact. The present data
does not justify an approximate default for every Lindhard calculation.

## Existing exact Numba path

`lindhard-dispatch.json` measures implicit scalar spin on 32 x 32 meshes with
1, 6, and 12 bands and 1, 3, 101, and 401 energies. Four-worker Numba was about
5.5x faster than one-worker NumPy for the 6- and 12-band 401-energy scans,
and 3.1x for the 12-band 101-energy scan. Maximum absolute complex disagreement
was below 1.5e-15 inverse meV per cell. One-worker Numba and short requests
showed mixed or small gains.

Decision: use the existing explicit `transition_backend="numba", workers=4`
API setting when appropriate (component settings:
`response_transition_backend="numba", response_workers=4`). No new
machine-dependent automatic dispatch heuristic was added. Initial JIT cost
and available CPU allocation can change the best choice.

## Exact shifted-q cache reuse

Parallel q evaluation previously discarded shifted eigensystems after each
call, even when the response cache was enabled. The correction uses the same
canonical eigensystem keys as serial evaluation, batches cold misses, and
stores independently owned read-only q slices in the existing bounded LRU.
A changed Hamiltonian invalidates reuse. Broadening and occupations can
change without rediagonalization while the entries remain cached.

`lindhard-cache.json` uses explicit spinor models, a 32 x 32 mesh, four off-mesh
q values, two workers, a 256 MiB response cache, and three rounds of three
warm calls with distinct broadenings. Both workloads naturally activate q
parallelism; no dispatch threshold is overridden. The comparison emulates
the old path by disabling shifted-cache reuse and the new slice copies while
preserving base-eigensystem reuse. It still runs through the new key lookup
code, rather than checking out the old implementation. Trial order alternates.

| Bands | Energies per q | Former cache behavior | Reuse enabled | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 3 | 1373.8 ms | 919.7 ms | 1.49x |
| 12 | 49 | 300.0 ms | 281.0 ms | 1.07x |

Responses matched bit-for-bit. This is a targeted warm-fit improvement, not
a general acceleration of long energy scans. Cold insertion now copies each
q eigensystem so an evicted q entry cannot keep an entire batch alive outside
its accounted byte size. Cold overhead was not timed separately. The 48-band
case retains about 36.4 MiB per q including eigenvalues; the four shifted
entries and base entry fit within the 256 MiB LRU. Cold working batches and
retained cache storage can coexist, within their separate budgets.

Decision: keep the correction to existing cache behavior. Do not add a new
cache layer, dispatch tuning, or extra machinery to chase the 7% scan gain.
