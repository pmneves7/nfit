# Large MDHisto rebin crossover

Fresh processes on `analysis-node23.sns.gov` warmed the selected path before
timing. Every JSON records the source digest, forced threshold, effective
worker count, input payload, peak RSS, and result statistics. The 10 GB pairs
also have saved-artifact comparisons: all arrays and channels pass
`allclose(rtol=1e-12, atol=1e-12, equal_nan=True)`. Raw floating-point digests
can differ because the dense and streamed reductions accumulate in a different
order.

For regular sparse inputs, dense versus stream medians were:

| Numerical payload | Dense | Stream |
| ---: | ---: | ---: |
| 0.105 MB | 4.788 ms | 5.306 ms |
| 1.343 MB | 7.034 ms | 7.637 ms |
| 5.374 MB | 18.097 ms | 11.968 ms |
| 10.748 MB | 36.738 ms | 20.168 ms |
| 86.0 MB | 246.397 ms | 160.076 ms |
| 10.318 GB | 29.033 s | 18.596 s |

At 10.318 GB, streaming also reduced peak RSS from 45.926 GB to 12.709 GB for
the sparse regular case, and from 58.105 GB to 12.748 GB for the 70%-valid
regular case. The fractional case was 82.431 s dense and 21.208 s streamed;
the symmetry case was 161.396 s dense and 122.187 s streamed.

Keep the existing 250,000-source-bin threshold. A three-repeat confirmation at
131,072 bins (5.374 MB) found streaming substantially faster for fractional
rebinning and for dense symmetry rebinning, but sparse symmetry was effectively
tied and slightly favored dense (68.869 ms versus 69.217 ms). The lower
threshold is therefore not a consistent win across representative workloads.
