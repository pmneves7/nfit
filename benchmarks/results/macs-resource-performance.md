# MACS project responsiveness and numerical timing

Measured on October 6, 2026 with the local nfit conda environment. The input
was the supplied August 2026 MACS `data.nfit`: 78 physical files, 144
spectroscopic (SPEC) and diffraction (DIFF)
stream entries and 2,493,640 detector observations. All timing samples use
`perf_counter` without a profiler. The OS file cache was warm; these are not
cold-storage or cloud-download measurements. The
[receipt](macs-resource-performance.json) records sample times, platform,
resource limits, project hash and output shapes.

The scientific project was read only. Save measurements wrote diagnostic
copies, and source reconstruction changed isolated in-memory entries. Fresh
group binnings used standalone wrappers and the existing default recipe for
each of the ten MACS groups; named variants, root combinations and background-subtracted outputs
are not included in that binning total.

## Resource Manager

The previous inventory used repeated array walks for pairwise sharing checks
and a full remaining-owner walk to calculate each row's reclaimable memory.
Automatic refresh repeated this work on the GUI thread every three seconds.
When an inventory itself took ten seconds, it blocked ordinary interaction.

The replacement walks each live payload once, indexes its storage identities
and looks up sharing owners. This includes private arrays retained by viewers
that close when a resource is unloaded. The index lasts for one scan and does
not retain numerical data between scans. Opening the window no longer takes
two initial inventories. Viewer detachment uses the same storage identities.

Median elapsed times:

| Operation | Before | After |
| --- | ---: | ---: |
| Inventory immediately after opening the saved project | 0.264 s | 0.0095 s |
| Inventory with all source streams loaded | 5.904 s | 0.0185 s |
| Inventory with all streams and ten viewer histogram payloads | 10.039 s | 0.0201 s |
| Construct Resource Manager in that loaded scenario | 10.093 s | 0.0303 s |

The loaded scenario has 174 rows and 281,817,846 distinct numerical bytes.
Before/after row identities, RAM bytes, reclaimable bytes, viewer labels and
total bytes agree. Regression coverage includes shared heap and mapped owners,
compressed caches, private viewer arrays, source protection and explicit unload.
These timings describe inventory and window construction, not every subsequent
operation or every possible model/analysis payload.

Save labels update no more than once per second. Display-only progress bursts
are coalesced before reaching the GUI queue. Cancellation remains checked at
every reported chunk, and explicit progress callbacks retain every checkpoint,
including publication acknowledgements. Fast saves keep the initial label
rather than flashing through array names.

## Reduction, binning and file I/O

| Operation | Median elapsed time |
| --- | ---: |
| Open the existing project, including current lazy cache binding | 1.376 s |
| Save with existing current histogram caches | 0.424 s |
| Reconstruct all 144 streams from the 78 NeXus files | 4.281 s |
| Ten fresh default group binnings after reconstruction, summed medians | 0.719 s |

Reconstruction includes opening, reading and decoding each source and making
its immutable point payload. SPEC and DIFF entries are reconstructed separately,
as in the current importer. Individual fresh group binnings take approximately
0.019–0.163 seconds after one warm-up; receipt samples give their exact shapes
and times. The first profiled group call took 0.339 seconds, including kernel
specialization. Source reads and a cached project open are different operations
and should not be compared as interchangeable reduction timings.

A separate profiler trace identifies HDF5 dataset reads as roughly three
quarters of reconstruction time. For binning, coordinate clustering for
data-driven axes, the second raw-count assignment pass and short-lived thread
pools are more visible than arithmetic in the compiled accumulator.

Potential next improvements are reading common logs once when reconstructing
both streams from one physical file, reusing already-resolved data-driven axis
centers with a complete source/configuration identity, and retaining count
channels during the primary assignment pass. Each requires numerical-equivalence
and allocation checks. They are candidates, not measured speedups. Given the
subsecond group-binning total here, the much larger verified improvement is the
resource inventory fix. No MACS scientific algorithm changed in this audit.
