# PA2 Report --- Errors and Corrections

## Purpose

This document lists the errors, inconsistencies, and technically risky
statements identified in the PA2 report. Items are grouped by severity.

------------------------------------------------------------------------

# 1. Definite errors / inconsistencies

## 1.1 Communication is said to dominate at 8 ranks for all datasets

### Problem

The report states:

> "Communication begins to exceed computation at 8 ranks across all
> datasets."

This is not supported by the report's own profiling table.

For Version 5 at 8 ranks:

-   Small: Computation 39.3%, Communication 60.7% → communication
    dominates.
-   Medium: Computation 55.7%, Communication 44.3% → computation
    dominates.
-   Large: Computation 28.0%, Communication 72.0% → communication
    dominates.

Therefore, the statement "across all datasets" is incorrect.

### Correction

Use:

> "At 8 ranks, communication becomes comparable to or larger than
> computation, particularly for the Small and Large datasets in Version
> 5 and for the Medium and Large datasets in Version 6. The Medium V5
> configuration remains computation-dominated at 8 ranks."

------------------------------------------------------------------------

## 1.2 MPI profiling percentages do not always add up to total time

### Problem

Several rows have:

`Computation time + Communication time != Total time`

Example:

-   V5 Small, 2 ranks:
    -   Total = 3.37 ms
    -   Computation = 2.49 ms
    -   Communication = 0.46 ms
    -   Sum = 2.95 ms
    -   Unaccounted = 0.42 ms

Another example:

-   V6 Small, 8 ranks:
    -   Total = 4.29 ms
    -   Computation = 1.25 ms
    -   Communication = 2.10 ms
    -   Sum = 3.35 ms
    -   Unaccounted = 0.94 ms

### Why this matters

The report currently makes the percentages look as if computation +
communication completely explain the total execution time. They do not.

### Correction

Add an **Other / Synchronization / Runtime Overhead** column, or
explicitly state that the measured computation and communication phases
do not account for all wall-clock time.

Possible wording:

> "The remaining time represents synchronization, MPI/runtime overhead,
> setup, and other phases not separately instrumented."

Do not assume the exact source of the remainder unless it was measured.

------------------------------------------------------------------------

## 1.3 CUDA results are inconsistent across sections

### Problem

Different parts of the report use different CUDA total execution times
and speedups.

Examples include:

-   CUDA Large total around **15.87--18.80 ms**
-   Another section gives **15.87--17.40 ms**
-   Another result uses **16.62 ms**
-   Kernel speedup is reported as **492×** in one place
-   Part C reports approximately **406.3--440.4×** kernel speedup
-   End-to-end speedup is reported as approximately **36.2--39.7×**

These values may come from different runs, but the report does not
clearly explain that.

### Correction

Choose one canonical benchmark result for Part C and use it
consistently.

For example, if using:

-   Sequential Large = 629.79 ms
-   CUDA Large = 17.40 ms
-   CUDA kernel = 1.55 ms

then:

-   End-to-end speedup ≈ 36.2×
-   Kernel-only speedup ≈ 406.3×

If other numbers came from Nsight or separate runs, explicitly label
them as profiling-run measurements.

------------------------------------------------------------------------

## 1.4 The 492× CUDA kernel speedup should be removed or explained

### Problem

The conclusion refers to a:

> "492× kernel speedup"

while the consolidated results give approximately:

> "406.3×--440.4×"

The report does not explain the source of the 492× value.

### Correction

Unless there is a clearly documented measurement producing 492×, remove
it.

Safer wording:

> "The kernel-only speedup is approximately 406--440×, while end-to-end
> speedup is approximately 36--40× after including host-device transfer
> and other overhead."

------------------------------------------------------------------------

# 2. Technical interpretation issues

## 2.1 407.5% is not pure parallel efficiency

### Problem

The report describes the optimized OpenMP result as having more than
400% "parallel efficiency."

The calculation itself is mathematically correct:

`8.15× speedup / 2 threads = 4.075 = 407.5%`

However, the optimized OpenMP version is not merely the same sequential
algorithm executed with two threads. It also includes algorithmic
optimization, branch elimination, and SIMD/vectorization.

Therefore, the entire 8.15× speedup cannot be attributed to parallelism.

### Correction

Use terminology such as:

> "407.5% baseline-normalized efficiency"

and immediately clarify:

> "This should not be interpreted as pure parallel efficiency because
> the optimized implementation also changes the algorithm and benefits
> from SIMD/vectorization."

------------------------------------------------------------------------

## 2.2 MPI performance explanation based on separate address spaces is too strong

### Problem

The report suggests that MPI is faster because each MPI process has its
own address space, thereby reducing cache contention.

This is an oversimplification.

Separate process address spaces do not automatically eliminate physical
cache or memory-system contention on a shared-memory machine.

### Correction

Use a more defensible explanation:

> "The MPI implementation may perform differently because MPI ranks are
> separate processes, the number of active CPU cores differs from the
> OpenMP configuration, and the memory-access, scheduling,
> synchronization, and runtime behavior differ."

Also mention that the benchmark environments/configurations should be
kept consistent when making direct performance comparisons.

------------------------------------------------------------------------

## 2.3 `MPI_Cart_create` process-mapping claim needs correction

### Problem

The report implies that `MPI_Cart_create` itself allows MPI to optimize
process-to-core mapping.

The implementation uses:

`reorder = 0`

Therefore, process/rank reordering is disabled.

### Correction

Use:

> "`MPI_Cart_create` provides a structured Cartesian topology and
> simplifies neighbor identification through `MPI_Cart_shift`. Since
> `reorder=0` is used, rank reordering is disabled and no
> process-placement optimization is claimed."

------------------------------------------------------------------------

## 2.4 `MPI_Gatherv` bottleneck explanation is too absolute

### Problem

The report describes `MPI_Gatherv` as effectively serializing
socket/shared-memory transfers.

That is too broad as a general statement about MPI.

### Correction

Use:

> "`MPI_Gatherv` places the complete output collection on the root rank,
> making root-side aggregation a potential scalability bottleneck as the
> number of ranks and output size increase."

------------------------------------------------------------------------

# 3. Energy-analysis issues

## 3.1 CUDA should not be declared definitively the most energy-efficient version

### Problem

The report labels CUDA as the "Most Energy-Efficient Version."

However, CPU energy was unavailable because RAPL was unavailable.
Therefore, the report has a direct GPU energy measurement but not a
comparable CPU Joule measurement.

### Correction

Use:

> "Among the versions for which energy was directly measured, CUDA
> required approximately 0.643 J per Large-dataset execution. Since CPU
> energy could not be measured, a definitive cross-platform energy
> ranking cannot be established."

------------------------------------------------------------------------

## 3.2 Energy coverage should be clarified for all input sizes

### Problem

The assignment asks for energy consumed for each version at each input
size, while the report's direct GPU energy measurement is primarily
presented for the Large dataset.

CPU energy is legitimately unavailable if RAPL could not be accessed,
but the CUDA coverage should be explicitly explained.

### Correction

Add a note such as:

> "Direct GPU energy measurement was performed for the Large dataset
> using the sustained-loop power measurement procedure. Small and Medium
> GPU energy values were not separately measured."

If Small and Medium energy measurements actually exist, include them
instead.

------------------------------------------------------------------------

# 4. GPU-performance wording

## 4.1 Do not present predicted shared-memory performance as measured

### Problem

The report suggests that shared-memory tiling could raise memory
throughput to more than 80% of peak.

If this was not actually implemented and benchmarked, it is a prediction
rather than a measured result.

### Correction

Use:

> "A shared-memory tiled implementation is expected to reduce redundant
> global-memory transactions and may substantially improve memory
> throughput."

Do not state a specific \>80% result unless it was actually measured.

------------------------------------------------------------------------

# 5. Consistency improvements

## 5.1 Distinguish the best MPI result from the 4-rank result

The report correctly gives:

-   Large, MPI Basic, 8 ranks = **57.249 ms**
-   Large, MPI Basic, 4 ranks = **57.990 ms**

Both are valid.

However, later discussion can make the distinction unclear.

### Correction

Use:

> "MPI Basic achieves its best Large-dataset time at 8 ranks (57.249
> ms), while 4 ranks gives a very similar 57.990 ms."

------------------------------------------------------------------------

## 5.2 Explain different measurement runs

If Nsight profiling, normal timing, and energy benchmarking produced
different CUDA timings, do not mix them as though they are the same
experiment.

Use labels such as:

-   **Standalone benchmark timing**
-   **Nsight profiling timing**
-   **Energy benchmark timing**

Then select one consistent measurement for the final comparison table.

------------------------------------------------------------------------

# 6. Recommended final corrections before submission

### Must fix

-   [ ] Correct the "communication exceeds computation at 8 ranks across
    all datasets" statement.
-   [ ] Explain/add the unaccounted MPI profiling time.
-   [ ] Reconcile all CUDA timings.
-   [ ] Remove or explain the 492× CUDA kernel speedup.
-   [ ] Replace the \>400% "parallel efficiency" wording with a
    qualified term.
-   [ ] Remove the definitive claim that CUDA is the most
    energy-efficient overall.
-   [ ] Correct the MPI separate-address-space/cache-contention
    explanation.
-   [ ] Correct the `MPI_Cart_create` process-mapping claim.

### Strongly recommended

-   [ ] Clarify which benchmark run is used as the canonical Part C CUDA
    result.
-   [ ] Clarify energy measurement coverage for Small/Medium/Large.
-   [ ] Change predicted GPU optimization results into clearly labelled
    expectations.
-   [ ] Distinguish 4-rank and 8-rank MPI results wherever both are
    discussed.

------------------------------------------------------------------------

# Overall assessment

The report is **substantively strong**, and the main issue is not
missing work. The biggest risks are **internal numerical consistency,
overclaiming, and a few technically inaccurate explanations**.

After correcting the items above, the report should be considerably
safer to submit.
