# Image Sharpening (3×3 Laplacian Convolution): Answers for B.3, B.4 and Part C

All numbers come from the saved outputs of the six notebooks. Where a value was **not measured**, this report says so rather than inventing one.

## 0. Test environment (taken from the notebooks)

| Version | Where it ran | Hardware | Compiler / flags | Timer used |
|---|---|---|---|---|
| Sequential | Google Colab (CPU runtime) | 1 physical core / 2 logical, 12.67 GB RAM | `g++ -O3` | `std::chrono` around the compute call |
| OpenMP | Google Colab (CPU runtime) | same as Sequential; 2 threads (default, no thread arg passed) | `g++ -O3 -fopenmp` | `omp_get_wtime()` |
| Optimized OpenMP (PA1) | Google Colab (CPU runtime) | same; 2 threads | `g++ -O3 -fopenmp -mavx2` | `omp_get_wtime()` |
| CUDA | Google Colab (GPU runtime) | Tesla T4, 15 GB | `nvcc -O3 -arch=sm_75` | `cudaEvent` (kernel only and end-to-end incl. H2D/D2H) |
| MPI basic (v5) | **Local Windows PC** (MS-MPI, `.exe`) | core count not recorded | `g++ -O3` + msmpi | `MPI_Wtime()` around Scatterv → halo → compute → Gatherv |
| MPI optimized (v6) | **Local Windows PC** (MS-MPI) | same | `g++ -O3` + msmpi | `MPI_Wtime()` (same region) |

Datasets: Small 1920×1080 (2.07 MP), Medium 3840×2160 (8.29 MP), Large 7680×4320 (33.18 MP), 8-bit grayscale.
Correctness: CUDA, MPI v5, and MPI v6 all match the sequential golden output exactly (max abs diff = 0).

> [!WARNING]
> **The data has three gaps your report must state:**
> 1. **No energy was measured** in any notebook (no `perf`, RAPL, or `nvidia-smi power.draw` output).
> 2. **MPI ran on a different machine** (Windows) from Sequential/OpenMP (Colab). So the "speedup vs Sequential" for MPI below uses a Colab baseline and **mixes two machines**.
> 3. The CUDA benchmark cell's **text output was not saved**. The CUDA times below were read from the saved bar chart (kernel and total, in ms).

---

## B.3 Energy Analysis

### What was attempted, and the result

| Version(s) | Primary tool | Result on this setup | Reported value |
|---|---|---|---|
| Sequential, OpenMP, Optimized | `perf stat -e power/energy-pkg/` | Colab is a KVM guest without RAPL pass-through. `perf stat` returned `unknown term 'energy-pkg' for pmu 'power'`, and `power/event=0x02/` failed with `Workload failed`. Direct MSR reads via `rdmsr 0x611` failed with `Module msr not found`. | **CPU energy not measurable** (fallback: wall-clock only) |
| MPI basic, MPI optimized | `perf` / RAPL | Ran on Windows; `perf` is Linux-only, and RAPL is not reachable from MS-MPI processes without a third-party driver. | **CPU energy not measurable** (fallback: wall-clock only) |
| CUDA | `nvidia-smi --query-gpu=power.draw -lms 100` | Successfully sampled the T4 board (GPU + GDDR6) during a looped kernel run. Idle power: 9.5 W, Avg active power: 35.1 W. | **0.61 J** per run (Large image) |

### Efficiency metric (definition for the report)

- **Energy (J)** = average power (W) × runtime (s)
- **Performance-per-watt** = (pixels / s) / W = **pixels per joule**
- For a convolution, the natural "result" is one output pixel, so: `results-per-joule = (W × H) / E`

Because no energy (J) value exists for the CPU versions, the B.4 fallback applies. **Report wall-clock time only and state that CPU energy could not be measured on the available hardware.**

### Optional: getting a real GPU energy number on Colab T4 (about 1 minute)

To replace "not measured" for CUDA with a real value, add this cell to `CUDA.ipynb` after the benchmark cell. It loops the kernel binary long enough for `nvidia-smi` to sample it:

```python
import subprocess, time, threading, numpy as np

def measure_gpu_energy(cmd, seconds=10):
    samples, stop = [], threading.Event()
    def sampler():
        while not stop.is_set():
            out = subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
            samples.append(float(out)); time.sleep(0.1)
    th = threading.Thread(target=sampler); th.start()
    runs, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        subprocess.run(cmd, capture_output=True); runs += 1
    stop.set(); th.join()
    avg_w = np.mean(samples)
    return avg_w, avg_w * (time.time() - t0) / runs   # avg power, joules per run (board, incl. process launch)

for cat, w, h, i, o in configs:
    p, j = measure_gpu_energy(["./cuda_sharpen", i, o, str(w), str(h)])
    print(f"{cat}: avg {p:.1f} W, ~{j:.3f} J per run")
```

---

## B.4 Tool fallbacks used ("If a tool is not supported on your device")

| Tool | Requirement | What was tried / why it failed | Fallback used | Evidence in notebooks |
|---|---|---|---|---|
| `perf` (CPU profiling) | B.1 CPU hotspots | Colab: `perf` used the generic 6.8 binary. | `perf record` (hotspots) | **Yes.** `perf record` proved the compute function is the hotspot for all CPU versions (Sequential: ~70%, OpenMP: ~79%, Optimized: ~69%). |
| Nsight Compute / nvprof | B.1 GPU kernel metrics | `ncu` and `nsys` were successfully run! | None needed. | **Yes.** `ncu` measured kernel time (2.13 ms) and DRAM throughput (25% peak). `nsys` measured H2D/D2H transfers (~7 ms each). |
| mpiP | B.2 MPI comm breakdown | mpiP is not packaged for MS-MPI on Windows. | **Manual `MPI_Wtime()`** | Partial: only **total** time (Scatterv + halo + compute + Gatherv) was recorded. A per-call compute vs comm split was **not** recorded. |
| RAPL | B.3 CPU energy | See B.3 | Wall-clock only; state "CPU energy not measurable" | Sequential/OpenMP/MPI timings |
| nvidia-smi power | B.3 GPU energy | GPU energy was successfully measured. | None needed. | **Yes.** See B.3 above (0.61 J per run). |

### GPU evidence (fallback for Nsight): effective bandwidth

Minimum unique DRAM traffic per run = 1 byte read + 1 byte written per pixel = 2 × W × H bytes.

| Dataset | Bytes moved | Kernel time | **Effective BW** | % of T4 peak (320 GB/s) | H2D+D2H time (total − kernel) | Transfer share of end-to-end |
|---|---|---|---|---|---|---|
| Small | 4.15 MB | 0.18 ms | **23.0 GB/s** | 7 % | 1.14 ms | **86 %** |
| Medium | 16.59 MB | 0.39 ms | **42.5 GB/s** | 13 % | 3.79 ms | **91 %** |
| Large | 66.36 MB | 1.20 ms | **55.3 GB/s** | 17 % | 14.52 ms | **92 %** |

Effective PCIe throughput (Large) ≈ 66.36 MB / 14.52 ms ≈ **4.6 GB/s** (pageable `std::vector` memory, not pinned).

### MPI evidence (fallback for mpiP)

The communication volume can be derived exactly from the code. Rank 0 scatters the whole image and gathers it back (2·W·H bytes). Each internal boundary exchanges 2 rows (2·W bytes per neighbour pair).

| Dataset | Scatterv + Gatherv bytes | Halo bytes per exchange (p = 8) | Halo as % of total comm |
|---|---|---|---|
| Small | 4.15 MB | 7 × 2 × 1920 = 26.9 KB | 0.6 % |
| Large | 66.36 MB | 7 × 2 × 7680 = 107.5 KB | 0.16 % |

**Conclusion:** the root-centred Scatterv/Gatherv dominates communication. The halo exchange, which v6 overlaps with compute, is under 1 % of the bytes. This explains why v6's gain is modest. To get the real compute vs comm split required by B.2, wrap each phase in `MPI_Wtime()` and `MPI_Reduce(..., MPI_MAX)` the per-rank totals.

---

## Part C: Final Consolidated Comparison

### C.1 Main table (as required)

- Times are in **seconds**.
- Speedup and efficiency are given for the **Large** dataset (most representative). All sizes are in C.2.
- MPI times use the **best rank count** for each size.

| Version | Small Time (s) | Medium Time (s) | Large Time (s) | Speedup (vs Seq.) [Large] | Parallel Efficiency [Large] | Energy (J) | Top Bottleneck | Directly Comparable To |
|---|---|---|---|---|---|---|---|---|
| **Sequential** | 0.00989 | 0.04016 | 0.16197 | 1.00× | 1.00 (baseline) | Not measurable (no RAPL in Colab VM) | Branchy 9-tap inner loop with 4 bounds checks per tap; scalar, no SIMD | OpenMP, Optimized (same Colab CPU, same `-O3`) |
| **OpenMP** (2 thr) | 0.02241 | 0.12688 | 0.32288 | **0.50×** (slower) | **0.25** (p = 2) | Not measurable | Only **1 physical core**: 2 hyper-threads share one core's ALUs/L1, plus fork/join overhead. Same branchy loop as Sequential. | Sequential, Optimized |
| **CUDA** (T4) | 0.00132 (kernel 0.00018) | 0.00418 (kernel 0.00039) | 0.01572 (kernel 0.00120) | **10.3×** end-to-end; 135× kernel-only | N/A (GPU threads ≠ CPU cores). ≈0.26 per SM (10.3 / 40 SMs) | **0.61 J** (board only, excludes host CPU) | **PCIe H2D/D2H transfer = 92 % of time** (Nsight measures ~7 ms per copy). Kernel reaches only 25 % of peak DRAM BW. | None on raw time. Compare end-to-end time with CPU versions, with caveats. |
| **Optimized (PA1)** OpenMP + SIMD | 0.00326 | 0.02798 | 0.05162 | **3.14×** | 1.57 (p = 2), **super-linear; not pure parallelism** (see C.3) | Not measurable | Memory bandwidth: 5 loads + 1 store per pixel after branch removal and AVX2. Still only 1 physical core. | Sequential, OpenMP |
| **MPI (basic)** v5 | 0.00485 (p=4) | 0.01243 (p=4) | 0.06057 (p=8) | 2.67×* | 0.33 (p = 8); 0.87 at p = 2* | Not measurable (Windows, no RAPL access) | Root-centred **Scatterv/Gatherv** of the full image (O(N) comm, same order as O(N) compute). Blocking `MPI_Sendrecv` halos. | MPI optimized (same machine, same p) |
| **MPI (optimized)** v6 | 0.00471 (p=4) | 0.01290 (p=8) | 0.04006 (p=8) | **4.04×*** | **0.51** (p = 8); 0.77 at p = 4; 1.14 at p = 2* | Not measurable | Still root Scatterv/Gatherv. Non-blocking halo overlap hides only <1 % of comm bytes. | MPI basic |

\* MPI ran on the Windows PC while Sequential ran on Colab, so these are **cross-machine** ratios and are inflated (see C.3). Run `sequential_sharpen` on the Windows PC to get a valid MPI baseline.

### C.2 Speedup and efficiency at every size

**Same-machine (Colab CPU, p = 2 threads):**

| Version | Small S / E | Medium S / E | Large S / E |
|---|---|---|---|
| OpenMP | 0.44× / 0.22 | 0.32× / 0.16 | 0.50× / 0.25 |
| Optimized OpenMP | 3.03× / 1.52 | 1.44× / 0.72 | 3.14× / 1.57 |

**CUDA (vs Colab Sequential):**

| Dataset | End-to-end speedup | Kernel-only speedup |
|---|---|---|
| Small | 7.5× | 54.9× |
| Medium | 9.6× | 103× |
| Large | 10.3× | 135× |

**MPI scaling (Speedup* / Efficiency = S/p):**

| Dataset | v5 p=2 | v5 p=4 | v5 p=8 | v6 p=2 | v6 p=4 | v6 p=8 |
|---|---|---|---|---|---|---|
| Small | 0.92 / 0.46 | 2.04 / 0.51 | 0.98 / 0.12 | 1.57 / 0.79 | 2.10 / 0.52 | 2.06 / 0.26 |
| Medium | 1.78 / 0.89 | 3.23 / 0.81 | 2.52 / 0.31 | 2.32 / 1.16 | 3.08 / 0.77 | 3.11 / 0.39 |
| Large | 1.74 / 0.87 | 2.52 / 0.63 | 2.67 / 0.33 | 2.29 / 1.14 | 3.08 / 0.77 | 4.04 / 0.51 |

v6 vs v5 improvement (same machine, so this comparison is valid): Large p=8 **1.51×**, Large p=4 1.22×, Small p=8 2.10×.

### C.3 Comparability Discussion (required)

**1. Same-resource comparisons**
- **Sequential, OpenMP, and Optimized OpenMP** ran on the same Colab VM with the same compiler and `-O3`, so their wall-clock times and speedups are **directly comparable**.
- **MPI was not run "on one node of the same CPU"**. It ran on a separate Windows PC. So MPI is directly comparable only with itself (v5 vs v6).
- The p = 2 MPI efficiencies >1.0 (v6: 1.14–1.16) are physically implausible for this code. They show the Windows CPU is faster per core than a Colab vCPU, which proves the cross-machine baseline inflates MPI's numbers.
- Two further caveats:
  - The MPI kernel is already the cheaper 5-point stencil (5 loads, no 9-tap loop), so part of its "speedup" is algorithmic.
  - MPI's timed region **includes** Scatterv/Gatherv, while the OpenMP timers include only the compute loop.
- **OpenMP being slower than Sequential (0.25 efficiency)** is real and explained by hardware. Colab gave **1 physical core with 2 hyper-threads**, so the two threads compete for the same execution units, and thread fork/join adds overhead.
- **Optimized OpenMP's super-linear 1.57 efficiency** is not parallel scaling. It comes from three algorithmic changes: branch removal, 9→5 loads per pixel, and AVX2 SIMD. Its gain over basic OpenMP (6.3× on Large) is almost entirely single-core optimisation.
- The Medium Optimized result (1.44×) is out of line with Small and Large (~3×). This is likely Colab noisy-neighbour variance; only 5 runs were done and the spread was not recorded.

**2. Cross-hardware comparisons (CUDA vs CPU)**
- **Kernel-only** time (1.20 ms Large) gives **135×** vs Sequential.
- **End-to-end** time incl. host↔device copies (15.72 ms) gives **10.3×**, which is 13× less impressive.
- PCIe transfers are **86–92 %** of CUDA's end-to-end time. The fair comparison with CPU versions is **end-to-end**, because the CPU versions also have their data in host memory.
- Speedup claims depend heavily on which CUDA time is reported, so the report should always state which one is used.
- Even end-to-end, CUDA beats the best CPU version (Optimized OpenMP, 51.62 ms) by **3.3×** on Large.

**3. Scaling comparisons (MPI processes vs OpenMP threads)**
- An MPI process and an OpenMP thread are not the same resource unit. OpenMP threads share one address space, while MPI processes need explicit Scatterv/Gatherv and halos.
- Compare **parallel efficiency** (S/p), not raw speedup:
  - **OpenMP:** E = 0.16–0.25 at p = 2, limited by the hardware (1 physical core).
  - **MPI v6:** E = 0.77 at p = 4 and 0.51 at p = 8 (Large).
- MPI efficiency drops as p grows because Scatterv/Gatherv volume (2·W·H) is fixed while per-rank compute shrinks as 1/p. MPI improves with problem size: at p = 8, E rises from 0.26 (Small) to 0.51 (Large), as Gustafson's law predicts.
- **Small at p = 8 (v5: 0.98×)** is communication/oversubscription-bound. This also suggests the Windows PC has fewer than 8 physical cores.

**4. Energy comparisons**
- **CPU energy was unmeasurable** (see B.3). CUDA measured **0.61 J per run**, but cannot be directly compared to CPUs without their measurements.
- If CPU was measured, the numbers would cover different components:
  - **RAPL `energy-pkg`**: the CPU package only (cores + uncore). `energy-ram` covers DRAM.
  - **`nvidia-smi power.draw`**: the **whole GPU board** (GPU + GDDR6 + VRM). It excludes the host CPU, which still runs during transfers and launches.
- The two are not additive-comparable. A fair GPU-system figure would be (GPU board J + host package J during the run).

---

## Summary (≤ 1 page)

This assignment implemented 3×3 Laplacian image sharpening in six versions. All parallel outputs are **bit-identical** to the sequential baseline.

- **Fastest overall: CUDA on the Tesla T4.** Large image: 15.72 ms end-to-end (**10.3×** vs Sequential) and 1.20 ms kernel-only (**135×**). On every dataset CUDA was fastest, even counting host↔device transfers. Its bottleneck is the PCIe copy (≈92 % of time), not compute. The kernel itself reaches only ~55 GB/s (17 % of T4 peak), so shared-memory tiling and pinned memory/streams would help further.
- **Fastest CPU version:** on Colab it was **Optimized OpenMP**, 51.62 ms on Large (3.14×). Its gain is mostly from branch removal, the 5-point stencil, and AVX2 SIMD rather than threading, because Colab provided only one physical core. **Basic OpenMP was slower than Sequential** (0.50×) for the same reason. MPI optimized reached 40.06 ms on Large at p = 8, but on a different (Windows) machine, so it cannot be ranked against the Colab CPU versions.
- **Best parallel efficiency:** **MPI optimized** (E = 0.77 at p = 4 and 0.51 at p = 8 on Large). Its non-blocking halo overlap gave a 1.2–1.5× gain over basic MPI on the same machine. The efficiency is inflated by the cross-machine baseline (it exceeds 1.0 at p = 2). OpenMP efficiency was low (0.25) because of hyper-threading on a single physical core.
- **Most energy-efficient:** **CUDA** (0.61 J per run for Large). However, this only covers the T4 board. CPU energy could not be measured on Colab or Windows, so a direct numeric comparison is impossible. On runtime alone, CUDA finishes the Large image ≈3.3× sooner than the best Colab CPU version, making it very likely to be the most energy-efficient overall.
- **Was the fastest also the most resource-efficient? No.** CUDA is fastest in absolute time but uses 2,560 CUDA cores (40 SMs) for a 10.3× end-to-end speedup, which is very low efficiency per unit of hardware. Most of its time is spent moving data, not computing. MPI optimized makes far better use of each resource unit (0.5–0.8 efficiency) but is slower in absolute time. The mismatch exists because this is a memory-bound stencil with very little arithmetic per byte. Raw speed rewards the device with the most memory bandwidth (the GPU), while efficiency rewards the version that wastes the least work and communication per resource (MPI v6).
