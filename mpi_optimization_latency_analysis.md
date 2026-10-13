# Architectural Analysis: Why Optimized MPI (Version 6) Shows Latency Overhead Over Basic MPI (Version 5)

## 1. Executive Summary

In Programming Assignment 2 (PA 2), two distributed implementations of 2D image sharpening (spatial Laplacian convolution) were developed and benchmarked:
1. **Version 5 (Basic MPI):** 1D row-block domain decomposition using blocking point-to-point communication (`MPI_Sendrecv`) for halo/ghost cell exchange.
2. **Version 6 (Optimized MPI):** 1D Cartesian topology (`MPI_Cart_create`, `MPI_Cart_shift`) with non-blocking asynchronous communication (`MPI_Isend`, `MPI_Irecv`, `MPI_Waitall`) designed to achieve **computation-communication overlap**.

Empirical benchmarking across Small ($1920\times 1080$), Medium ($3840\times 2160$), and Large ($7680\times 4320$) datasets revealed an important architectural phenomenon:
> **The "Optimized" Version 6 often matches or slightly trails the "Basic" Version 5 in execution time on a single multicore node, rather than achieving superlinear speedups.**

This document provides the theoretical, architectural, and empirical justification for why this occurs, formatted directly for inclusion in the final course report and viva discussion.

---

## 2. Benchmark Comparison (Empirical Data)

| Dataset | Image Size | Ranks ($P$) | Basic MPI (v5) Avg Time | Optimized MPI (v6) Avg Time | Observed Behavior |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Small** | $1920 \times 1080$ | 2 | ~5.3 – 6.2 ms | ~5.3 – 6.3 ms | Parity / negligible difference |
| **Small** | $1920 \times 1080$ | 4 | ~4.8 – 5.8 ms | ~4.7 – 6.0 ms | Parity |
| **Small** | $1920 \times 1080$ | 8 | ~8.2 – 10.0 ms | ~4.8 – 8.8 ms | Communication overhead dominates |
| **Medium** | $3840 \times 2160$ | 2 | ~22.5 – 24.1 ms | ~17.3 – 27.2 ms | Slight variance |
| **Medium** | $3840 \times 2160$ | 4 | ~12.4 – 14.4 ms | ~13.0 – 16.4 ms | Basic matches or edges Opt |
| **Large** | $7680 \times 4320$ | 2 | ~63.5 – 91.9 ms | ~65.2 – 104.9 ms | Basic slightly faster |
| **Large** | $7680 \times 4320$ | 4 | ~58.0 – 62.0 ms | ~52.6 – 70.8 ms | Mixed / comparable |
| **Large** | $7680 \times 4320$ | 8 | ~57.2 – 58.0 ms | ~40.0 – 56.6 ms | Opt edges out when computation is large |

---

## 3. Four Core Architectural Reasons for the Overhead

### Reason 1: The "Shared-Memory vs. Network" Paradigm Mismatch
* **The Theoretical Assumption:** Non-blocking asynchronous messaging (`MPI_Isend` / `MPI_Irecv`) was architected for distributed-memory supercomputing clusters interconnected via physical Ethernet or InfiniBand fabrics. In those clusters, network latency is on the order of milliseconds, and the CPU would otherwise stall idle waiting for packets to cross physical cables.
* **The Reality on a Single Node:** When running on a personal computer or single workstation via MS-MPI / MPICH, inter-process communication is performed via **shared-memory byte copies (OS loopback shared memory)**.
* **The Consequence:** The physical transfer of a boundary row ($1920$ to $7680$ bytes) between two local processes takes **sub-microsecond time**. There is essentially **no network latency to hide**. The benefit of hiding a sub-microsecond memory copy is dwarfed by the software overhead of setting up asynchronous non-blocking state machines.

---

### Reason 2: Loop Disruption, Cache Thrashing, and Compiler Optimization Penalties
The structural difference in the computational loop between Version 5 and Version 6 significantly affects CPU microarchitecture:

#### Basic MPI (v5) — One Monolithic, Highly Vectorizable Loop:
```cpp
for (int r = 1; r <= local_rows; ++r) {
    for (int c = 0; c < W; ++c) {
        // Uniform, contiguous memory access
        int val = 5 * center - (up + down + left + right);
        local_out[...] = clamp_pixel(val);
    }
}
```
* **Hardware Benefits:** The CPU's hardware prefetcher reads rows sequentially from top to bottom. The GNU compiler (`-O3`) can easily vectorize the inner loop, unroll iterations, and minimize pipeline bubbles.

#### Optimized MPI (v6) — Fragmented Into Three Disjoint Passes:
```cpp
// 1. Asynchronous communication post
MPI_Irecv(...); MPI_Isend(...);

// 2. Interior pass: rows 2 to local_rows - 1
for (int r = 2; r < local_rows; ++r) { ... }

// 3. Synchronization barrier
MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

// 4. Boundary pass 1: row 1
for (int c = 0; c < W; ++c) { ... }

// 5. Boundary pass 2: row local_rows
for (int c = 0; c < W; ++c) { ... }
```
* **Hardware Drawbacks:**
  1. Splitting the row loop introduces extra branch conditions and loop prologues/epilogues.
  2. The CPU's instruction pipeline and data prefetching are interrupted by the `MPI_Waitall` synchronization call.
  3. When the thread returns to compute rows $1$ and $local\_rows$, the cache lines for those border rows may have been evicted during the interior computation, causing **L1/L2 data cache misses**.

---

### Reason 3: MPI Request Handle Management & Polling Overhead
To execute non-blocking communication in Version 6, the runtime must allocate and track four separate communication requests:
- `MPI_Request reqs[4]`
- Tracking two active receive queues (`MPI_Irecv`)
- Tracking two active send queues (`MPI_Isend`)
- Invoking `MPI_Waitall()` to query and resolve request states

Each non-blocking call involves internal memory allocation, queue insertions, and state transitions within the MPI library. In contrast, `v5_mpi_basic.cpp` uses two straightforward `MPI_Sendrecv` calls that directly copy memory and immediately retire without request management overhead.

---

### Reason 4: Communicator Creation Overhead (`MPI_Cart_create`)
Version 6 initializes a virtual 1D Cartesian topology:
```cpp
MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart_comm);
MPI_Cart_shift(cart_comm, 0, 1, &prev_rank, &next_rank);
```
While mathematically elegant and standard for multidimensional domain decomposition:
1. `MPI_Cart_create` is an **internal collective call** that duplicates and restructures the communicator, building process routing tables and topology metadata.
2. For a simple 1D linear pipeline of ranks, manual neighbor arithmetic (`rank - 1` and `rank + 1`) in Version 5 executes in zero CPU cycles, whereas topology creation incurs startup communication overhead across all ranks.

---

### Reason 5: Process Over-Subscription & Core Contention ($P = 8$)
On typical consumer/lab CPUs (which typically feature 4 physical cores and 8 logical hyperthreads):
- Launching $8$ full OS processes forces the operating system to schedule 8 heavy processes simultaneously.
- When all 8 processes poll active asynchronous MPI requests and simultaneously calculate memory-bound 2D convolutions, the **shared memory bus and shared L3 cache become saturated**.
- As Amdahl’s Law and communication scaling predict, when the number of processes increases, the ratio of boundary communication relative to local computation increases, creating a communication bottleneck at higher process counts.

---

## 4. How to Present This in the Final Report & Viva

When asked:
> **"Why does Version 6 not show significant speedup over Version 5, and in some cases run slower?"**

Use the following structured answer:

> *"The design goal of Version 6 was to achieve computation-communication overlap using non-blocking primitives (`MPI_Isend`/`MPI_Irecv`) and a structured Cartesian topology (`MPI_Cart_create`).*
> 
> *However, because the benchmark was conducted on a single shared-memory node rather than a distributed physical cluster, inter-process communication latency is already sub-microsecond through local RAM copies. Therefore, there is virtually no network latency to overlap.*
> 
> *Meanwhile, Version 6 incurs administrative costs: managing asynchronous request objects (`MPI_Waitall`), communicator duplication (`MPI_Cart_create`), and splitting a clean, contiguous loop into separate interior and boundary loops. This loop splitting disrupts CPU hardware prefetching and vectorization optimizations.*
> 
> *This result validates a key principle of parallel and distributed systems: **latency-hiding optimizations yield benefits only when physical communication latency is significantly greater than software dispatch overhead**."*
