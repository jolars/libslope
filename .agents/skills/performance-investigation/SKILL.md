---
name: performance-investigation
description: Investigate and improve performance in libslope with its Catch2 benchmarks, representative solver workloads, profiler evidence, and reproducible before-and-after measurements. Use when asked to benchmark, profile, explain a performance regression, locate bottlenecks, or optimize runtime, memory use, allocations, or scaling in this repository. Do not use for correctness debugging without a performance question.
---

# libslope Performance Investigation

Find the limiting resource under a representative libslope workload, then
verify that an optimization improves the user-visible metric without changing
solver behavior.

## Choose the evidence

- Identify the relevant metric—usually path-fitting latency or throughput—and
  the matrix type, loss, dimensions, sparsity, normalization, and thread count.
- Use `tests/real_data_benchmarks.cpp` for end-to-end solver claims. RCV1 covers
  sparse logistic regression, while E2006 covers sparse quadratic regression.
- Use a focused Catch2 `BENCHMARK` in `tests/benchmarks.cpp` to isolate a hot
  operation. A microbenchmark can explain an end-to-end result, but cannot
  establish that result by itself.
- Prefer an existing benchmark tag or task when it measures the requested path.
  Relevant tasks include `sparse-cluster-benchmark`, `full-set-benchmark`, and
  `high-dimensional-benchmark`.
- Treat `setUpdateClusters(false)` as an intentionally unsafe benchmark mode.
  It can measure update overhead, but its solver output is not a valid baseline
  for an optimization.

If the request is underspecified, inspect the affected code and existing
benchmarks, choose the least speculative workload, and state the choice. A
diagnosis request permits measurement and analysis, but not implementation; an
optimization request includes the smallest relevant benchmark and code changes.

## Establish the baseline

1. Run the focused correctness tests before changing code. Use `task test` as
   the final correctness gate for an implementation.
2. Time optimized code. The ordinary `task build` is a Debug build, and the
   current general `task benchmark` inherits it; do not use either for speed
   claims. The dedicated performance tasks build Release binaries under
   `build/benchmark`.
3. For profiling, build `build/profile` with optimized code, debug information,
   and frame pointers. Read
   [references/native-code.md](references/native-code.md) for the exact setup
   and tool choices.
4. Keep input, build flags, thread count, and benchmark parameters identical
   across comparisons. Warm up when initialization is not the target, retain
   repeated samples, and treat differences within observed noise as
   inconclusive.

Downloading the high-dimensional datasets changes external and local state.
Use existing files in `.cache/benchmarks` when present; otherwise run
`task high-dimensional-benchmark-data` only when downloading benchmark data is
within the user's request.

## Investigate and optimize

1. Profile the representative workload before choosing an implementation.
   Rank hot paths by measured cost, distinguish self cost from inclusive cost,
   and remember that inclusive percentages overlap and cannot generally be
   added.
2. Trace library frames back to their libslope callers. Eigen kernels often
   expose a cost caused by expression temporaries, dense passes, sparse
   materialization, or repeated column traversal in project code.
3. Form a concrete hypothesis about the responsible algorithm, allocation,
   data movement, cache behavior, synchronization point, or system call. Do not
   replace a container or algorithm merely because its theoretical operation
   looks cheaper when the measured path is cold.
4. Add or tighten the focused benchmark before changing the implementation when
   no stable measurement exists. Keep input construction outside the timed
   region and ensure Catch2 consumes the computed result.
5. Make one attributable change at a time when practical. Run correctness
   checks, compare it with the unchanged baseline, and report absolute results
   as well as ratios or percentages.
6. Reprofile a material improvement. Confirm that the targeted cost shrank, and
   identify whether another bottleneck now dominates.

Stop when the requested goal is met, the remaining difference is within
measurement noise, or further progress requires a materially different design
or new authority. Preserve numerical tolerances, convergence behavior, sparse
and dense paths, normalization modes, and multi-response behavior affected by
the change.

## Keep results reproducible

- Prefer Catch2's in-process benchmark harness for short operations. Use
  `hyperfine` for complete commands or executables where process startup is
  genuinely part of the comparison.
- Use elapsed time for speed claims. Samples, hardware counters, simulated
  instructions, and allocation counts explain a result but are not substitutes
  for elapsed time.
- Record the benchmark command, build type and relevant flags, dataset and
  dimensions, compiler, thread settings, and sample count.
- Keep generated `perf` and Callgrind output under `/tmp` by default. Do not
  commit traces, decompressed datasets, or other large benchmark artifacts.

## Report the evidence

Give the user the workload and commands, baseline and new measurements with
variability, correctness checks, profiler findings with self and inclusive costs
labeled, and the mechanism responsible for the change. Separate observations
from inferences. If the evidence cannot distinguish two explanations, propose
the smallest experiment that can.
