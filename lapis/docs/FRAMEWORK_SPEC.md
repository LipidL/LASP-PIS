# Periodic Pairwise Framework — v1 Specification

## Problem Statement

Atomistic models and other short-range physical simulations repeatedly apply the same computational pattern inside a periodic cell:

1. Start with atom or node data such as coordinates, elements, and learned features.
2. Find every periodic neighbor inside a cutoff.
3. Compute potentially large edge-level intermediates.
4. Reduce those intermediates back into node-level results.

Conventional tensor-framework implementations commonly materialize neighbor connectivity and large edge-feature tensors in global GPU memory. For large systems and wide equivariant models, the edge intermediates dominate peak memory. A representative MH1 workload with 12,288 atoms and 1,085,440 directed edges uses an estimated 46.80 GiB for PyTorch edge tensors alone, while a specialized fused CUDA implementation keeps edge state transient and uses approximately 4.35 GiB in total.

The desired framework must preserve that fused execution model without hard-coding one model architecture. It must let expert CUDA model developers implement pairwise operations as if they consume a neighbor tile, while the framework discovers, compacts, stages, and reuses that tile on demand. Durable state should normally remain node-level. Edge-level state should remain local to a kernel invocation and should not be materialized globally unless a model explicitly chooses that escape hatch.

The framework must also make its limitations explicit. Version 1 is not a general graph runtime, automatic-differentiation system, or high-level neural-network API. Its L1 interface targets CUDA experts who are prepared to control thread mapping, shared memory, synchronization, reductions, atomics, and analytical force logic.

## Solution

Build a CUDA framework specialized for fused periodic `node -> transient edge -> node` computations.

The public L0 layer exposes a blocking, reusable calculator. A caller supplies one fully periodic structure in host memory and receives caller-owned energy, force, virial, or model-specific outputs only after GPU execution and host transfer have completed. One calculator is single-flight: a second call before the first returns is rejected.

The expert L1 layer exposes a statically specialized `PairOp` concept. A `PairOp` declares its launch and workspace requirements and participates in a collective lifecycle consisting of planning, setup, per-CTA preparation, repeated tile consumption, finalization, and optional postprocessing. The framework owns the outer kernel, receiver scheduling, spatial preprocessing, candidate traversal, accepted-edge compaction, shared-memory tile lifetime, and final restoration of original atom order. L1 owns model mathematics, operation-specific workspace layout, reductions, atomic destinations, analytical force/backward programs, and model-level numerical guarantees.

For each receiver atom, the framework assigns a CUDA thread block. Threads traverse candidate atoms, calculate periodic displacement and squared distance, accept edges satisfying the cutoff, and compact accepted edges into a structure-of-arrays queue in shared memory. Complete edge windows are passed collectively to the inlined `PairOp::consume` operation. The final partial window is accompanied by a valid-entry count; padded fields are unspecified. The operation may retain node-level accumulators in registers or declared shared scratch, but it may not retain tile pointers after `consume` returns.

Two neighbor-discovery backends are provided. The brute-force backend enumerates sender and periodic-image tuples and is the correctness fallback for small or geometrically unsuitable cells. The grid backend spatially permutes atoms, uses a dense periodic cell grid whose bin heights are at least the cutoff, and traverses the receiver bin plus the 26 adjacent bin offsets. An occupancy-aware `Auto` policy validates geometry, caps or coarsens grid dimensions, estimates candidate work from bin occupancy, and chooses between valid backends.

The first production model is MACE MH1. The same runtime grammar also supports models such as NequIP whose durable messages are node-level. Architectures with persistent edge state, edge-to-edge message passing, or nonlocal iterative solvers are excluded from v1.

## User Stories

1. As an atomistic-simulation user, I want to evaluate energy and forces for a periodic structure through one calculator call, so that I do not need to understand CUDA execution details.

2. As an atomistic-simulation user, I want the calculator call to block until results are valid, so that host-side dependency handling is unambiguous.

3. As an atomistic-simulation user, I want returned per-atom outputs in my original atom order, so that spatial permutation remains invisible to my application.

4. As an atomistic-simulation user, I want the calculator to reuse previously allocated capacity, so that repeated molecular-dynamics steps avoid frequent GPU allocation.

5. As an atomistic-simulation user, I want concurrent use of a single-flight calculator to return a defined error, so that accidental multi-threaded calls do not silently corrupt results.

6. As an ASE user, I want structures to enter through the same frontend used by comparison implementations, so that end-to-end benchmarks reflect a realistic workflow.

7. As a model integrator, I want immutable weights and lookup tables separated from structure-dependent buffers, so that one loaded model can process many sequential structures.

8. As a model integrator, I want grow-only execution capacity, so that normal fixed-size trajectories stabilize after their first allocation.

9. As a model integrator, I want access to selected-backend diagnostics, so that I can explain unexpected neighbor-search performance.

10. As a model integrator, I want the framework to report invalid geometry and excessive kernel resources cleanly, so that unsupported inputs fail predictably.

11. As an L1 CUDA developer, I want to provide a template-specialized pair operation, so that the compiler can inline and specialize my computation.

12. As an L1 CUDA developer, I want a host-callable workspace specification, so that operation-specific global and shared storage is reserved before launch.

13. As an L1 CUDA developer, I want explicit alignment and scratch-region partitioning, so that my dynamic shared memory cannot overlap framework storage.

14. As an L1 CUDA developer, I want collective `prepare`, `consume`, and `finalize` phases, so that I can coordinate all threads assigned to one receiver.

15. As an L1 CUDA developer, I want an optional setup phase before pair traversal, so that atomic destinations and temporary buffers can be initialized without cross-CTA races.

16. As an L1 CUDA developer, I want optional postprocessing after traversal, so that per-CTA partials can be reduced hierarchically.

17. As an L1 CUDA developer, I want a tile entry count instead of per-edge validity bytes, so that final-tile predication has minimal storage and initialization cost.

18. As an L1 CUDA developer, I want every CTA thread to enter every lifecycle callback in the same dynamic order, so that block collectives can be used safely.

19. As an L1 CUDA developer, I want explicit access to the integer periodic-image shift, so that geometry-dependent models and virial calculations retain complete periodic information.

20. As an L1 CUDA developer, I want edge-tile accessors to inline to raw shared-memory addressing, so that abstraction does not impose material overhead.

21. As an L1 CUDA developer, I want to declare operation-specific block shape, edge subgroup size, and scratch requirements, so that specialized MACE kernels can preserve their existing schedules.

22. As an L1 CUDA developer, I want to keep persistent per-thread accumulators across tile callbacks, so that values can remain in registers when resource pressure permits.

23. As an L1 CUDA developer, I want operation-specific shared scratch for cross-thread state, so that large tensor-product accumulators and staged matrix products are supported.

24. As an L1 CUDA developer, I want to implement exclusive stores, atomic scatter, or hierarchical reductions myself, so that model-specific output shapes do not constrain the framework.

25. As an L1 CUDA developer, I want device-detected failures reported through a status buffer, so that errors can cross the device/host boundary without pretending a per-thread return value is a host CUDA error.

26. As a force-kernel author, I want to provide analytical backward or force `PairOp`s, so that force evaluation can recompute transient edge quantities without storing global edge tensors.

27. As a force-kernel author, I want to choose atomics or an additional gather traversal, so that I can trade memory, determinism, and recomputation for each model.

28. As a pair-potential author, I want the full directed edge convention documented, so that I can apply correct half-pair energy counting.

29. As a message-passing author, I want both directions of every neighbor relationship, so that receiver-centered aggregation has the expected semantics.

30. As a model author, I want to declare a maximum search cutoff and apply smaller species-dependent cutoffs inside my operation, so that pair-specific envelopes remain possible in v1.

31. As a model author, I want cutoff smoothness and analytical-force consistency to remain my responsibility, so that the runtime does not impose one physical interaction form.

32. As a framework developer, I want a brute-force neighbor backend, so that small cells, repeated images, and geometrically unsuitable grids retain a simple correctness path.

33. As a framework developer, I want a periodic grid backend, so that large systems avoid all-pairs candidate traversal.

34. As a framework developer, I want reciprocal-plane-height criteria for grid validity, so that triclinic cells are handled geometrically rather than through vector-length assumptions.

35. As a framework developer, I want grid dimensions coarsened under a memory budget, so that sparse cells do not create cell metadata much larger than atom data.

36. As a framework developer, I want occupancy-aware backend selection, so that clustered or sparse structures do not rely on geometry alone.

37. As a framework developer, I want accepted candidates compacted with warp ballots and prefix ranks, so that expensive pair operations do not execute over mostly rejected candidates.

38. As a framework developer, I want enough accepted-edge queue capacity for one candidate batch plus a carried remainder, so that no accepted edge is overwritten before consumption.

39. As a framework developer, I want explicit synchronization before callback reads and before queue reuse, so that shared-memory lifetime is well defined.

40. As a framework developer, I want internal-to-original and original-to-internal permutations, so that internal coalescing does not change public atom identity.

41. As a framework developer, I want a counting/histogram, prefix-scan, and scatter preprocessing pipeline, so that spatial permutation scales with atoms and grid cells rather than requiring a comparison sort.

42. As a framework developer, I want the grid and brute-force backends to implement the same edge contract, so that backend selection does not change model semantics.

43. As a framework developer, I want test-only edge collection, so that an implicit-neighbor runtime can still compare exact edge multisets against a reference enumerator.

44. As a framework developer, I want shared-memory and register resource diagnostics for each specialized kernel, so that low occupancy and launch infeasibility are observable.

45. As a performance engineer, I want the generalized MH1 implementation to stay within 10% of the specialized implementation, so that abstraction does not erase the optimization benefit.

46. As a performance engineer, I want the framework to run at least twice as fast as matched PyTorch MACE with cuEquivariance and without `torch.compile`, so that the CUDA specialization has a meaningful payoff.

47. As a performance engineer, I want implicit-neighbor execution to stay within 15% of a matched explicit-list CUDA backend when that backend fits, so that memory savings do not cause excessive slowdown.

48. As a performance engineer, I want a representative 50,000-atom MH1 workload to fit within 24 GiB, so that the framework demonstrates its intended capacity advantage.

49. As a benchmark consumer, I want every implementation to replay the same coordinate trajectory, so that performance comparisons do not diverge onto different MD paths.

50. As a benchmark consumer, I want closed-loop MD reported separately, so that physical stability and energy drift are not confused with matched-workload timing.

51. As a benchmark consumer, I want host transfers, preprocessing, permutation, forces, synchronization, and output copies included consistently, so that end-to-end latency is comparable.

52. As a benchmark consumer, I want fixed dataset revisions, structure identifiers, seeds, model checksums, and software versions, so that published results are reproducible.

53. As a benchmark consumer, I want both allocator-tracked bytes and process-level GPU memory reported, so that hidden runtime overhead remains visible.

54. As an external developer, I want the benchmark environment packaged in a container, so that I can reproduce correctness, memory, and timing results conveniently.

55. As an external developer, I want a small Lennard-Jones reference operation, so that I can learn the L1 contract without first understanding the complete MACE implementation.

56. As an external developer, I want unsupported architecture classes documented, so that I do not attempt to force persistent edge-state or nonlocal algorithms into the streaming abstraction.

57. As a future API designer, I want the low-level `PairOp` contract to remain a stable expert layer, so that a later restricted high-level map/reduce interface can be added without rewriting validated kernels.

## Implementation Decisions

- The fundamental execution grammar is `node -> transient edge -> node`. Multiple message-passing stages are expressed as multiple pair traversals separated by globally stored node-level intermediates.

- The principal memory optimization is the elimination of global edge-feature materialization. Avoiding a global neighbor list is beneficial but is not, by itself, expected to provide the measured multi-fold memory reduction.

- Version 1 accepts one fully periodic three-dimensional structure. Partial periodicity and heterogeneous structure batches are not supported.

- Geometry and model computation use `float32` in v1. Neighbor, atom, grid, and image indices use signed 32-bit integers.

- The public calculator is reusable but blocking and single-flight. Inputs and outputs reside in host memory. Results become valid when the call returns and are owned by the caller. A stable internal pinned-host staging buffer is used for result transfer.

- Concurrent calls on one calculator produce a defined error. The v1 calculator owns one internal stream and does not expose an asynchronous API.

- Model weights, lookup tables, and other structure-independent state are immutable after calculator initialization. Structure-dependent device arrays belong to a reusable execution context whose capacity grows only when required.

- CUDA Graph capture is not included in v1. A future implementation may capture stable-capacity trajectories, but must account for changing atom counts, grid counts, launch dimensions, and buffer addresses.

- `PairOp` integration uses template specialization and compile-time static polymorphism. Virtual device dispatch is not used. Concepts or traits are preferred to an inheritance requirement.

- The operation lifecycle includes a host-visible workspace plan, host-side setup, collective per-receiver preparation, repeated collective tile consumption, collective finalization, and optional host-orchestrated postprocessing.

- Model initialization and invocation setup are distinct. Immutable model allocations occur during calculator construction or loading. Invocation setup reserves or initializes context storage, including stream-ordered clearing of atomic destinations.

- Every CTA thread invokes `prepare`, every `consume`, and `finalize` in the same dynamic order. L1 code may use CUDA synchronization only when it preserves block-wide convergence. Early return or divergent block barriers are contract violations.

- The framework partitions dynamic shared memory into aligned framework storage and aligned operation scratch. L1 declares dynamic bytes and alignment through a host-callable trait or planning function. Compiled static shared memory and register use are queried for resource validation.

- Ordinary L1 device functions do not return `cudaError_t` to the host. Launch errors are checked after submission, asynchronous failures are observed at the blocking synchronization boundary, and model-detected device errors use an invocation status buffer.

- For each receiver, the framework compacts accepted edges into a shared-memory structure-of-arrays queue. The queue holds at least one full candidate batch plus `TE - 1` carried entries.

- Candidate threads calculate an acceptance predicate, use warp ballots and within-warp ranks, publish per-warp accepted counts, synchronize, and derive block-wide compacted offsets.

- After compaction, all complete `TE`-entry windows are passed to `consume`. The CTA synchronizes before callback reads and before any consumed queue storage is reused. The final remainder is moved or retained at the queue front and emitted once after candidate exhaustion.

- `TileView` exposes a valid-entry count. Entries from zero through count minus one are defined; all padded entries are unspecified. L1 must guard numerical work on invalid edge lanes while still participating in required collectives.

- L1 may not retain pointers or references into a tile after `consume` returns because the framework may immediately reuse the underlying shared memory.

- Tile fields use a structure-of-arrays layout and inline accessors. Required fields are internal sender index, three integer image shifts, displacement components, and squared distance. The receiver is implicit in CTA scheduling. Original atom indices are not duplicated into edge tiles.

- Each shared-memory region is explicitly aligned. Scalar arrays respect natural alignment and SoA region starts may use at least 16-byte alignment where useful for vector access.

- The v1 edge set is a full directed neighborhood. Both directions of a physical pair are emitted as distinct receiver-centered edges.

- The displacement convention is sender-directed: `sender position + image translation - receiver position`. The image translation is represented by an exposed integer triplet applied to the lattice.

- Edges are accepted using `r2 <= rc2`. The central self-edge with identical atom and zero image shift is excluded by identity, not by a distance epsilon. Periodic self-images are included when they lie inside the cutoff.

- Edge ordering is unspecified. Any regularity produced by sender iteration, spatial sorting, or image iteration is an implementation detail and cannot be used by L1 code.

- Pair- or species-dependent search radii are not native v1 edge semantics. An operation declares a maximum search cutoff and applies its exact pair-specific cutoff internally.

- Full directed enumeration leaves double-count policy to L1. A pair-energy operation normally applies a half factor; receiver-only message or force aggregation normally consumes the directed edge directly. Operations updating both endpoints must handle their own counting.

- Cutoff smoothness and energy/force consistency are model responsibilities. Operations advertised as conservative should make energy and the required derivatives vanish appropriately at their own cutoff.

- The brute-force backend enumerates receiver, sender, and periodic-image tuples using conservative image bounds derived from the cell and cutoff. It is the fallback for small cells or cells that do not satisfy grid requirements.

- For a lattice whose three lattice vectors are stored as rows of `H`, reciprocal plane heights are `h_i = 1 / norm(column_i(inverse(H)))`.

- A 27-offset grid stencil is valid only when each chosen bin's perpendicular height is at least the maximum search cutoff. Grid counts obey `1 <= n_i <= floor(h_i / rc)`.

- Grid counts need not use the finest valid partition. `Auto` may coarsen dimensions to keep dense-grid counts and offsets within a configured byte or grid-to-atom budget.

- Periodic grid wrapping preserves both the wrapped grid index and the integer winding shift. Different offsets that wrap to the same bin are still distinguishable by image shift.

- The dense grid is built with atom-to-bin assignment, bin counts, prefix scan, and scatter. Atoms in the same bin are contiguous; no within-bin coordinate sort is required.

- Coordinates, elements, and initial node features are physically placed in internal spatial order. All intermediate node tensors retain that internal order for the complete inference.

- Two node mappings are retained: internal-to-original and original-to-internal. A final node-level permutation restores caller order. Original indices are not required by ordinary `PairOp`s.

- `NeighborBackend` supports explicit `BruteForce`, explicit `CellList`, and `Auto` choices. Explicit grid selection validates its geometric requirements and fails cleanly if they are not met.

- `Auto` is occupancy-aware. It validates geometry, selects a memory-safe coarsened grid, builds or samples occupancy, estimates grid candidate comparisons, compares them with brute-force work, and chooses the predicted cheaper valid backend.

- Backend diagnostics expose the selected algorithm, grid dimensions, occupancy statistics, estimated candidate work, and the reason for fallback.

- The framework does not provide general reverse-mode differentiation. Each model author supplies analytical force or backward operations and is responsible for consistency with energy.

- Analytical backward operations may recompute edge quantities rather than store them. They may use atomic scatter, sender-owned gather, additional node-level intermediates, or hierarchical postprocessing according to model needs.

- Deterministic floating-point execution is not guaranteed. Fixed CTA ownership can be deterministic locally, but global atomics have unspecified ordering. Hierarchical reductions are preferred for highly contended small outputs such as virial tensors.

- L1 owns operation-specific reduction mathematics and logical global workspace layout. The framework provides ordered lifecycle phases and allocation context but does not impose a universal output-reduction descriptor in v1.

- Multi-pass operations such as conventional softmax may be expressed as multiple `PairOp`s with node-level global intermediates in v1. Future work may add persistent-CTA multi-pass traversal or online associative reductions.

- MACE MH1 is the flagship v1 integration. Its edge-local geometry, basis, radial network, tensor-product work, analytical force, and virial calculations are fused into specialized pair traversals while hidden states, messages, densities, and linear results remain node-level.

- NequIP-style models are expected to fit the same runtime grammar when their durable state is node-level.

- The framework is initially performance-tested on SM80, SM86, SM89, and SM90 hardware. SM70 and SM75 are experimental or compile-only until tested. Early releases build from source and do not require distributed fat binaries.

- Unsupported devices or operations that exceed per-block register/shared-memory limits fail through resource validation. Low occupancy alone is accepted in v1 when the kernel remains launchable and meets measured performance gates.

- The current specialized MH1 kernels are reference evidence, not guarantees for the generalized implementation. The largest forward operation uses 128 threads and approximately 50 KiB of combined shared memory; the largest backward operation uses 128 threads, approximately 31.6 KiB of static shared memory, and about 188 registers per thread on SM89. Both reach roughly two resident CTAs and eight active warps per SM on RTX 4090.

- At the specialized reference point of 12,288 atoms and 1,085,440 directed edges, measured framework-style CUDA memory is approximately 4.35 GiB: about 27 MiB of parameters and 4.32 GiB of node tensors, including about 1.67 GiB of saved forward state. Cell metadata is negligible at that case.

- The matched PyTorch memory estimate at the same reference point is approximately 49.88 GiB, dominated by approximately 46.80 GiB of edge tensors. This estimate must be validated against measured smaller cases and scaling because it is not a direct same-device peak measurement.

- The v1 capacity target is at least 50,000 atoms for a representative MH1 system on a 24-GiB RTX 4090.

- Warm steady-state generalized runtime may be at most 10% slower than the current specialized implementation.

- The matched PyTorch MACE implementation with cuEquivariance and without `torch.compile` is targeted at no more than half the framework throughput, equivalently the framework must be at least twice as fast.

- When an explicit-list CUDA backend fits in memory, implicit-neighbor execution may be at most 15% slower on the representative workload. Larger regressions trigger revision of the backend or fusion premise.

- A future restricted high-level interface will be additive above the expert `PairOp` contract. It is not part of v1 and should not destabilize validated low-level operations.

## Testing Decisions

- The primary test seam is the blocking calculator boundary exercised through ASE. Tests supply complete structures and validate public energy, force, virial, error, ordering, memory, and timing behavior. This is the highest seam that covers preprocessing, pair traversal, node operations, analytical force logic, synchronization, and output permutation together.

- One necessary lower seam validates neighbor enumeration independently. A test-only `PairOp` materializes or hashes the implicit edge multiset so that brute-force and grid results can be compared with a canonical CPU or high-precision enumerator without making global edge storage part of production execution.

- The repository currently has no existing implementation or test suite to reuse. Prior art consists of the specialized MH1 CUDA implementation, official PyTorch MACE output, float64 ports, repeated-run checks, and finite-difference force comparisons.

- A small Lennard-Jones calculator is the first conformance model. It validates edge direction, full-list counting, periodic images, central-self exclusion, cutoff behavior, energy, analytical forces, virial, and original-order restoration through the public seam.

- Complete MH1 inference is the performance and realistic-correctness model. It validates every node-only and pairwise stage described by the reference implementation, including ZBL, both message layers, readouts, analytical backward passes, force accumulation, and virial.

- Neighbor tests compare edge multisets rather than edge order. They include orthogonal and triclinic cells, one- and two-bin axes, wrapped bins mapping to the same physical bin with different image shifts, periodic self-images, duplicate prevention, high occupancy, sparse cells, coarsened grids, and brute-force fallback.

- Away from an explicitly documented floating-point boundary band, grid and brute-force backends must enumerate the same directed `(receiver, sender, image)` multiset. Both backends use the same displacement and squared-distance acceptance routine to minimize boundary differences.

- Invalid-lane tests use operations that would produce non-finite values if padded geometry were consumed. They verify that `TileView::size` is respected while all threads still reach collectives.

- Queue tests cover zero accepted candidates, exact full tiles, one-entry remainder, `TE - 1` remainder, candidate batches larger than `TE`, and the maximum case of one carried remainder plus a fully accepted candidate batch.

- Collective-safety tests run under CUDA checking tools and exercise divergent feature validity, padded edge lanes, repeated callback invocations, and scratch-region boundaries.

- Resource tests query compiled kernel attributes and device limits before launch. They verify declared dynamic scratch, compiled static shared memory, register usage, and supported block dimensions for each tested architecture.

- Public API tests verify blocking completion, result-buffer ownership, sequential reuse, grow-only capacity, output order, defined concurrent-call rejection, allocation failures, invalid-cell failures, and excessive-resource failures.

- Model-level numerical tests declare units, norms, absolute tolerances, and relative tolerances. Relative error is used away from zero; model-specific absolute tolerances govern near-zero energy, force, and virial values.

- Default MH1 cross-implementation targets are approximately `1e-5` for energy per atom and `1e-4` for force and virial under the declared comparison norms. These are model validation defaults rather than universal framework guarantees.

- Repeated-run variability targets are approximately `1e-6` for energy and `1e-5` for force and virial. Atomic accumulation remains nondeterministic, so tests validate a tolerance rather than bit identity.

- Equivariance and invariance tests cover atom permutation, translation, lattice-equivalent wrapping, and rotation. For models that claim these properties, end-to-end energy and force discrepancies target approximately `1e-4`.

- Analytical force tests use central finite differences with a step-size sweep. The provisional force agreement target is `1e-3`, acknowledging float32 cancellation, potential-surface curvature, and runtime accumulation variability.

- Float64 diagnostic builds compare continuous recomputation and CUDA/PyTorch outputs. Existing evidence of roughly 12–13 matching digits is retained as a debugging reference, not a v1 float64 product promise.

- Cutoff tests sample edges immediately below, at, and above the search and operation-specific cutoffs. Conservative model tests verify the continuity order claimed by that model.

- Closed-loop MD tests measure stability, physical plausibility, and long-run energy drift. They are not used for direct implementation speed ratios because trajectories can diverge from small force differences.

- Authoritative performance tests replay an identical pre-recorded sequence of changing coordinates and cells through every implementation. Atom counts, edge counts, model weights, dtype, outputs, and frontend behavior are matched.

- Benchmarks include host-to-device input movement, spatial preprocessing, permutation, neighbor discovery, pair and node computation, analytical force calculation, synchronization, and device-to-host output movement for all implementations.

- Each benchmark performs 100 warm-up steps followed by 100 measurement windows of 100 steps each. Window distributions, central tendency, and tail latency are reported; every timed blocking call ends only after results are host-valid.

- Both wall-clock end-to-end latency and device-side CUDA timing are reported. End-to-end release gates use wall-clock blocking latency.

- Benchmarks cover small, medium, and approximately 50,000-atom cases and report actual directed edge counts. Atom and edge scaling are analyzed separately.

- If a baseline cannot fit the 50,000-atom case, the OOM is reported explicitly. Comparisons then use the largest matched cases plus fitted fixed, per-node, and per-edge scaling; extrapolation is labeled as such.

- Peak memory is measured both from implementation-side allocation accounting and from a process-level GPU observation tool. Tensor-size estimates, live allocated bytes, allocator-reserved bytes, and total process memory are not conflated.

- Benchmark reproducibility requires fixed MPTraj revision and structure identifiers, deterministic supercell construction and perturbation seeds, model-weight checksum, container definition, CUDA toolkit, driver, compiler, PyTorch, cuEquivariance, ASE, and GPU model.

- Implementation ordering is interleaved or randomized during comparative runs to reduce thermal, boost-clock, and long-run system drift.

## Out of Scope

- General reverse-mode automatic differentiation.
- Framework-generated force kernels or guarantees that arbitrary model outputs are conservative.
- A high-level network-author API that hides CUDA scheduling and collective behavior.
- Persistent global edge features or mutable edge embeddings.
- Edge-to-edge and triplet message-passing architectures such as DimeNet and GemNet.
- Long-range electrostatics based on Ewald summation or particle-mesh methods.
- Drude, induced-dipole, and other nonlocal iterative polarization solvers.
- Heterogeneous batching of multiple periodic structures.
- Partially periodic or nonperiodic boundary conditions.
- Asynchronous public calculator calls, caller-provided CUDA streams, and overlapping calls on one calculator.
- Deterministic global reductions.
- CUDA Graph capture in v1.
- Runtime NVRTC compilation or device-function-pointer dispatch.
- Distributed multi-GPU execution.
- Stable precompiled fat-binary distribution for all CUDA architectures.
- Architecture-specific optimization beyond the initially tested SM80, SM86, SM89, and SM90 devices.
- A mandatory explicit-neighbor-list production backend in the first release, although matched explicit-list code remains a benchmark and potential future adaptive backend.

## Further Notes

### Domain vocabulary

- **L0 user:** A calculator consumer who supplies structures and receives model outputs.
- **L1 developer:** An expert CUDA model developer who implements `PairOp` mathematics, resources, synchronization, reductions, and analytical force logic.
- **Framework developer:** A runtime developer responsible for periodic candidate enumeration, grid preprocessing, queue compaction, tile lifetime, launch orchestration, and public ordering.
- **Receiver:** The atom whose node-level result is owned or partially accumulated by the current CTA.
- **Sender:** A periodic neighbor contributing to the receiver.
- **Candidate:** A sender/image tuple considered before the cutoff test.
- **Accepted edge:** A candidate satisfying the framework search cutoff.
- **Accepted-edge queue:** The compact shared-memory SoA buffer holding accepted edges until consumption.
- **Edge tile:** A contiguous window of up to `TE` accepted edges passed collectively to a `PairOp`.
- **Execution context:** Reusable structure-dependent buffers and status associated with the calculator's single in-flight invocation.
- **Explicit-list backend:** A comparison or future backend that materializes edge connectivity globally and reuses it.
- **Implicit-neighbor execution:** The production strategy that reconstructs accepted edge tiles during each pair traversal.

### Supported model shape

MH1 demonstrates the intended shape: node embedding; fused ZBL pair computation; node linear-up transforms; two fused message traversals; node-only interaction, gating, product, skip, and readout operations; then model-specific analytical backward traversals that recompute transient edge quantities and accumulate forces and virial. Node states such as hidden features, linear-up results, densities, messages, and adjoints remain global. Geometry, bases, radial-network intermediates, path weights, tensor-product edge terms, and edge derivatives remain transient.

### Product positioning

The v1 name and documentation should emphasize elimination of global edge materialization, not claim that all neighbor-related global storage disappears. Coordinates, permutations, grid counts, offsets, atom ordering, model state, and node intermediates remain in global memory. The cell index may be reused across pair traversals within one calculation, but accepted neighbors are rediscovered for each traversal.

### Pivot condition

The implicit-neighbor premise is not ideological. If matched benchmarks show more than 15% slowdown against an explicit-list CUDA implementation in memory-feasible regimes, or more than 10% abstraction overhead against the specialized implicit implementation, the design must add an adaptive explicit backend, revise tile reconstruction, or narrow its performance claims.

### Publication status

This specification was synthesized before the workspace had a populated Git repository, project issue tracker, ADRs, domain glossary, or triage-label configuration. It therefore establishes initial terminology rather than adapting existing repository vocabulary. Issue publication and the required `ready-for-agent` label remain pending project tracker setup.
