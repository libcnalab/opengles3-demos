# es31-compute-shared-memory

**Minimum:** OpenGL ES 3.1 · **Shaders:** GLSL ES 3.10 · **Run:** `es31-compute-shared-memory [--headless --validate]`

## Purpose

Work groups are not just a way to carve up a dispatch. The invocations of one work group can **cooperate**: they share a small
block of fast on-chip memory (`shared` variables) and can wait for each other (`barrier()`). That is what makes algorithms such as
reductions, prefix sums, sorting networks and tile-based convolution possible in a compute shader, and it is also where most compute
bugs live, because the moment invocations read what other invocations wrote, *order* matters, and the GL gives you no order unless you
ask for one.

The demo implements the two textbook shared-memory algorithms and checks every output value exactly:

1. **Reduction.** A work group of 128 invocations folds 128 integers into their sum, minimum and maximum in seven halving steps. A
   second dispatch of one work group reduces the 24 partial results of the first pass (3000 inputs = 23 full groups + 56 elements) to
   the total of the whole array.
2. **Prefix sum (scan).** Hillis-Steele inclusive scans of each work group, a scan of the group totals, and a pass that adds the
   offsets: together an inclusive scan of all 3000 elements. A work-efficient Blelloch *exclusive* scan (up-sweep, clear the root,
   down-sweep; 64 invocations handling 128 elements) runs next to it.

It then queries `MAX_COMPUTE_SHARED_MEMORY_SIZE` and compiles a shader whose shared array is exactly that size, and it shows, with a
CPU simulation, why the barriers in those algorithms are not decoration.

## What the image shows

Four horizontal bands, drawn from the very buffers the compute shaders wrote (the SSBOs are bound as vertex attribute arrays after
`glMemoryBarrier(VERTEX_ATTRIB_ARRAY_BARRIER_BIT)`; each value is one instanced bar). A thin gray line marks the value 0 in each band.

| band | color | content |
|---|---|---|
| 1 | blue | the first 128 input values, `[-500, 499]`, one 2-pixel bar each |
| 2 | green | the **inclusive** scan of those 128 values (Hillis-Steele): a random walk |
| 3 | orange | the **exclusive** scan of the same values (Blelloch): the same walk shifted by one bar, starting at 0 |
| 4 | yellow, red, purple | for each of the 24 work groups: the maximum (up), the sum, the minimum (down) of its inputs |

Each band is scaled to its own largest magnitude, so the bars reach about 29 of the 31 available pixels.

## Central calls

| Call | Role |
|---|---|
| `glDispatchCompute(groups, 1, 1)` | one dispatch per pass: 24 groups for the first level, 1 group for the second level |
| `glMemoryBarrier(SHADER_STORAGE_BARRIER_BIT)` | between dependent dispatches: the second pass reads what the first one stored |
| `glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT)` | before `glMapBufferRange` reads a result |
| `glMemoryBarrier(VERTEX_ATTRIB_ARRAY_BARRIER_BIT)` | before the same buffers are used as vertex attributes |
| `glGetIntegerv(MAX_COMPUTE_SHARED_MEMORY_SIZE)` | the limit, in basic machine units (bytes) |

Inside the shaders: `shared`, `barrier()`, `memoryBarrierShared()`, `groupMemoryBarrier()`, `memoryBarrier()`.

## Spec

- ES 3.2 **§17 Compute Shaders** and **§17.1 Compute Shader Variables**: "There is a limit to the total size of all variables declared
  as shared in a single program object. This limit, expressed in units of basic machine units, may be queried as the value of
  MAX_COMPUTE_SHARED_MEMORY_SIZE. The total size required for a given program object is implementation-dependent. However, this total
  size may not exceed the largest block size that would be obtained if all active variables declared as shared were instead declared as
  arbitrarily ordered members of a shader storage block qualified with std430." Table 21.49: minimum 16384.
- ES 3.2 **§7.11.1** Shader Memory Access Ordering and **§7.11.2** Shader Memory Access Synchronization (the `glMemoryBarrier` bits).
- GLSL ES 3.10 **§4.3.8 Shared Variables**: shared variables are "implicitly coherent", "may not have initializers and their contents are
  undefined at the beginning of shader execution"; "any data written to shared variables will be visible to other work items ... within the
  same local work group"; "in the absence of synchronization, the order of reads and writes to the same shared variable by different
  invocations ... is not defined"; ordering needs "a combination of control flow and memory barriers".
- GLSL ES 3.10 **§8.14 Shader Invocation Control Functions** (GLSL ES 3.20 numbers it §8.15): `barrier()` — "all invocations for a single work
  group must enter it before any will continue beyond it"; but it "only affects control flow and does not by itself synchronize memory
  accesses. In particular, it does not ensure that values written by one invocation prior to a given static instance of barrier() can be
  safely read by other invocations after their call to the same static instance of barrier(). To achieve this requires the use of both
  barrier() and a memory barrier." In compute shaders it may sit in control flow only if that control flow is *uniform*: "Otherwise, some
  shader invocations will stall indefinitely".
- GLSL ES 3.10 **§8.15 Shader Memory Control Functions** (3.20: §8.16): `memoryBarrierShared()` orders accesses to shared variables
  "as viewed by other invocations in the same work group"; `groupMemoryBarrier()` does the same for all memory types, within the work group;
  `memoryBarrier()` for all memory types, for all invocations. All three are available in compute shaders.

## Shader concepts

- **Two different synchronizations, always used as a pair.** `barrier()` orders *execution*: nobody leaves it before everybody has arrived.
  The memory barrier orders *memory*: a store done before it is visible, to the other invocations of the work group, before any access done
  after it. A step that writes shared memory and a following step that reads what *other* invocations wrote therefore needs
  `memoryBarrierShared(); barrier();` between them: the barrier guarantees the other invocation has executed its store at all, the memory
  barrier guarantees the store is visible. (Not to be confused with the API call `glMemoryBarrier`, which orders shader writes of one
  *dispatch* against later consumers.)
- **`reduce.comp`** (`#define PASS 1|2`, `#define SHARED_FENCE ...` inserted by the host): load phase, then seven folding steps
  (`stride = 64 ... 1`), each followed by fence + barrier. Invocations past the end of the input contribute the identity element (0, `INT_MAX`,
  `INT_MIN`). The loop bound is the same for every invocation, so the `barrier()` inside the loop is in uniform control flow. Pass 2 reads the
  `ivec4` partials of pass 1. Three programs differ only in the fence: `memoryBarrierShared()`, `groupMemoryBarrier()`, `memoryBarrier()`.
- **`scan_hillis_steele.comp`**: log2(128) = 7 steps in which every element adds the element `2^k` places to its left. In place, every step has *two*
  hazards: a **write-after-read** (an invocation must not overwrite `s[i]` before its right neighbour has read it) and a **read-after-write** (the
  next step must see every new value), hence two fence+barrier pairs per step.
- **`scan_blelloch.comp`**: an up-sweep builds partial sums in a binary tree, thread 0 records the group total and zeroes the root, a down-sweep
  distributes the sums. Each tree level starts with a fence + barrier because it reads nodes the previous level wrote.
- **`scan_add.comp`**: no shared memory; each invocation touches only its own element.
- **`shared_max.comp`** (`#define SHARED_WORDS <limit / 4>`): fills a `shared uint s[SHARED_WORDS]` with a pattern that depends on word *and work group*, then
  each invocation copies out words that *another* invocation wrote (the array is read backwards: invocation `lid` reads what invocation
  `127 - lid` wrote). 4 work groups; no two groups may share storage.

## How it is validated

All comparisons are exact: the values are integers and the arithmetic is `int`/`uint`, so there is nothing to round.

- **Reduction.** Sum, minimum, maximum and input count of all 24 work groups (including the 56-element last one) equal the CPU values; the guard record
  after them is untouched (buffers are pre-filled with `0xDEADBEEF`). Pass 2 gives the array's total (for the default seed: sum 2882, min -500, max 499).
  The identical result comes out with `groupMemoryBarrier()` and with `memoryBarrier()` as the fence.
  *Negative control:* a CPU model of the same input with element 1234 changed by +1 differs from the GPU result in **exactly one** work group (work group 9, which holds elements 1152-1279).
- **Scans.** Per-group Hillis-Steele inclusive scans, the 24 group totals, the second-level scan of those totals (the same program, 24 inputs and one work
  group), and the final three-pass result against `std::inclusive_scan` of the whole array; the Blelloch exclusive scan and its totals against a CPU
  exclusive scan; and two GPU-only cross-checks that need no CPU model: the last element of the full scan equals the reduction's sum, and
  `Hillis-Steele inclusive == Blelloch exclusive + input` for every element. *Negative controls:* the three-pass result is not the per-group scan (the
  offsets matter), and the inclusive result is not the exclusive model.
- **Why barriers.** A CPU simulation executes the 128 invocations of a work group strictly one after the other in a chosen order, *without* barriers or
  fences, which is one of the schedules the specification allows. For the scan, running the invocations highest first happens to give the right answer, while
  running them lowest first (each read of `s[lid - offset]` now sees an already updated value) gives a wrong scan; for the reduction, highest first is right,
  and lowest first is wrong because invocation 0 folds before invocation 64 has stored its value. The demo asserts these four CPU facts. It **never**
  asserts a wrong GPU result without the barriers: such a program has an undefined result, and a given driver may well produce the right one.
- **Shared memory limit.** `MAX_COMPUTE_SHARED_MEMORY_SIZE >= 16384`; a shader declaring `shared uint s[limit / 4]` — exactly the limit — compiles,
  links and, dispatched as 4 work groups, returns every one of `4 x limit / 4` words exactly. In the reference runs the limit is 65536 bytes on radeonsi and
  32768 on llvmpipe and zink. Information only (not asserted, the specification does not require an error): one more word fails to link on all three Mesa drivers
  with `Too much shared memory used`.
- **Image.** The full framebuffer is read once. For every bar with a visible height (at least 6 px) two pixels are probed in its first pixel column: 2-3
  pixels *inside* the tip must have the series' exact color, and 2-3 pixels *beyond* the tip (if still inside the band) must have the band color. The tip is
  computed on the CPU from the CPU model of the value, so the probes test the compute-to-render path independently of the buffer readback. 106 + 96 + 95 + 16 + 24 + 24
  bars are probed (the short ones are skipped). *Negative control:* the same probes applied to the input band with the *neighbouring* bar's value fail for most
  bars (94 of 106).
- **Not exercised.** `memoryBarrierBuffer()`, `memoryBarrierImage()` and `memoryBarrierAtomicCounter()` need buffers/images/atomics written by other
  invocations; they belong to the atomics and image demos. `glMemoryBarrier` itself: exact results after the documented barriers are verified, but a missing
  barrier cannot be shown to fail, because implementations may synchronize more than the minimum.

## Run

```
./build/debug/bin/es31-compute-shared-memory --headless --validate --capture compute-shared-memory.png
```

## Pitfalls

- `barrier()` alone does not make shared memory visible, and `memoryBarrierShared()` alone does not make anyone wait. You need both. A shader
  that works with `barrier()` alone on one GPU relies on an implementation detail that the specification does not promise.
- Every `barrier()` must be reached by *all* invocations of the work group, in uniform control flow. A barrier inside `if (lid < stride)` hangs or
  corrupts; put the `if` around the *work* and the barrier after it, as the shaders do. Early `return` for out-of-range invocations *before* a barrier
  has the same problem, which is why the shaders pad with an identity element instead of returning.
- A hazard is not only read-after-write. Overwriting a value that a neighbour has not read yet (write-after-read) needs a barrier too: the Hillis-Steele scan has
  two per step for that reason.
- Shared memory is **uninitialized** at the start of every work group and is private to it; work groups cannot communicate through it, and nothing
  survives the dispatch.
- Pad partial work groups with the identity of the operation (0 for a sum, `INT_MAX` for a minimum, `INT_MIN` for a maximum); do not leave stale
  values in shared memory.
- The shared memory limit applies to the *total* of all `shared` variables in the program and is in bytes. The minimum guaranteed by ES is only 16 KiB (the
  GL implementations here offer 32 KiB or 64 KiB); a portable shader must stay below 16384 bytes or query the limit. Large shared arrays typically also reduce how many work groups
  can run at the same time on a compute unit.
- The tree algorithms here assume a work group size that is a power of two and fixed at 128 (or 64 x 2 for Blelloch); that size is a compile-time constant,
  so it is also baked into the host code.
- Under `MESA_GLES_VERSION_OVERRIDE=3.1` (the `es-min` runs) only the reported version changes; the context is still Mesa's ES 3.2 driver.
