# es31-atomics

**Minimum:** OpenGL ES 3.1 · **Shaders:** GLSL ES 3.10 (`#version 310 es`) · **Run:** `es31-atomics [--headless --validate]`

## Purpose

Thousands of shader invocations run at once and are free to run in any order. When they must update shared memory
without losing updates, the memory must be changed with an **atomic** operation: a read-modify-write that no other atomic
operation can interleave with. OpenGL ES 3.1 offers two families, both shown here under heavy contention — 65536
invocations that all hit the same few words — with results that are known *exactly* in advance:

- **atomic counters** (`atomic_uint`, `atomicCounterIncrement/Decrement/atomicCounter`; GLSL ES 3.10 §4.4.6, §8.10): opaque
  handles for 32-bit words in an `ATOMIC_COUNTER_BUFFER`, bound with `layout(binding = N, offset = M)` (ES 3.2 §7.7);
- **atomic memory functions** (`atomicAdd/Min/Max/And/Or/Xor/Exchange/CompSwap`; GLSL ES 3.10 §8.11) on **buffer variables**
  and on **`shared`** variables.

Correctness is judged by *order-independent* properties: the final value of a commutative operation, or a SET property
of what the invocations received. Which invocation got which ticket, who won a compare-and-swap, or in which order a
stream compaction appended its elements is the scheduler's business and is never asserted.

## Reading the image

The 256 × 256 surface is drawn entirely by shaders that **read what the atomics produced**.

- **Top strip (16 px).** One 16 px cell per accumulator of the atomic-memory test. A fragment shader compares the word in the
  SSBO with the value the CPU computed and paints the cell **green** (equal) or **red**. Twelve cells are used (add, signed
  add, min, max, signed min, signed max, and, or, xor, add of 2³²−3, compare-and-swap loop, ticket); the other four stay dark.
- **Middle (112 px): a histogram.** 16 bars built with `atomicAdd(hist[bin], 1u)` by 65536 invocations, `bin = ((h mod 16) · ((h >> 7) mod 16)) / 15`,
  which makes the counts fall off quickly (bin 0: 18374 of 65536 → 100 px; bin 15 is about one pixel). A fragment shader draws the bars.
- **Bottom (two 128 × 128 panels).** The **pixels of a ring pattern that survived a stream compaction.** Element *i* of the input
  is the squared distance of pixel (*i* mod 128, *i* / 128) from the centre; the predicate keeps the even rings
  (`(d / 300) & 1 == 0`, 8537 of 16384 elements). The compute shader appends `(index, value)` to a dense list at a slot handed
  out by an atomic append pointer; a second dispatch scatters the list back into a mask, and a fragment shader draws it.
  **Left:** the append pointer is an **atomic counter** (`atomicCounterIncrement`). **Right:** it is `atomicAdd` on a buffer
  variable. The panels are identical although the lists are ordered completely differently.

## Atomic counters

`counters.comp` declares eight counters in **one** buffer, binding point 0 — exactly the guaranteed minimums
(`MAX_COMPUTE_ATOMIC_COUNTERS` = 8, `MAX_ATOMIC_COUNTER_BUFFER_SIZE` = 32 bytes):

| Offset | Counter | Per invocation | Initial value (glBufferSubData) | Final value after N = 65536 |
|---|---|---|---|---|
| 0 | `c_ticket` | `atomicCounterIncrement` (return value = ticket) | 5000 | 5000 + N |
| 4 | `c_inc` | 16 × increment | 1000 | 1000 + 16 N = 1049576 |
| 8 | `c_dec` | `atomicCounterDecrement` (returns the *new* value) | 1000000 | 1000000 − N |
| 12 | `c_wrap` | increment | `0xFFFFFFF0` | wraps: (0xFFFFFFF0 + N) mod 2³² = N − 16 |
| 16 | `c_updown` | +1 then −1 | 5 | 5 (net zero) |
| 20 | `c_odd` | `(i & 1)` → increment | 0 | N / 2 (its `offset` is *inherited*: 16 + 4) |
| 24, 28 | `c_hist[2]` | `[0]` once, `[1]` twice | 0, 100 | N, 100 + 2 N (an **array** of counters, stride 4) |

Return values are recorded too. `atomicCounterIncrement` returns the value *before* the increment, `atomicCounterDecrement`
the value *after* the decrement; so across all invocations every ticket value `5000 … 5000+N−1`, every decrement result and every
wrapped value must be handed out **exactly once**. The shader keeps a histogram of the returned values in an SSBO with
`atomicAdd`, and the application checks that all 3 × 65536 entries are exactly 1. A second dispatch, separated by
`glMemoryBarrier(ATOMIC_COUNTER_BARRIER_BIT)`, only *reads* with `atomicCounter()`: all 65536 invocations must see the
same final `c_inc`, and eight more reads return the final value of each counter.

The counters are initialised with **`glBufferSubData`** into a buffer created with `glBufferData`. A second run (1024
invocations) puts the same counters at **byte offset 16 of a 64-byte buffer** and binds them with
`glBindBufferRange(ATOMIC_COUNTER_BUFFER, 0, buf, 16, 32)`; the 16 canary bytes before and after must be untouched, showing that
`offset` in the shader is relative to the start of the bound range. `glBindBufferRange` on this target needs an offset that is a
multiple of 4 (Table 6.5): offset 2 → `INVALID_VALUE`.

### The atomic counter program interface

ES has no `glGetActiveAtomicCounterBufferiv` (that is desktop GL). In ES the same information comes from
`glGetProgramResourceiv`/`glGetProgramInterfaceiv`:

- `glGetProgramiv(ACTIVE_ATOMIC_COUNTER_BUFFERS)` = 1; the `ATOMIC_COUNTER_BUFFER` interface has one active resource with
  `BUFFER_BINDING` 0, `NUM_ACTIVE_VARIABLES` 7 (the array `c_hist` is one variable), `BUFFER_DATA_SIZE` ≥ 32 (32 here),
  `ACTIVE_VARIABLES` = the indices of the seven counters, `REFERENCED_BY_COMPUTE_SHADER` 1;
- each counter in the `UNIFORM` interface: `TYPE` `UNSIGNED_INT_ATOMIC_COUNTER`, `ATOMIC_COUNTER_BUFFER_INDEX` 0,
  `OFFSET` as declared (including the inherited 20), `ARRAY_SIZE`/`ARRAY_STRIDE` (2 and "always 4", §7.7.1) for `c_hist`,
  `BLOCK_INDEX` −1, `LOCATION` −1; `glGetUniformLocation("c_odd")` is −1 — atomic counters are addressed by binding and
  offset, not by location.

## Atomic memory functions

`memory.comp` has 65536 invocations (512 work groups of 128) apply each operation to one word with data derived from the
invocation index by a hash (`lowbias32`, computed identically on the CPU):

| Operation | Data | Check |
|---|---|---|
| `atomicAdd(uint)`, `atomicAdd(int)` | `h % 1000`, `h % 2001 − 1000` | exact sum (wraps modulo 2³²) |
| `atomicMin/Max (uint, int)` | `h`, `int(h)` | exact extremum (signed and unsigned order differ) |
| `atomicAnd` / `atomicOr` / `atomicXor` | `~(1 << (4 + …))`, `1 << (…)`, `h` | exact fold |
| `atomicAdd(0u − 3u)` | wrap-around | exact: 100 − 3 N mod 2³² |
| `atomicAdd(hist[bin], 1)` | skewed bin | exact histogram, sums to N |
| `atomicAdd(ticket, 1)` | 1 | each of 0 … N−1 returned once (a histogram of ones) |
| `atomicExchange(word, id + 1)` | id | **set**: the values received by all invocations plus the final value are exactly the initial value and all `id + 1`, each once |
| `atomicCompSwap(owner, 0, id + 1)` | id | **set**: exactly one invocation receives 0, its id + 1 is the final value, all others received that value |
| `atomicCompSwap` loop (1024 invocations) | read–publish `old + 1`–retry | exact count |

Shared variables: every work group initialises `shared` words, calls `memoryBarrierShared(); barrier();` (GLSL ES 3.10 §8.14:
`barrier()` alone does *not* order memory), applies `atomicAdd/Max/Or/Xor/Min/And` and a `atomicCompSwap` ownership claim, and
after another `memoryBarrierShared(); barrier();` invocation 0 stores the results. The application recomputes all 512 groups
on the CPU; every group must have exactly one compare-and-swap winner.

A small extra shader, `racy.comp`, does 65536 *non-atomic* increments of one word next to 65536 `atomicAdd`s and **prints** the
two results (between a handful and a few thousand of the plain increments survive depending on the driver, all of the atomic ones do). This is an illustration, not a check:
the outcome of a data race is not defined by the specification.

## Stream compaction

`compact_counter.comp` and `compact_ssbo.comp` differ in one line: `slot = atomicCounterIncrement(c_append)` versus
`slot = atomicAdd(dst.n, 1u)`. The application validates each result as a **set**: the final count equals the number of
elements the predicate keeps, the indices in `idx[0 .. count)` sorted equal the CPU's ascending list (each index once), every
`val[slot]` belongs to `idx[slot]`, all slots at or beyond the count still hold the sentinel (nothing was appended past the
end), and the mask scattered back from the list equals the predicate image at all 16384 elements. How many slots happen to
hold the element a sequential compaction would have put there is printed (0 to a few hundred of 8537): the order is up to the
scheduler.

Fragment-stage atomic counters are used too, when `MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS`/`COUNTERS` allow it: while drawing each
panel, `panel.frag` increments one counter for every fragment and another for every lit pixel. Without multisampling a
fullscreen triangle runs exactly one fragment-shader invocation per pixel (ES 3.2 §7.11.1), so the counters must be exactly
128 × 128 and 8537.

## How it is validated

- **Counters**: 8 final values, wrap-around, the three "each value returned once" histograms, the `atomicCounter()` read
  phase, and the whole sequence again in a 16-byte-offset range with canary words.
- **Memory atomics**: 12 exact final values, the 16-bin histogram and its total, the ticket histogram, the exchange and
  compare-and-swap set properties, 512 per-group shared-variable results plus the 512 compare-and-swap winners.
- **Compaction**: count, set, value consistency, untouched tail, scatter mask — for both append pointers.
- **Pixels**: the 16 strip cells (12 green, 4 dark), the 16 histogram bars (inside and above the bar), and **all 2 × 16384
  panel pixels**, plus the two fragment-counter pairs.
- **State and interface**: `ATOMIC_COUNTER_BUFFER_BINDING/START/SIZE` after `glBindBufferBase/Range`; the interface queries above.
- **Errors**: index = `MAX_ATOMIC_COUNTER_BUFFER_BINDINGS`, offset 2, size 0 (`INVALID_VALUE`, state unchanged).
- **Layout rules**: a counter at binding `MAX_ATOMIC_COUNTER_BUFFER_BINDINGS − 1` links and one at `MAX_…` is rejected; as many
  bindings as `min(MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS, MAX_ATOMIC_COUNTER_BUFFER_BINDINGS)` link; two counters at the same
  binding and offset, `offset` without `binding`, and an `atomic_uint` inside a structure are rejected.
- **Negative controls** (the set checks must be able to fail): a ticket histogram with one duplicate and one missing ticket,
  an exchange log with a duplicated value, a compare-and-swap log with two winners, a compacted list with one slot
  overwritten, and an add result off by one are each rejected by the same predicates that accept the genuine data.

## Pitfalls

- `atomicCounterDecrement` returns the value **after** the decrement, `atomicCounterIncrement` the value **before**.
- Counters wrap modulo 2³² silently; start `c_wrap` near the top to see it.
- A counter's `offset` is relative to the start of the *bound range*; its binding is fixed by the shader (no GL call changes it).
- `barrier()` is an execution barrier only; shared-memory communication needs `memoryBarrierShared()` too.
- Atomic counter arrays may only be indexed with dynamically uniform expressions (GLSL ES 3.10 §4.1.7.3); the loop counters
  of `counters.comp` qualify.
- After a dispatch that updates counters or SSBOs, `glMemoryBarrier` with `ATOMIC_COUNTER_BARRIER_BIT`,
  `SHADER_STORAGE_BARRIER_BIT` or `BUFFER_UPDATE_BARRIER_BIT` is needed before the next consumer (a dispatch that reads the
  counters, a draw that reads the SSBO, a buffer mapping). The tests do not claim the barrier is *necessary*.
- Optional stages: ES 3.1 allows zero vertex/fragment shader storage blocks and zero fragment atomic counters. The demo queries the
  limits and prints a note when it has to skip the fragment-shader displays or the fragment counting.

## Atomic counters need a binding

`uniform atomic_uint a;` without a `binding` is a compile-time error, and the demo checks that it is rejected. GLSL ES 3.10
§4.4.5 reads as if it were allowed (*"Any uniform sampler, image or atomic counter variable declared without a binding
qualifier is initially bound to unit zero"*), and an earlier version of this demo expected it to compile, with an XFAIL on
Mesa. Mesa and ANGLE, two independent compilers, both reject it, which prompted a re-read: GLSL ES 3.10 §4.1.8 already
forbids `atomic_uint` in structures *"since there is no mechanism to specify the binding"*, and GLSL ES 3.20 revision 6
("Clarify error conditions when declaring atomic counters") states in §4.4.6: *"It is a compile-time error if any such
declaration does not include a binding layout qualifier."* The demo follows the clarified rule on every driver; all its
own shaders declare a binding.
