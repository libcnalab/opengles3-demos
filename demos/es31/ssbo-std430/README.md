# es31-ssbo-std430

**Minimum:** OpenGL ES 3.1 · **Shaders:** GLSL ES 3.10 (`#version 310 es`) · **Run:** `es31-ssbo-std430 [--headless --validate]`

## Purpose

A uniform block is read-only and small. A **shader storage block** is a block of `buffer` variables that shaders read
*and write*, that can be as large as `MAX_SHADER_STORAGE_BLOCK_SIZE` (at least 128 MiB), and whose last member may be an
array whose length nobody knows at compile time. Because the application and the shader must agree on where every
member lives in the buffer, the demo concentrates on the memory layout and on how the GL reports it:

- the **std430** packing rules next to **std140** (ES 3.2 §7.6.2.2), predicted by a CPU layout calculator
  (`measure_basic`, `measure_struct`, `place_fields` in `main.cpp`) that is independent of the GL;
- the same 16-member block moved in **both directions**: data packed by the calculator is read by a compute shader
  (one flag per member), and data written by the compute shader is read back with `glMapBufferRange` and compared with
  the calculator's image;
- what happens when the layouts are **confused** (std140 data read through a std430 block and the reverse): the effect on
  every member is predicted exactly, and the shader agrees;
- **introspection** with `glGetProgramResourceiv` on the `BUFFER_VARIABLE` and `SHADER_STORAGE_BLOCK` interfaces
  (`OFFSET`, `ARRAY_STRIDE`, `MATRIX_STRIDE`, `IS_ROW_MAJOR`, `TOP_LEVEL_ARRAY_SIZE`, `TOP_LEVEL_ARRAY_STRIDE`, ...);
- **runtime-sized arrays** (`float values[];`) whose `length()` comes from the size of the range bound with
  `glBindBufferRange`, at offsets that are multiples of `SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT`;
- the indexed **binding state** (`SHADER_STORAGE_BUFFER_BINDING/START/SIZE`), the **limits**, link-time limit
  experiments, the defined **errors**, and `glMemoryBarrier` before every consumer of the compute shader's stores
  (buffer mapping, a fragment shader, a vertex shader).

## Reading the image

The 256 × 256 surface has two parts.

**Top, 64 px: the verification grid.** Four rows of 16 cells (cell *m* = member *m* of block `Probe`), drawn by a
**fragment** shader that reads the flag buffer the compute shader wrote (`readonly buffer Flags`):

| Row | Block declared | Buffer packed with | Expected |
|---|---|---|---|
| 0 | `std430` | the std430 calculator | all 16 cells **green** |
| 1 | `std140` | the std140 calculator | all 16 cells **green** |
| 2 | `std430` | the **std140** layout (a mistake) | members 1-4 green, 5-16 **magenta** |
| 3 | `std140` | the **std430** layout (a mistake) | members 1-4 green, 5-16 **magenta** |

Magenta means "the flag is 0 *as the calculator predicted*"; **red** anywhere would mean the flag differs from the
prediction. Members 1-4 (`f`, `v3`, `after_v3`, `v2`) come before the first array, so both layouts put them at the
same offsets; every member from `farr[3]` on moves.

**Below, three 48 px bands: bar charts** drawn by a **vertex** shader that reads `values[]`, the runtime-sized array the
compute shader filled. The bands show buffer ranges holding 32, 16 and 4 elements; the bar width is `2.0 /
values.length()` computed in the vertex shader, so the bars tile the width only if `length()` is right. The bar for
element *i* is `1 + (7 i) mod 13` units high, coloured by its value. The lower 48 px are intentionally empty.

## std430 versus std140

ES 3.2 §7.6.2.2: shader storage blocks may use `std140` and `std430`. The two are identical except that std430 does **not**
round up (to a `vec4`) the base alignment and stride of **arrays of scalars and vectors** (rule 4) and the base alignment
of **structures** (rule 9). Matrices follow rule 4, so the same applies to their column (row) stride. Measured on the
block `Probe` (offsets and strides in bytes; "stride" is the array stride, the matrix stride, or for `pairs` the
structure stride):

| Member | std430 offset / stride | std140 offset / stride |
|---|---|---|
| `f`, `v3`, `after_v3`, `v2` | 0, 16, 28, 32 | 0, 16, 28, 32 |
| `float farr[3]` | 40 / 4 | 48 / 16 |
| `vec2 v2arr[3]` | 56 / 8 | 96 / 16 |
| `vec3 v3arr[2]` | 80 / 16 | 144 / 16 |
| `int i`, `uvec2 uv2` | 112, 120 | 176, 184 |
| `mat2 m2` | 128 / 8 | 192 / 16 |
| `mat3 m3` | 144 / 16 | 224 / 16 |
| `mat3x2 m3x2` (three `vec2` columns) | 192 / 8 | 272 / 16 |
| `row_major mat2x3 m2x3r` (three rows of two) | 216 / 8 | 320 / 16 |
| `struct Pair { float a, b; } pairs[2]` | 240 / 8 | 368 / 16 |
| `struct Combo { float f; vec3 v; } combo` | 256 (f), 272 (v) | 400 (f), 416 (v) |
| `float tail` | 288 | 432 |

The rows are derived by hand in the check "std430 vs std140, derived by hand ..." and compared with what the GL reports.
Block sizes: the calculator ends at 292 / 436 bytes; the drivers report `BUFFER_DATA_SIZE` 304 / 448 (rounded up, which
"large enough to hold all variables" allows). The demo prints the full measured table.

## Top-level arrays of structures are listed once

`pairs[2]` is an array of an aggregate type declared at the top level of a shader storage block. ES 3.2 §7.3.1.1: *"entries
will not be generated for each array element. Such block members are referred to as top-level arrays and will generate
entries only for the first array element."* So the program interface contains `Probe.pairs[0].a` and `Probe.pairs[0].b`
but **not** `Probe.pairs[1].a`; that is why `TOP_LEVEL_ARRAY_SIZE` (2) and `TOP_LEVEL_ARRAY_STRIDE` (8 in std430, 16 in
std140) exist. The stride is non-zero only for arrays of structures (§7.3.1: "For top-level block members not declared as an
array of an aggregate type, zero is written"), so `farr[0]` reports 0 although its `ARRAY_STRIDE` is 4 or 16. The CPU
calculator generates all elements (they are needed to pack the buffer) and marks the ones the GL lists.

## Runtime-sized arrays and `glBindBufferRange`

```glsl
layout(std430, binding = 0) buffer Samples { uint n_samples; uint tag; float values[]; } s;
layout(std430, binding = 1) buffer Records { uint n_records; Rec recs[]; } r;     // struct Rec { float a, b, c; }
```

`values.length()` is a run-time value ("determined at run time based on the size of the buffer object providing storage
for the block", GLSL ES 3.10 §4.1.9). With `glBindBufferRange(SHADER_STORAGE_BUFFER, 0, buf, offset, size)` the block is
backed by the *range*, and ES 3.2 §6.1.1 says `size` "specifies the amount of data that can be read from or written to the
buffer object while used as an indexed target". The demo binds four ranges of one buffer whose sizes are `8 + 4 n` bytes
for `n` = 32, 16, 4 elements (and a fourth range of 39 bytes, not a multiple of the stride), each at a multiple of
`SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT` and separated from the next by a canary gap. The compute shader records
`length()`, stores `tag`, writes the elements `0 .. length()-1` and nothing else. Checks:

- `length()` equals `(size - offset of the array) / stride` for the exact-multiple ranges (32, 16, 4 and 10, 5, 2 elements;
  `Rec` has stride 12, a std430 structure of three floats having alignment 4);
- the whole buffer equals the predicted image: header words, the written elements, and **every other byte** (gaps, bytes
  past each range, the partial tail) still holds the canary — nothing was written beyond the bound range;
- for the non-multiple range the spec gives no rounding rule, so the measured value is printed (all drivers here give the
  floor) and only "no more than the floor" is asserted.

`BUFFER_DATA_SIZE` of a block ending in a runtime-sized array is computed "as if the array had one element" (§7.3.1), so
`Samples` needs at least 12 bytes and `Records` 16. `ARRAY_SIZE` of such an array is 0.

## Binding model

A storage block's binding comes only from `layout(binding = N)` (default 0). **There is no `glShaderStorageBlockBinding` in
OpenGL ES**: "It is not possible to change the binding point associated with a shader storage block after a program is
linked" (§7.8); the demo reads the binding with `BUFFER_BINDING`. Binding points are indexed context state shared by all
stages (a compute, a vertex and a fragment program all use `SHADER_STORAGE_BUFFER` index 0 here). `glBindBufferBase` binds
the whole buffer (`START` and `SIZE` then read 0, "not specified"), `glBindBufferRange` stores offset and size, and both
also set the general binding used by `glBindBuffer`/`glMapBufferRange`.

## Memory barriers

Stores by shaders are not synchronized automatically (§7.11.2). After a dispatch the demo calls
`glMemoryBarrier(SHADER_STORAGE_BARRIER_BIT)` before the vertex/fragment shaders read the data and
`glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT)` before `glMapBufferRange` / `glBufferData` touch it. The tests cannot prove the
barrier is *necessary* (without it the result is undefined, not necessarily wrong), so no check claims that.

## How it is validated

- **Both directions, both layouts**: flags (compute reads CPU-packed data) and a byte image (CPU reads compute-written
  data), compared at every scalar position the calculator gives. Padding bytes are *not* asserted (the spec does not say a
  shader preserves them); the number of padding words the shader changed is printed (0 on all drivers here).
- **Wrong layouts**: rows 2-3 of the grid. A member verifies exactly when all its scalars sit at the same byte offsets in both
  layouts; values are unique (`member*10000 + element*1000 + field*100 + scalar`), so a scalar read from another place can
  never equal the expected one.
- **Introspection**: 18 listed buffer variables per layout (type, array size, offset, strides, row-major flag, block index,
  top-level size and stride), names, block properties (`BUFFER_BINDING`, `BUFFER_DATA_SIZE`, `NUM_ACTIVE_VARIABLES`,
  `ACTIVE_VARIABLES`, `REFERENCED_BY_*`), interface counts, runtime-array properties.
- **Pixels**: all 64 grid cells and every bar (inside the bar: colour within 1 of the 8-bit quantization of
  `(v/13, 1-v/13, 0.4)`; two pixels above the bar: the clear colour).
- **Errors** (`glGetError`): index = `MAX_SHADER_STORAGE_BUFFER_BINDINGS` for `glBindBufferBase/Range/glGetIntegeri_v/
  glGetInteger64i_v`, an unaligned offset, size 0 — each `INVALID_VALUE`, and the failed calls change nothing.
- **Limits**: ES 3.2 minimums (`MAX_COMPUTE_SHADER_STORAGE_BLOCKS` ≥ 4, `MAX_COMBINED_...` ≥ 4, bindings ≥ 4,
  `MAX_SHADER_STORAGE_BLOCK_SIZE` ≥ 2^27, alignment ≤ 256); a compute shader with exactly the limit of blocks links, one
  more fails; a block of exactly `MAX_SHADER_STORAGE_BLOCK_SIZE` bytes links, 4 bytes more fails ("If the amount of storage
  required for the fixed size portion of any shader storage block ... exceeds this limit, a program will fail to link").
- **Negative controls**: one wrong word in an otherwise correct buffer makes exactly its member's flag 0; the GL's std430
  offsets compared with the std140 calculator mismatch at exactly the 14 variables where the calculators differ; the std430
  write result compared with the std140 image fails at `farr[0]`.
- **Optional stages**: the fragment-shader grid and the vertex-shader bars need `MAX_FRAGMENT/VERTEX_SHADER_STORAGE_BLOCKS` ≥ 1 (the
  ES 3.1 minimum is 0). When a limit is 0 the demo prints a note and skips that display and its checks.

## Pitfalls

- A `vec3` is 12 bytes but has alignment 16; the next scalar may sit at +12. An array of `vec3` has stride 16 in both
  layouts, an array of `float` has stride 4 in std430 and 16 in std140.
- A structure's alignment is its largest member's (std430) — `Pair` of two floats is 4-aligned with size 8, `Combo` is 16-aligned
  and padded to 32, and an array of structures has the structure's (padded) size as stride.
- Do not rely on `BUFFER_DATA_SIZE` being exactly the end of the last member; allocate at least that.
- For a `row_major` matrix `MATRIX_STRIDE` is the stride between rows, not columns (`m2x3r` has three rows of two floats).
- The names of arrays of structures: only element 0 is enumerated. Looking up `Probe.pairs[1].a` returns `GL_INVALID_INDEX`.

## Known driver deviation

`glBindBufferRange` with a negative offset that is a multiple of the alignment raises no error on Mesa (llvmpipe, radeonsi,
zink), contrary to ES 3.2 §6.1.1 ("INVALID_VALUE if buffer is non-zero and offset is negative"). The demo keeps the strict
expectation and reports XFAIL via `expect_or_known_deviation("Mesa")`; the binding is restored immediately and never used
(see `es30-uniform-buffer-objects`, same behaviour).
