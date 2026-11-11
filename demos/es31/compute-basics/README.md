# es31-compute-basics

**Minimum:** OpenGL ES 3.1 · **Shaders:** GLSL ES 3.10 · **Run:** `es31-compute-basics [--headless --validate]`

## Purpose

A compute shader is a program that is not part of the drawing pipeline: there are no vertices, no fragments, no framebuffer.
You hand the GL a *grid of work groups*, every work group is a small block of *invocations* that run the same code, and the
only way the shader communicates is through memory (buffers, in this demo). Everything that identifies "which piece of the
problem am I?" comes from built-in variables, and everything that limits the size of the problem is a queryable implementation
limit.

The demo takes those pieces apart one by one. It dispatches 1-D, 2-D and 3-D grids with different local sizes; has *every*
invocation write all six built-ins into a shader storage buffer and compares the whole buffer with a CPU model of the
specification's formulas; queries and exercises the work group limits; provokes the errors around dispatching; and then does
the three things compute shaders are usually used for first: vector addition, a second dispatch that consumes the first one's
result, and a 2-D parallel transform (a matrix transpose), all without images.

## What the image shows

Four panels, in reading order, one per dispatch configuration. **Every invocation is one square cell.** The cells are not drawn
from CPU data: the compute shader wrote one 64-byte record per invocation into an SSBO, and that same buffer is bound as a
vertex array with `glVertexAttribIPointer` and a divisor of 1, so each record becomes one instanced quad. Where the quad goes
comes from the *recorded* `gl_GlobalInvocationID`; its **hue** is the recorded work group (`gl_WorkGroupID` flattened with
`gl_NumWorkGroups`, 12-entry palette) and its **brightness** is the recorded `gl_LocalInvocationIndex`. Cells have a one pixel gap.
y grows upward, as in window coordinates.

| panel | dispatch | local size | invocations | what to look for |
|---|---|---|---|---|
| top left | `(5, 1, 1)` | `local_size_x = 8` only (y, z default to 1) | 40 | the 1-D line is wrapped so that each row is a work group: red, orange, yellow, lime, green from the bottom; brightness ramps with the local index |
| top right | `(3, 2, 1)` | `(4, 4)` | 96 | a 12 x 8 block made of six 4 x 4 tiles, work group x fastest |
| bottom left | `(2, 2, 2)` | `(4, 2, 2)` | 128 | the four z slices are tiled 2 x 2; each slice holds four 4 x 2 blocks |
| bottom right | `(2, 2, 3)` | `(3, 5, 2)` | 360 | a non-power-of-two local size; six slices tiled 3 x 2, twelve distinct hues |

If a built-in were wrong anywhere (a swapped axis, an off-by-one, a work group that did not run), cells would be missing,
displaced or mis-colored.

## Central calls

| Call | Role |
|---|---|
| `glCreateShader(COMPUTE_SHADER)`, `glCompileShader`, `glLinkProgram` | compute shaders are ordinary shader objects of a new type; a program holds exactly one |
| `glDispatchCompute(x, y, z)` | launches `x * y * z` work groups; the work group size is *not* a parameter, it is in the shader |
| `glGetProgramiv(program, COMPUTE_WORK_GROUP_SIZE, int[3])` | reads back the local size the shader declared |
| `glGetIntegeri_v(MAX_COMPUTE_WORK_GROUP_COUNT / MAX_COMPUTE_WORK_GROUP_SIZE, 0..2)`, `glGetIntegerv(MAX_COMPUTE_WORK_GROUP_INVOCATIONS)` | the three limits |
| `glBindBufferBase(SHADER_STORAGE_BUFFER, n, buffer)` | the buffer behind `layout(std430, binding = n) buffer` |
| `glMemoryBarrier(SHADER_STORAGE_BARRIER_BIT / BUFFER_UPDATE_BARRIER_BIT / VERTEX_ATTRIB_ARRAY_BARRIER_BIT)` | make shader writes visible to the next dispatch / to a CPU mapping / to the vertex fetch |
| `glMapBufferRange(..., MAP_READ_BIT)` | the CPU reads the result |

## Spec

- ES 3.2 **§17 Compute Shaders** (§17.1 for shared memory, see the sibling demo `es31-compute-shared-memory`); §7.3 Program Objects;
  §7.12 Shader, Program, and Program Pipeline Queries; **§7.11.2** Shader Memory Access Synchronization; §6.3 Mapping and Unmapping
  Buffer Data; §2.3.1 Errors; §20.1 Simple Queries; **Table 21.49** Implementation Dependent Compute Shader Limits.
- GLSL ES 3.10 **§4.4.1.1** Compute Shader Inputs and **§7.1.3** Compute Shader Special Variables.

The rules, shortened:

- `DispatchCompute(num_groups_x, y, z)` starts that many work groups. "While the individual shader invocations within a workgroup
  are executed as a unit, workgroups are executed completely independently and in unspecified order." `gl_NumWorkGroups` is
  initialized with the three counts. "If the workgroup count in any dimension is zero, no workgroups are dispatched" (not an error).
- The local size is declared with `layout(local_size_x = X, local_size_y = Y, local_size_z = Z) in;` and is a compile-time constant;
  "the default size in each dimension is 1". Exceeding the per-dimension maximum is "a compile-time error"; a compute shader without
  a fixed local size is "a link-time error". Using `gl_WorkGroupSize` before a size is declared is "a compile-time error".
- Definitions (§7.1.3): `gl_GlobalInvocationID = gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID` and
  `gl_LocalInvocationIndex = id.z * size.x * size.y + id.y * size.x + id.x`. `gl_WorkGroupSize` is a constant.
- Errors of `DispatchCompute`: `INVALID_OPERATION` "if there is no active program object for the compute shader stage";
  `INVALID_VALUE` "if any of num groups x, y and z are greater than the maximum workgroup count for the corresponding dimension".
  §7.3: the current program is active for a stage only "if it contains executable code for that stage", so a vertex+fragment
  program that is current counts as "no active program" for the compute stage.
- A command that generates an error "is ignored so that it has no effect on GL state", and if it writes through a pointer "no
  change is made to these values" (§2.3.1). `GetIntegeri_v` with an index outside the valid range is `INVALID_VALUE` (§20.1).
- A program that contains a compute shader must not contain a shader of any other type (§7.3, link failure).
  `GetProgramiv(COMPUTE_WORK_GROUP_SIZE)` is `INVALID_OPERATION` for a program that has no compute shader (§7.12).
- Table 21.49 minimums: group count 65535 in each dimension, group size 128 x 128 x 64, 128 invocations per group.
- §7.11.2: shader stores are not synchronized automatically. `SHADER_STORAGE_BARRIER_BIT` makes later shader accesses to buffer
  variables see earlier shader writes; `BUFFER_UPDATE_BARRIER_BIT` does the same for later buffer reads/writes by the API
  (`glMapBufferRange` and friends); `VERTEX_ATTRIB_ARRAY_BARRIER_BIT` for vertex fetches from buffers.

## Shader concepts

- **`builtins.comp`** is one shader body compiled five times. The host inserts a single `#define LOCAL_SIZE local_size_x = ..., ...`
  line below the `#version 310 es` line and the shader says `layout(LOCAL_SIZE) in;`, so the same code runs for 1-D, 2-D, 3-D, a
  non-power-of-two local size and the maximum size the implementation allows. Each invocation stores
  `gl_NumWorkGroups`, `gl_WorkGroupID`, `gl_LocalInvocationID`, `gl_GlobalInvocationID`, `gl_LocalInvocationIndex` and `gl_WorkGroupSize`
  into the record at its own flat global index, so no two invocations ever write the same words (no atomics needed).
  `uvec4` members keep the std430 layout trivial (a `uvec3` would be padded).
- **`vecadd.comp`, `transform.comp`** use a one-dimensional grid whose size is not a multiple of the local size; the
  `if (i < u_count)` guard keeps the surplus invocations from writing past the end.
- **`transpose.comp`** uses a 16 x 8 local size and a 7 x 9 grid over a 100 x 70 matrix.
- **`hits.comp`** is a one-invocation work group used to dispatch *exactly* the maximum group count: `hits[id] = id + 1`.
- **`display.vert`** turns each record into a cell with integer arithmetic only (palette indexing, `(pal * brightness) / 255u`), so the
  expected 8-bit colors are exact on the CPU.
- Negative shaders: `no_local_size.comp`, `workgroup_size_without_layout.comp`.

## How it is validated

- **Limits.** All seven values are compared with Table 21.49 (>= 65535, 65535, 65535; >= 128, 128, 64; >= 128). `GetIntegeri_v`
  with index 3 raises `INVALID_VALUE` and leaves its output untouched. The measured values are printed: radeonsi reports a group count
  of 2147483646 in x and 65535 in y and z, llvmpipe and zink 65535 everywhere, size 1024 in every dimension and 1024 invocations on all
  three.
- **Built-ins.** For each of the four configurations *and* for a maximum-size work group (the largest group the limits allow, capped at
  4096 invocations: 1024 as `(1024, 1, 1)`, 3 groups, on all three Mesa drivers) the whole record buffer is mapped after
  `glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT)` and compared with the CPU model **exactly** (every word of every record: 40, 96, 128,
  360 and 3072 records). The buffer is pre-filled with `0xDEADBEEF` and has one extra *guard record* behind the last invocation; a
  sentinel left anywhere means an invocation did not run, a changed guard means one wrote out of its slot. The shader additionally
  stops any invocation whose computed slot is out of range, so a wrong `gl_GlobalInvocationID` cannot corrupt memory.
- **`COMPUTE_WORK_GROUP_SIZE`** equals the layout qualifier in each case, `(8, 1, 1)` for the shader that names only `local_size_x`; the
  same query on the display program (no compute shader) is `INVALID_OPERATION` and leaves its output untouched.
- **Dispatch limits and errors** with the `hits` program: dispatching exactly the limit in y and z (and x wherever the limit is at most
  2^20) raises no error and every one of the 65535 work groups runs (`hits[i] == i + 1` for all i, nothing behind them); one group above
  the limit, in each dimension (including x on radeonsi, where the limit is 2147483646), is `INVALID_VALUE` and *no* group runs; a zero
  count in x, y or z is silent and runs nothing; with no program current, and with a vertex+fragment program current, the dispatch is
  `INVALID_OPERATION`; the same dispatch with the compute program current is error-free (the positive control).
- **Shaders that must not build.** `local_size_x = MAX_COMPUTE_WORK_GROUP_SIZE[0] + 1` fails to compile (the largest legal size
  compiles); a compute shader without a local size compiles but fails to link; `gl_WorkGroupSize` without a declaration fails to compile;
  a program with a compute shader and a vertex shader fails to link.
- **Array computations.** Vector addition of 1000 floats (a[i] = i, b[i] = 3i + 0.5, so every sum 4i + 0.5 is exactly representable)
  in 16 groups of 64 = 1024 invocations: every sum is bit-exact and the 24 guard words behind the array are untouched. A dependent second
  dispatch converts the sums to `uint` and hashes them; it only works because `glMemoryBarrier(SHADER_STORAGE_BARRIER_BIT)` sits between the
  dispatches. A 100 x 70 transpose (7 x 9 groups of 16 x 8 = 112 x 72 invocations) is exact, including the 24 guard words.
- **Image.** The full framebuffer is read once; for all 624 cells the center pixel must equal the color the *CPU model* predicts for that
  invocation (exact: the palette arithmetic is integer, tolerance 0), the last pixel column of every cell must show the panel color
  (cells are exactly one pixel smaller than their pitch), and the margins between the panels keep the clear color.
- **Negative controls (checks that can fail).** (a) One flipped `gl_LocalInvocationID.y` in a copy of the CPU model is reported as exactly
  one bad record; (b) a buffer that was never dispatched mismatches the model in every record (the sentinel is what an invocation that did
  not run leaves behind); (c) the vector sums compared with a + b + 1 mismatch at all 1000 positions; (d) the transpose is not equal to the
  untransposed matrix; (e) no cell of the image would also match a model whose hues are shifted by one palette entry. Deliberately breaking a
  shader (for example swapping `gl_WorkGroupID.xy` in the record) fails the record checks and the image check, as intended.
- **Not exercised.** `glDispatchComputeIndirect` (next demo). `glMemoryBarrier` with an undefined bit is `INVALID_VALUE`, but meta-gl's typed
  `MemoryBarrierMask` has no such value (by design, see META_GL_GAPS.md O1). A local size whose *product* exceeds
  `MAX_COMPUTE_WORK_GROUP_INVOCATIONS` is not given a specific error by the specification text, so nothing is asserted. And `glMemoryBarrier`
  itself: the demo shows that results are exact after the documented barriers, but it cannot show that they would be wrong without them,
  because implementations may synchronize more than the minimum.

## Run

```
./build/debug/bin/es31-compute-basics --headless --validate --capture compute-basics.png
```

## Pitfalls

- A dispatch counts work **groups**, not invocations. The global size is `groups * local_size`; to cover `n` elements dispatch
  `ceil(n / local_size)` groups and guard the surplus invocations with `if (i < n)`.
- The local size is part of the shader: it cannot come from a uniform. To change it, compile another shader (this demo does that with a
  `#define`). The size is bounded in each dimension *and* in total (`MAX_COMPUTE_WORK_GROUP_INVOCATIONS`); a size within each per-dimension
  limit can still exceed the total.
- Work groups run in unspecified order, possibly in parallel: never rely on one group seeing another's output inside a single dispatch.
  Give every invocation its own output location, or use the atomics demo.
- A dispatch does not make its results visible by itself. Pick the barrier bit by *consumer*: `SHADER_STORAGE_BARRIER_BIT` for the next
  shader reading the buffer, `BUFFER_UPDATE_BARRIER_BIT` for the CPU (`glMapBufferRange`, `glCopyBufferSubData`), `VERTEX_ATTRIB_ARRAY_BARRIER_BIT`
  for vertex fetch. Omitting a barrier can happen to work on one driver and fail on another.
- `glUseProgram(compute_program)` replaces the graphics program for *all* stages. Draw calls issued while a compute-only program is
  current have undefined vertex and fragment results (not an error); switch back to the graphics program before drawing, as `render()` does.
- `uvec3`/`vec3` members in std430 blocks occupy 16 bytes like a `vec4`; a `uvec3[]` or a struct of `uvec3` is the classic source of
  "my data is shifted" bugs. The demo uses `uvec4` records.
- Under `MESA_GLES_VERSION_OVERRIDE=3.1` (the `es-min` runs) only the reported version changes; the context is still Mesa's ES 3.2 driver.
