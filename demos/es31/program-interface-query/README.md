# es31-program-interface-query

**Minimum:** OpenGL ES 3.1 · **Shaders:** GLSL ES 3.10 (`#version 310 es`) · **Run:** `es31-program-interface-query [--headless --validate]`

## Purpose

Before ES 3.1 an application could ask a linked program for its uniforms (`glGetActiveUniform`), its attributes
(`glGetActiveAttrib`), its uniform blocks and its transform feedback varyings, each with its own command and its own
vocabulary. ES 3.1 adds one uniform mechanism, the **program interface query**, and with shader storage blocks and
atomic counters there is more to ask for. A program exposes eight *interfaces*, each a list of *active resources*:

| Interface | Resources |
|---|---|
| `UNIFORM` | active uniforms of the default block, members of uniform blocks, atomic counters |
| `UNIFORM_BLOCK` | active uniform blocks (one entry per element of an array of blocks) |
| `PROGRAM_INPUT` | active inputs of the **first** stage (vertex attributes, `gl_VertexID`, ...) |
| `PROGRAM_OUTPUT` | active outputs of the **last** stage (fragment outputs) |
| `BUFFER_VARIABLE` | members of shader storage blocks |
| `SHADER_STORAGE_BLOCK` | active shader storage blocks |
| `ATOMIC_COUNTER_BUFFER` | one entry per binding point used by an active atomic counter |
| `TRANSFORM_FEEDBACK_VARYING` | the varyings named in `glTransformFeedbackVaryings`, in that order |

Five commands cover them: `glGetProgramInterfaceiv` (how many, how long are the names, how many members per block),
`glGetProgramResourceIndex` (name to index), `glGetProgramResourceName` (index to name), `glGetProgramResourceiv`
(properties of Table 7.2: type, array size, location, block index, offset, strides, bindings, "referenced by which
stage", ...) and `glGetProgramResourceLocation` (name to location). The demo verifies all of it against numbers that
it **derives from the shader source by hand**, and then shows that the numbers are not just reported but *usable*.

## The two programs

ES 3.1 requires shader storage blocks and atomic counters only in compute shaders, so two programs are needed:

* **gfx** (`shaders/query.vert`, `query.frag`): `UNIFORM`, `UNIFORM_BLOCK`, `PROGRAM_INPUT`, `PROGRAM_OUTPUT`,
  `TRANSFORM_FEEDBACK_VARYING`. It declares, among others, explicit locations on uniforms, a structure uniform, a sampler
  array with `binding`, two uniform blocks (one an array of blocks), an integer vertex input without a location, a
  built-in input (`gl_VertexID`) and a fragment output array.
* **comp** (`shaders/query.comp`): `SHADER_STORAGE_BLOCK`, `BUFFER_VARIABLE`, `ATOMIC_COUNTER_BUFFER`, and again `UNIFORM`
  (the counters), with std430 layouts, a runtime-sized array of structures, a block without instance name and without
  binding, and counters in two buffers.

`shaders/loc_edge.frag`, `loc_duplicate.frag`, `loc_overlap.frag` and `fullscreen.vert` are tiny shaders for the explicit
uniform location rules.

## Reading the image

The 256 x 256 surface (the size of every test run; the pixel positions are written for it) has three parts.

**Upper half: eight 64 x 64 cells,** one per resource group of the gfx program (cell 0 is bottom-left of the grid of
4 x 2). Each cell is coloured by a different mechanism, and each is compared with a CPU model:

| Cell | Mechanism | Expected colour (8-bit) |
|---|---|---|
| 0 | `u_tint`, `layout(location = 4)`, declared in **both** shaders; `a_color * tint.rgb` in the vertex stage, `* (tint.a + 0.5)` in the fragment stage | (143, 143, 48) |
| 1 | `u_weights[3]` set by **one** `glUniform1fv` at location 8; the elements are at 8, 9, 10 | (64, 191, 32) |
| 2 | structure `u_light`: `dir` at location 20, `power` at 21 (consecutive members) | (191, 96, 48) |
| 3 | `layout(binding = 2) sampler2D u_tex`: unit 2, **no `glUniform1i`** | (255, 64, 191) |
| 4 | `layout(binding = 4) sampler2D u_texarr[2]`: left half unit 4, right half unit 5 | (64, 191, 255) / (191, 255, 64) |
| 5 | block `Params` (`std140`, `binding = 3`) summed in the vertex shader: `scale + basis[1] + skew + k + offset`, read at hand-derived offsets | (143, 159, 143) |
| 6 | `Cell cells[2]` (`binding = 6`): left half element 0 (binding point 6), right half element 1 (**7**) | (64, 191, 64) / (191, 64, 191) |
| 7 | `u_pick` without a location, set with the location returned by `glGetProgramResourceLocation` | (191, 64, 191) |

In none of these cells does the application call `glUniform1i` for a sampler or `glUniformBlockBinding` for a block: the
`layout(binding)` qualifiers did it. The grid is drawn from a vertex buffer whose attributes sit at locations obtained from
the query API (`a_cell` is an `ivec2` without an explicit location, so its location is the linker's choice).

**Strip at y = 96..127: the explicit-location patch.** A program with `layout(location = MAX_UNIFORM_LOCATIONS - 1)
uniform float u_edge` is drawn into the first 32 pixels, with `glUniform1f` at that location (here 81920 - 1 on radeonsi):
magenta (191, 64, 191).

**Lower 96 rows: the check matrix.** Every recorded check of `validate()` becomes one small cell, painted after all checks
have run: green = passed, red = failed. Groups start on a new row, in this order from the top: `picture` (15),
`UNIFORM (gfx)` (45), `UNIFORM_BLOCK (gfx)` (14), `empty interfaces` (12), `PROGRAM_INPUT / PROGRAM_OUTPUT` (21),
`TRANSFORM_FEEDBACK_VARYING` (11), `SHADER_STORAGE_BLOCK / BUFFER_VARIABLE` (30), `ATOMIC_COUNTER_BUFFER / UNIFORM (comp)` (12),
`names and locations` (46), `layout(binding)` (12), `explicit uniform locations` (10), `errors` (53) and `compute run with
queried layout` (8). Everything is green.

## What is derived by hand

The numbers in `main.cpp` come from the spec, not from the driver. The layout rules are written next to them:

* **std140** block `Params` (ES 3.2 §7.6.2.2): `vec4 scale` at 0; `mat3 basis` is three `vec3` columns rounded to 16 bytes,
  `MATRIX_STRIDE` 16, at 16; `row_major mat2x3 skew` is *three rows* of a `vec2` rounded to 16, `MATRIX_STRIDE` 16,
  `IS_ROW_MAJOR` 1, at 64; `float k[2]` has `ARRAY_STRIDE` 16, at 112; `vec3 offset` at 144, ending at 156. The queried
  `BUFFER_DATA_SIZE` (160 on Mesa) must be at least 156, because the spec calls it an "implementation-dependent minimum".
* **std430** block `Source` (§7.8): `head` 0, `m` 16 with stride 16, `w[3]` at 64 with `ARRAY_STRIDE` 4 (not rounded up in
  std430), `rm` (row-major `mat2x3`) at 80 with `MATRIX_STRIDE` 8. Block `Result`: `count` at 0, then `Item items[]` at 8
  with a structure stride of 16; it is a *top-level array* of an aggregate type of unspecified size, so its entries exist
  for element 0 only (`Result.items[0].p`, `.s`), `TOP_LEVEL_ARRAY_SIZE` is 0 and `TOP_LEVEL_ARRAY_STRIDE` 16, while
  `ARRAY_SIZE` of `p` and `s` is 1.
* **Names** (§7.3.1.1): an array of basic type is one entry named `name[0]`; structure members are `s.m`; a block member
  is `Block.member` with the *block* name, even if the instance is called differently; an array of blocks has one entry per
  element (`Cell[0]`, `Cell[1]`) whose members are listed once as `Cell.color`, with `BLOCK_INDEX` of the first element; the
  members of a block without instance name are bare (`plain_value`); array elements other than the first are not entries.
* **Locations** (§7.3.1.1, §7.6): explicit locations are used; array elements and structure members take consecutive
  locations; generated locations never collide with explicit ones; block members, atomic counters and built-ins have
  location -1; `PROGRAM_INPUT` lists only the first stage and `PROGRAM_OUTPUT` only the last (so `v_color` appears in neither).
* **Referenced by** (Table 7.2): `u_tint` is used by both stages, `u_weights` only by the fragment shader, `Params` only by the
  vertex shader, `Cell` only by the fragment shader.

## Central calls

`glGetProgramInterfaceiv`, `glGetProgramResourceIndex`, `glGetProgramResourceName`, `glGetProgramResourceiv`,
`glGetProgramResourceLocation` (`Query` helper in `main.cpp`, one wrapper per command), `glTransformFeedbackVaryings`
(before the link), `glBindBufferBase`, `glDispatchCompute`.

## Spec sections

ES 3.2 §7.3.1 Program Interfaces (Table 7.2 and Table 7.3), §7.3.1.1 Naming Active Resources, §7.6 Uniform Variables,
§7.6.2.2 Standard Uniform Block Layout, §7.7 Atomic Counter Buffers, §7.8 Shader Buffer Variables and Shader Storage Blocks,
§11.1.2.1 (transform feedback varyings). GLSL ES 3.10 §4.4.1 and §4.4.2 (locations on `in`/`out`), §4.4.3 (uniform locations),
§4.4.4 (blocks, `binding`), §4.4.5 (opaque uniforms, `binding`), §4.4.6 (atomic counters, `binding` and `offset`).

## Shader concepts

`layout(location = N)` on default-block uniforms (including arrays and structures), `layout(binding = N)` on samplers,
uniform blocks, shader storage blocks and atomic counters, `layout(offset = N)` on atomic counters, `std140` and `std430`,
`row_major`, arrays of blocks, runtime-sized arrays in a storage block, blocks without instance name, `atomic_uint` in a
compute shader, an `out` array at consecutive fragment output locations.

## Validation

* **Tables vs the driver.** For every resource of every interface the demo looks the entry up by name
  (`glGetProgramResourceIndex`), reads the name back, and compares every property of Table 7.2 that the interface supports.
  A mismatch lists the differing properties.
* **Counts and lengths.** `ACTIVE_RESOURCES` equals the number of entries derived from the shaders and the ES 3.0 counts
  (`ACTIVE_UNIFORMS`, `ACTIVE_UNIFORM_BLOCKS`, `TRANSFORM_FEEDBACK_VARYINGS`, `ACTIVE_ATOMIC_COUNTER_BUFFERS`);
  `MAX_NAME_LENGTH` is the longest name plus the terminator; `MAX_NUM_ACTIVE_VARIABLES` the most members of one block.
  Interfaces without resources report 0.
* **Order and truncation.** The transform feedback list is in the order passed before the link; `glGetProgramResourceName`
  truncates and terminates; `glGetProgramResourceiv` writes values in the order of `props` and no more than `count`.
* **Pixels.** The eight cells above are compared with a CPU model, ±1 per channel (8-bit quantization; every value is chosen
  away from a `.5 / 255` rounding boundary). A negative control changes `u_tint`, redraws cell 0 and requires the old
  expectation to fail.
* **Compute run.** The `Source` buffer is packed with the *queried* offsets and strides and the result buffers are read at the
  queried offsets, after five invocations; the three atomic counters (two buffers, three offsets) must hold 5, 10 and 15.
* **Explicit locations.** The highest legal location works; one past it, a duplicate location and an array that overlaps
  another uniform are rejected at compile or link time (GLSL ES 3.10 §4.4.3: "a compiler or linker error").
* **Errors.** 17 properties that Table 7.2 does not allow for their interface, `MAX_NAME_LENGTH` on atomic counter buffers,
  `MAX_NUM_ACTIVE_VARIABLES` on non-block interfaces, name queries on `ATOMIC_COUNTER_BUFFER`, out-of-range indices,
  `glGetProgramResourceLocation` on the five interfaces it does not accept, unlinked programs, unknown and shader names.

## Pitfalls

* **The order of the lists is not specified** (except transform feedback), so the demo identifies entries by name or by
  `BUFFER_BINDING`, never by index.
* **`BUFFER_DATA_SIZE` is a minimum.** Mesa rounds up (112 for a 104-byte std430 block, 160 for a 156-byte std140 block);
  the demo asserts "at least the end of the last member".
* **`ARRAY_STRIDE` and `MATRIX_STRIDE` are -1 for default-block uniforms** (not backed by a buffer) and **0** for block
  members that are not arrays or matrices. `IS_ROW_MAJOR` is 0 outside blocks.
* **Arrays of aggregates in storage blocks list element 0 only**; the application has to use `TOP_LEVEL_ARRAY_STRIDE`.
* **`gl_VertexID` is a program input** with location -1; a count of `PROGRAM_INPUT` resources that ignores built-ins is
  off by one.
* **Drivers (Mesa).** radeonsi reports a uniform block that only the vertex shader declares (and its members) as referenced
  by the fragment shader as soon as the fragment shader declares *other* uniform blocks, although "the value one is written
  ... if an active uniform block ... contains at least one variable referenced by the corresponding shader" (llvmpipe and zink
  conform; XFAIL on radeonsi only). All Mesa drivers ignore `count` when `glGetProgramResourceiv` returns `ACTIVE_VARIABLES`
  (they write every index and report the full length, although "only the first count integer values will be written"; XFAIL
  on Mesa). Both are recorded in `docs/KNOWN_LIMITATIONS.md` by the integrator.
* **Drivers (ANGLE).** ANGLE lists the single entry `o_id[0]` for `out ivec2 o_id[2]` but returns its index for
  `glGetProgramResourceIndex(PROGRAM_OUTPUT, "o_id[1]")`, although §7.3.1.1 says "a name identifying any array element other
  than the first (e.g., "a[1]") is not considered to match" (XFAIL on ANGLE).
* `IS_PER_PATCH` and the `REFERENCED_BY_TESS_*` / `_GEOMETRY_` properties are ES 3.2 only and have no program to query here.
* `MATRIX_STRIDE` is the distance between **columns** of a column-major matrix but between **rows** of a `row_major` one, and
  a `row_major` `mat2x3` occupies three rows, not two columns (see `Params.skew` and `Source.rm`). Packing a row-major matrix
  with the column-major formula is the classic mistake; the compute run would then read the wrong element.
