# es30-instancing

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es30-instancing [--headless --validate]`

## Purpose

Instancing draws the same piece of geometry many times with one call. For every instance the vertex shader runs
again on the same vertices, but it can see which instance it is working on (`gl_InstanceID`), and chosen vertex
attributes advance *per instance* instead of per vertex (`glVertexAttribDivisor`). A field of grass blades, a
particle system or a grid of tiles then costs one draw call, not thousands.

This demo draws a 4 × 4 grid of quads in several ways, then switches to the other part of ES 3.0's drawing
machinery that sits in the same section of the specification: `glDrawRangeElements`, **primitive restart**
(`PRIMITIVE_RESTART_FIXED_INDEX`, which splits one `TRIANGLE_STRIP` index list into separate strips) and the
`MAX_ELEMENT_INDEX` limit. Every instance, every strip and every gap is checked against a CPU model.

## What the image shows

256 × 256 default framebuffer, a 4 × 4 grid of 64 × 64 cells; row 3 is the top row, cells are numbered in
reading order.

| Cell | What is drawn | What it demonstrates |
|---|---|---|
| 0 | `glDrawArraysInstanced(TRIANGLE_STRIP, 0, 4, 16)`; offset attribute divisor 1, color divisor 1 | 16 instances, each with its own offset and its own color |
| 1 | `glDrawElementsInstanced(TRIANGLES, 6, USHORT)`; offset divisor 1, color **divisor 2** | instances 0 and 1 share color 0, 2 and 3 share color 1, ...: pairs of equal colors |
| 2 | color **divisor 3** | element = `floor(instance / 3)`: groups of three, which do not line up with the rows of four |
| 3 | no vertex attributes at all: quad from `gl_VertexID`, slot, size and color from `gl_InstanceID` | the built-in inputs; the squares grow with the instance number |
| 4 | the offset attribute with **divisor 0** inside an instanced draw | a divisor of 0 means "advance per vertex": every instance reads offsets 0..3 for its four vertices, so all 16 instances coincide in one twisted shape (the last instance's color wins) |
| 5 | `glDrawArrays` (not instanced) with the divisor-1 VAO, then the `gl_InstanceID` shader with `glDrawArrays` | one quad: a non-instanced draw is instance 0; `gl_InstanceID` reads 0 (the small dark-blue square) |
| 6 | `glDrawArraysInstanced(..., 5)` | only the first five instances |
| 7 | `glDrawArraysInstanced(POINTS, 0, 1, 4)`; `gl_PointSize = 3 + 2 * gl_InstanceID` | four squares of side 3, 5, 7, 9; position and color from `gl_InstanceID` |
| 8 | `TRIANGLE_STRIP` over a two-ribbon vertex set, indices `0..11`, restart enabled | the strip continues between the ribbons: two "glue" triangles fill the gap |
| 9 | same vertices, `USHORT` indices with `0xFFFF` after index 5, restart enabled | the strip restarts there: **a gap** |
| 10 | `UBYTE` indices with `0xFF` | the restart index is the maximum of the index type |
| 11 | `UBYTE` indices with `0xFF`, restart **disabled** | `0xFF` is just vertex 255, a far-away apex: the "spike" |
| 12 | `UINT` indices with `0xFFFFFFFF` | restart with 32-bit indices |
| 13 | `glDrawElementsInstanced(..., 3)` with restart | every one of three stacked, fading copies has the gap |
| 14 | `glDrawRangeElements(TRIANGLE_STRIP, 0, 11, 13, ...)` with the restart index in the list | the restart index is exempt from the `[start, end]` rule |
| 15 | `glDrawRangeElements(TRIANGLE_STRIP, 6, 11, 6, ...)` from byte offset 14 | only the second ribbon |

## Central calls

| Call | Role |
|---|---|
| `glDrawArraysInstanced(mode, first, count, instanceCount)` | `count` vertices, `instanceCount` times |
| `glDrawElementsInstanced(mode, count, type, indices, instanceCount)` | same with an element array |
| `glVertexAttribDivisor(index, divisor)` | 0: advance per vertex; d > 0: advance once per d instances. Part of the VAO. |
| `glDrawRangeElements(mode, start, end, count, type, indices)` | `glDrawElements` plus a promise: all indices lie in `[start, end]` |
| `glEnable/glDisable(PRIMITIVE_RESTART_FIXED_INDEX)` | turn restart on or off; the index is fixed to `2^N - 1` (N = 8, 16, 32) |
| `glGetInteger64v(MAX_ELEMENT_INDEX)` | the largest usable `UNSIGNED_INT` index; `MAX_ELEMENTS_VERTICES` / `_INDICES` via `glGetIntegerv` |

## Spec

- ES 3.0 §2.9 Vertex Arrays (`VertexAttribDivisor`); **§2.9.1** Transferring Array Elements (primitive restart);
  **§2.9.3** Drawing Commands (`DrawArraysInstanced`, `DrawElementsInstanced`, `DrawRangeElements`,
  `MAX_ELEMENT_INDEX`, the recommended maxima).
- ES 3.0 §2.11 (divisors are VAO state, Table 6.2), Table 6.3 (the restart flag is not), §2.12.9.5 Shader Inputs,
  §3.4 Points (point size and its clamping), §2.5 GL Errors, §6.1.2 Data Conversions.
- GLSL ES 3.00 §7.1 Vertex Shader Special Variables: `gl_VertexID`, `gl_InstanceID`, `gl_Position`, `gl_PointSize`.

The rules, shortened:

- *Instanced attributes.* "If divisor is zero, the attribute at slot index advances once per vertex. If divisor is
  non-zero, the attribute advances once per divisor instances." The element transferred for every vertex of
  instance `i` is `floor(i / divisor)`. A draw that is not instanced is instance 0, and `gl_InstanceID` is 0.
- *Zero instances.* `DrawArraysInstanced` is defined as a loop over `instanceCount`; with 0 nothing is drawn, no error.
- *Restart.* "If the index ... is equal to 2^N - 1, where N is 8, 16 or 32 if the type is UNSIGNED_BYTE,
  UNSIGNED_SHORT, or UNSIGNED_INT, then the GL does not process those elements as a vertex. Instead, it is as if
  the drawing command ended with the immediately preceding transfer, and another drawing command is immediately
  started ..." This applies to `DrawElements`, `DrawElementsInstanced` and `DrawRangeElements`, and only while the
  flag is enabled.
- *Range.* `end < start` is `INVALID_VALUE`; indices outside `[start, end]` (other than the restart index) are the
  caller's problem. `MAX_ELEMENTS_VERTICES` and `MAX_ELEMENTS_INDICES` are *recommended* maxima: exceeding them "may
  operate at reduced performance", it is not an error.
- *`MAX_ELEMENT_INDEX`.* Minimum 2^24 - 1, queried with `GetInteger64v`. An `UNSIGNED_INT` index above it is undefined
  "unless primitive restart is enabled and the index value is 2^32 - 1".

## Shader concepts

`instanced.vert` is attribute-driven: `a_corner` (divisor 0), `a_offset` and `a_color` (the divisors under test). The
shader does not know the divisors; the VAO state decides which array element it receives. `instance_id.vert` has no
inputs: `gl_VertexID & 1` and `gl_VertexID >> 1` make the corner of a `TRIANGLE_STRIP` quad, `gl_InstanceID % 4` and
`/ 4` the grid slot, and `gl_InstanceID` also drives size and color. `points.vert` writes `gl_PointSize` from
`gl_InstanceID`; the window position is built so that the point centers lie on pixel centers and the odd point sizes
put the square edges on integer coordinates, which no pixel center can touch. `strip.vert` places and fades copy
`gl_InstanceID` of a strip by a uniform step. The fragment shader is the same trivial `color.frag` everywhere.

## How it is validated

- **CPU model.** Instanced cells are triangle lists built from the uploaded arrays with the spec's element rule
  (`floor(i / divisor)`, or the vertex index for divisor 0). Strip cells apply the strip-assembly and restart rule to
  the index list (an index equal to the restart value ends the current strip, then every window of three consecutive
  indices of a run is a triangle). Points are squares around window positions. Later primitives overwrite earlier ones.
- **Whole-cell comparison.** Every pixel the model is certain about is compared, so the instance quads, the gaps between
  them, the ribbons and their gaps are all covered. Pixels within 1 px of a triangle edge are skipped: which fragment
  an edge pixel yields is up to the rasterizer's tie-breaking rules. Tolerance: ±1 for flat colors (8-bit rounding),
  ±3 for the Gouraud-shaded ribbons (interpolation with finite-precision barycentrics); the largest deviation seen
  on radeonsi, llvmpipe and zink is 1/255. Point cells use ±1 and have no ambiguous pixels by construction.
- **Probes.** For each grid cell one pixel per slot is compared with the model *and* the model is required to agree with
  the intent (drawn slots are covered, the others are background): 16 of 16 in cells 0-3, 2 of 16 in cell 5, 5 of 16 in
  cell 6. For each strip cell, one probe in each of the four ribbon quads and two probes in the gap that the restart
  empties (and that the connecting triangles fill when there is no restart). Cell 13 probes both ribbons and the gap of each of the
  three instances; cell 11 probes the spike; cell 15 probes that the first ribbon is *not* drawn.
- **Restart evidence independent of the model.** The restarted strips (USHORT, UBYTE, UINT, `glDrawRangeElements`)
  cover the same set of pixels, fewer than the unbroken strip; with restart disabled the spike covers pixels above the
  ribbons that no other strip touches.
- **Negative controls.** Pixels of one cell are compared with the model of a *different* configuration and must disagree:
  divisor 1 vs divisor 2 (both ways), 16 instances vs a 15-instance model, the restarted strip vs the model without
  restart (and the reverse), the restart-disabled strip vs the restarted model, the non-instanced draw vs a 16-instance
  model.
- **State.** `VERTEX_ATTRIB_ARRAY_DIVISOR` of attributes 0/1/2 in each of the four instancing VAOs; a new VAO starts with
  divisor 0; setting a divisor in one VAO does not change another; `ELEMENT_ARRAY_BUFFER_BINDING` is per VAO. The restart
  flag: initially disabled (recorded in `init`), `glEnable` / `glIsEnabled` / `glGetBooleanv`, not VAO state.
- **Limits.** `glGetInteger64v(MAX_ELEMENT_INDEX)` >= 2^24 - 1, and the 32-bit query returns the 64-bit value clamped to
  `INT_MAX`. Mesa reports 2^32 - 1, `MAX_ELEMENTS_VERTICES = MAX_ELEMENTS_INDICES = 3000`. With such small recommended
  maxima the demo can *exceed both*: it redraws cell 15 from a 3004-vertex buffer with `start = 0`, `end = 3003` and 3006
  indices (the extra indices repeat the last one, so they only add degenerate triangles) and requires the same picture. If an
  implementation reports huge maxima the test is skipped (it would need huge buffers).
- **Errors.** `glVertexAttribDivisor` with index `MAX_VERTEX_ATTRIBS`, negative `instanceCount`, negative `count`, an invalid
  index `type` (FLOAT, INT) for `glDrawElementsInstanced` / `glDrawRangeElements`, and `end < start`; each must give the
  error the spec names. `instanceCount` 0 raises no error and leaves every pixel of the cell untouched.
- **Point sizes.** `ALIASED_POINT_SIZE_RANGE` is queried. Only the point sizes inside the range are compared; a size beyond the
  range is clamped by the GL (§3.4) and its surroundings are skipped. (All three drivers support at least 256.)

## Run

```
./build/debug/bin/es30-instancing --headless --validate --capture instancing.png
```

## Pitfalls

- Divisors belong to the VAO. Switching VAOs switches them; creating a VAO resets them to 0.
- A divisor of 0 inside an *instanced* draw is legal and means "per vertex" (cell 4). Instanced attributes are the ones
  with a divisor *above* zero.
- The restart index depends on the index type: `0xFF`, `0xFFFF` or `0xFFFFFFFF`. With `UNSIGNED_BYTE` and restart off,
  `0xFF` is vertex 255, which must exist in the buffer (cell 11 has 256 vertices for exactly this reason).
- Primitive restart is a state flag, not part of the VAO, and it only affects the `*Elements*` commands, not `glDrawArrays`.
- ES 3.0 only has the *fixed index* restart. A configurable restart index (`glPrimitiveRestartIndex`) is desktop GL.
- `glDrawRangeElements` does not check that indices lie in `[start, end]`; an index outside is implementation-dependent.
- `gl_InstanceID` always starts at 0: ES 3.0 has no base instance, so the first instance of a draw cannot be offset.
- A degenerate triangle (repeated index) produces no fragments; that is why repeating the last index is a safe way to pad
  a strip, and why restart can often be replaced by degenerate triangles in code that must also run on ES 2.0.
