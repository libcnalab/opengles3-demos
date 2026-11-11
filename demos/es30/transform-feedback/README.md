# es30-transform-feedback

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es30-transform-feedback [--headless --validate]`

## Purpose

Transform feedback records the outputs of the vertex shader into buffer objects *before* rasterization. The vertex
shader becomes a small compute stage: a GPU particle system steps its particles, a mesh is warped once and kept, a
generator produces vertex data from `gl_VertexID` alone. A later pass uses the recorded buffer as ordinary vertex data.
In ES 3.0 this is the main core way to get computed vertex data back into the pipeline without a round trip through the
CPU (there is no compute shader before ES 3.1, rendering to floating-point textures is not core, and there is no
`glDrawTransformFeedback`: the number of primitives to draw comes from a primitive query or is known in advance).

The demo does it four times and then looks at every rule around it: buffer modes, buffer ranges, transform feedback
objects, pause and resume, `TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN` queries, introspection, `RASTERIZER_DISCARD` and the
errors the specification defines while a capture is active.

## What the image shows

The 256 × 256 default framebuffer is a 2 × 2 grid of 128 × 128 cells. Every cell is a *second pass* that draws what a
first, capture-only pass recorded. All capture passes run with `RASTERIZER_DISCARD` enabled and a loud red fragment
color, so a single leaked pixel would be visible and fail the comparison.

| Cell | Capture pass | Second pass |
|---|---|---|
| top left (A) | `warp.vert` displaces 16 quads (96 vertices) by `p += (0.25 y², -0.125 x²)`; `INTERLEAVED_ATTRIBS`, varyings `v_pos` + `v_color` (20 bytes per vertex), captured as `TRIANGLES` on the default transform feedback object | `glDrawArrays(TRIANGLES, 0, 3 × primitives written)` of the captured buffer: sixteen curved flat-colored quads |
| top right (B) | `step.vert` moves 32 particles (gravity, floor bounce) 11 times. Two transform feedback objects, each with its own buffer attached, ping-pong: step *k* reads buffer *k mod 2* through a vertex array and records into the other one | the particles of the final buffer as points: a fountain that has bounced off the floor in its lowest row |
| bottom left (C) | 8 points; the capture is *paused* after 3 points, a point and a line are drawn while paused, then resumed for the last 3 | the 6 recorded points: a gap where the 2 missing points would be |
| bottom right (D) | `stream.vert` generates 12 lattice points from `gl_VertexID`; `SEPARATE_ATTRIBS`: `v_pos` -> binding 0, `v_color` -> 1, `v_size` -> 2, as three ranges at byte offsets 0 / 100 / 260 of **one** buffer (`glBindBufferRange`) | the three ranges bound as three attribute arrays: 12 squares of side 6, 8 and 10 |

## Central calls

| Call | Role |
|---|---|
| `glTransformFeedbackVaryings(program, count, names, mode)` | which vertex outputs are recorded, in which order, in which mode. **Before** `glLinkProgram`. |
| `glBindBufferBase(TRANSFORM_FEEDBACK_BUFFER, i, buf)` / `glBindBufferRange(..., offset, size)` | the destination of binding point *i*; offset and size must be multiples of 4 |
| `glBeginTransformFeedback(POINTS \| LINES \| TRIANGLES)` / `glEndTransformFeedback` | the capture window; the mode restricts what may be drawn |
| `glPauseTransformFeedback` / `glResumeTransformFeedback` | draws in between are not recorded; the capture continues right after the last record |
| `glGenTransformFeedbacks` / `glBindTransformFeedback` / `glIsTransformFeedback` / `glDeleteTransformFeedbacks` | transform feedback *objects*: the indexed buffer bindings (and the active/paused flags) belong to the object |
| `glBeginQuery(TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN)` / `glGetQueryObjectuiv` | how many primitives were really recorded |
| `glGetTransformFeedbackVarying`, `glGetProgramiv(TRANSFORM_FEEDBACK_*)` | introspection of the declared list |
| `glGetIntegeri_v`, `glGetInteger64i_v` (`TRANSFORM_FEEDBACK_BUFFER_BINDING / _START / _SIZE`) | what is attached to each binding point |
| `glEnable(RASTERIZER_DISCARD)` | after the capture, throw the primitives away instead of rasterizing them |

## Spec

- ES 3.0 §2.15 Transform Feedback (§2.15.1 objects, §2.15.2 primitive capture), §2.16 Primitive Queries, §2.14 Asynchronous
  Queries, §2.12.8 Output Variables (`TransformFeedbackVaryings`, `GetTransformFeedbackVarying`, link failures),
  §2.10.1.1 Binding Buffer Objects to Indexed Targets, §3.1 Discarding Primitives Before Rasterization,
  §6.1.7, §6.1.9 (indexed queries), §6.1.11, Table 6.24 (transform feedback state), Table 6.34 (limits), §2.5 (errors).
- **ES 3.1 §12.1 and ES 3.2 §12.2.1-12.2.2, Table 12.1, §12.3**: three rules changed in ES 3.2, see below.
- GLSL ES 3.00 §4.3.6 (integer vertex outputs must be `flat`), §7.1 (`gl_VertexID`, `gl_InstanceID`, `gl_Position`).

The rules, shortened:

- *Layout.* In `INTERLEAVED_ATTRIBS` all varyings go into the buffer at binding point 0, one vertex after the other, in the
  order of the list; arrays element by element, vectors component by component. In `SEPARATE_ATTRIBS` varying *i* goes to
  binding point *i*, tightly packed. `gl_Position` and `gl_PointSize` can be named like user outputs.
- *Order and completeness.* Primitives are recorded in the order they reach the capture stage (all vertices of instance 0
  before instance 1). Incomplete primitives are not recorded. A paused capture records nothing; after the resume the
  recording continues immediately after the last record.
- *State.* Begin needs a buffer at every binding point that the program writes, a current program with at least one varying,
  and a capture that is not active. While capturing, `UseProgram`, `BindBufferBase/Range(TRANSFORM_FEEDBACK_BUFFER)`,
  `BindTransformFeedback` and `LinkProgram` of the program in use are `INVALID_OPERATION`; while *paused* only `UseProgram`
  and `BindTransformFeedback` are allowed (another object may even run a capture meanwhile), and `ResumeTransformFeedback`
  needs the original program current again.
- **The three rules that depend on the version.** ES 3.0 and 3.1: `DrawArrays` / `DrawArraysInstanced` need a mode *identical*
  to the capture primitive, every other drawing command (`DrawElements`, `DrawElementsInstanced`, `DrawRangeElements`) is
  `INVALID_OPERATION` "regardless of mode", and a draw that would overflow a buffer (or the `BindBufferRange` range) is
  `INVALID_OPERATION` and records nothing. ES 3.2 replaced all three: Table 12.1 allows `LINE_STRIP` / `LINE_LOOP` for `LINES`
  and `TRIANGLE_STRIP` / `TRIANGLE_FAN` for `TRIANGLES` (and indexed draws of an allowed mode), strips, fans and loops are
  recorded as separate lines or triangles, and a primitive that does not fit is silently dropped and not counted (the
  overflow error was removed because it is "very difficult when using geometry shaders", ES 3.2 changelog). The demo asserts
  the rule of the context version it runs on (`es-max` runs a 3.2 context, `es-min` reports 3.0).

## Shader concepts

All five capture shaders are vertex shaders paired with the same do-nothing fragment shader (a program needs both in ES 3.0).
`step.vert` and `warp.vert` use only values and operations that are exact in a 32-bit float (small dyadic fractions,
additions and multiplications), so the CPU model reproduces every bit whether or not the GPU fuses multiply-adds. `probe.vert`
captures `flat out int` values (`gl_VertexID`, `gl_InstanceID`), an array output and `gl_Position` to show the layout rules.
`stream.vert` has no inputs at all. `render.vert` has `layout(location)` 0, 2 and 3 so that the very same vertex arrays that fed
the capture, or the capture's output buffers, can drive it.

## How it is validated

- **Pixels.** Every pixel of every cell is compared with a CPU rendering model: flat-colored triangles (cell A) and point
  squares (B, C, D) built from the *CPU-computed* data, never from what was read back. Pixels within 1 px of a triangle edge
  and within 0.25 px of a point edge are skipped (rasterizer tie-breaking, point position snapping); the largest deviation on
  radeonsi, llvmpipe and zink is 0/255 (tolerance ±1, the float -> 8-bit conversion).
- **Bytes.** Every capture buffer is read back through `glMapBufferRange` and compared *bit for bit* with the CPU model: the
  96 warped vertices (A), both ping-pong buffers after 11 and 10 steps (B, including the floor-bounce branch, which the model
  counts), the six records around the pause (C), and the three streams in their ranges (D). Buffers are pre-filled with `0xA5`, so
  "the other bytes were not touched" is checked too: the gaps between the ranges, the space after a short capture, the bytes
  before a range offset and after a range end.
- **Negative controls.** The pixels of each cell must *disagree* with the model of a different configuration (no warp; one step
  fewer; all 8 points, as if the pause had no effect; all point sizes 8), the two ping-pong buffers must differ from each other and
  from the initial state, the discarded capture must disagree with the model of the visible one, and the glClear-ignored cell
  must change when RASTERIZER_DISCARD is turned off. Positive controls precede the error cases (the same call with valid state
  raises no error), so a failure is never caused by a broken setup.
- **Introspection and state.** Varying names, types, sizes (including the array `v_trail`: `FLOAT_VEC2`, size 2, and
  `gl_Position`), truncation with a small `bufSize`; `TRANSFORM_FEEDBACK_BUFFER_MODE`, `_VARYINGS`, `_VARYING_MAX_LENGTH`; the
  initial state (default object bound, nothing attached, discard off); per-object `TRANSFORM_FEEDBACK_BUFFER_BINDING`, `_START`,
  `_SIZE`; `TRANSFORM_FEEDBACK_ACTIVE` / `_PAUSED` through every transition, including binding another object while paused.
- **Queries.** Counts 8 (3 + 5 points), 2, 4, 6 (paused draws not counted), 352 (11 steps x 32 points across two objects), 0 for
  draws while the capture is inactive; `glBeginQuery` resets the count; `CURRENT_QUERY`; the active query cannot be read.
- **Errors** (positive control first, then the failing call, then the state it must not have changed): pairing of
  begin/end/pause/resume, `TRIANGLE_STRIP` as `primitiveMode` (`INVALID_ENUM`), mode mismatches, indexed draws and overflow
  (version dependent, see above), state changes while active or paused, `BeginTransformFeedback` without a buffer / program /
  varyings and with a missing binding point in `SEPARATE_ATTRIBS` mode, `BindBufferRange` offset 2 / size 6 / size 0 / offset -4 /
  index `MAX`, `glTransformFeedbackVaryings` with `MAX + 1` separate varyings, link failures for an undeclared and a duplicate
  varying, `glGetTransformFeedbackVarying` with index = count.
- **Driver deviations (reported as XFAIL, strict everywhere else).** radeonsi raises `INVALID_OPERATION` for `glUseProgram` while
  capturing but still makes the rejected program current (ES 3.0 §2.5: a command that generates an error has no effect). On an
  ES 3.0/3.1 context Mesa accepts `TRIANGLE_STRIP`, `TRIANGLE_FAN`, `LINE_STRIP` and `LINE_LOOP` draws while capturing triangles
  or lines (it implements the ES 3.2 rule in every version; `MESA_GLES_VERSION_OVERRIDE` changes only the reported version).
  ANGLE (Vulkan backend) resumes a paused capture while a program other than the one that began it is current, although
  ES 3.0 §2.15.2 (and ES 3.2 §12.2.2) require `INVALID_OPERATION` "by ResumeTransformFeedback if the program object being used by
  the current transform feedback object is not active". When that happens the demo pauses again, so the following check
  (resume with the original program) still tests only its own rule.

## Run

```
./build/debug/bin/es30-transform-feedback --headless --validate --capture transform-feedback.png
```

## Pitfalls

- Varyings must be declared **before** linking; changing the list has no effect until the next link.
- Reading a buffer that is also an active capture target (as vertex data, `glMapBufferRange`, `glReadPixels` into it) is
  undefined (ES 3.0 §2.15.2). The ping-pong therefore alternates buffers: a step reads buffer *k mod 2* and writes the other.
  The demo reads captured data back only after `glEndTransformFeedback`.
- Without `glDrawTransformFeedback` the draw count comes from a `TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN` query (a pipeline
  stall) or from knowledge about the draw: `count × instances / vertices-per-primitive`, in the right primitive.
- A capture needs all its binding points filled when it begins; `BindBufferBase` binds the *whole* buffer, `BindBufferRange` wants
  offset and size divisible by 4 (the queries report START and SIZE as 0 after `BindBufferBase`).
- Binding state is per transform feedback object; buffers must be bound **before** `glBeginTransformFeedback`, and binding or
  program changes are errors while the capture runs. Pause/resume is the way to draw something unrelated in the middle.
- A program must have a fragment shader even if nothing is rasterized; `RASTERIZER_DISCARD` also turns `glClear` and
  `glClearBufferfv` into no-ops, so clear *before* enabling it.
- Integer vertex outputs must be `flat` (GLSL ES 3.00 §4.3.6). Mesa does not reject a missing `flat` on a vertex output (see
  `docs/KNOWN_LIMITATIONS.md`), but other drivers may.
- The ES 3.0 and ES 3.2 rules for draw modes and overflow differ (see above); portable code should stay inside both: draw
  exactly the primitive named in `glBeginTransformFeedback` with `glDrawArrays*`, and size the buffers for the whole draw.
