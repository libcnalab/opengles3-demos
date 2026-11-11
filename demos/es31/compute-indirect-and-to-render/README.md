# es31-compute-indirect-and-to-render

**API introduced in:** OpenGL ES 3.1 · **Minimum context:** ES 3.1 · **Shaders:** GLSL ES 3.10 ·
**Run:** `es31-compute-indirect-and-to-render [--headless --validate]`

Spec numbering follows ES 3.2 (compute: section 17; indirect buffers: 10.3.9 and 10.5; barriers: 7.11.2).

## Purpose

A GPU can decide how much work to do next. In ES 3.1 two commands read their arguments from a buffer object instead of from the CPU: **`glDispatchComputeIndirect`** (three group counts from
`DISPATCH_INDIRECT_BUFFER`) and **`glDrawElementsIndirect` / `glDrawArraysIndirect`** (a draw command from `DRAW_INDIRECT_BUFFER`). If a shader wrote those arguments, the CPU never has to read
a result back to launch the next stage, which is what the demo shows in three settings:

1. **Indirect dispatch** with group counts written by a previous compute pass: a compaction pass counts the "active" elements of an array and writes `ceil(n / 64)`, the next pass is launched with exactly that many
   groups; the group counts of an image fill are derived from `imageSize()` of the image; a table of counts is dispatched from ten different byte offsets, three of them with a zero count.
2. **Compute to render**: a compute shader generates the vertices and indices of a ring of triangles and the whole `DrawElementsIndirectCommand` (`count` from an atomic counter, `firstIndex = 2`, `baseVertex = 3`),
   and one `glDrawElementsIndirect` draws them.
3. **`glMemoryBarrierByRegion`** between fragment-shader passes whose pixels depend only on the same pixel of the previous pass.

## What the image shows

Four cells, each half of the window in both directions (128 x 128 in the default 256 x 256 window; named as they appear in the
capture PNG, whose first row is the top of the window). The layout follows the window size; the minimum is **128 x 128**, so
that every cell holds the whole 64 x 64 image at one pixel per texel, and smaller windows are skipped. The images and the buffer
of the ByRegion pipelines are addressed by `gl_FragCoord`, so they are allocated from the window size (the buffer rows are
`u_stride` = window width words apart). Beyond about 650 pixels across, the green channel of the buffer ramp
(`v / 256 / 28`) saturates; the red channel (`v % 256`) still differs from pixel to pixel and is compared.

| Cell | Content |
|---|---|
| top left | the compute-generated ring, drawn into the largest centered square of the cell: of 8 sectors the shader keeps those with `(s * 7 + 2) % 5 < 3`, i.e. five triangles, each with a palette color derived from the `uint` sector attribute |
| top right | a 64 x 64 image painted by an indirectly dispatched grid of 8 x 8 work groups (shown from the bottom-left corner of the cell at the largest integer number of pixels per texel that fits, 2 at 256 x 256): orange tiles with an x-ramp, blue tiles with a y-ramp, a white texel at the origin of every group |
| bottom left | five fragment-shader passes over one `r32ui` image (init, three read-modify-write passes, a final pass that turns the value into a color), `glMemoryBarrierByRegion(SHADER_IMAGE_ACCESS)` between them: stripes of 8 pixels |
| bottom right | upper strip: `imageStore` to an `r32f` image, then `texelFetch` of the same pixel (`TEXTURE_FETCH`), an XOR pattern; lower strip: a buffer store, then a buffer load of the same pixel (`SHADER_STORAGE`), a diagonal ramp |

## Central calls

`glDispatchComputeIndirect(offset)`, `glDrawElementsIndirect`, `glBindBuffer` with `DISPATCH_INDIRECT_BUFFER` / `DRAW_INDIRECT_BUFFER`, `glMemoryBarrier`
(`COMMAND`, `VERTEX_ATTRIB_ARRAY`, `ELEMENT_ARRAY`, `SHADER_STORAGE`, `TEXTURE_FETCH`, `FRAMEBUFFER`, `BUFFER_UPDATE`), **`glMemoryBarrierByRegion`**, `glBindBufferBase` (SSBO),
`glVertexAttribIPointer`. In GLSL: shader storage blocks with `std430` and runtime-sized arrays, `atomicAdd`, `gl_NumWorkGroups`, `gl_WorkGroupID`, `imageSize`, `imageLoad`, `imageStore`.

## Key ideas

- **One buffer, several roles.** A buffer object is just storage in ES 3.1. The vertex buffer is an SSBO for the compute shader and the `ARRAY_BUFFER` of the draw; the index buffer is an SSBO and the `ELEMENT_ARRAY_BUFFER`;
  the command buffer is an SSBO (with an atomic counter in its tail) and the `DRAW_INDIRECT_BUFFER`; the compaction work buffer holds the dispatch command in its first twelve bytes and the list after it.
- **Every consumer needs its own barrier bit.** The writes of a shader are not visible to the fixed-function fetches that follow unless `glMemoryBarrier` says so (section 7.11.2): `VERTEX_ATTRIB_ARRAY_BARRIER_BIT`
  for vertices sourced from buffers, `ELEMENT_ARRAY_BARRIER_BIT` for indices, `COMMAND_BARRIER_BIT` for `Draw*Indirect` and `DispatchComputeIndirect` arguments. Between two compute passes that communicate through an SSBO
  the bit is `SHADER_STORAGE_BARRIER_BIT`; the compaction chain needs it and `COMMAND` together because pass 2 reads the list through an SSBO and its group count through the indirect binding.
- **The command is 12 bytes at any 4-byte offset.** `DispatchComputeIndirect` reads three `uint` at `indirect`, which "must be a multiple of the size of a `uint`": the demo dispatches from the table `3 2 2 4 1 0 1 5 1 2 1 3` at byte
  offsets 0, 4, ..., 36, so the same words take part in several overlapping commands; `(4, 1, 0)`, `(1, 0, 1)` and `(0, 1, 5)` have a zero count and launch nothing.
- **Indirect draw fields are all live.** `firstIndex = 2` skips two junk indices and `baseVertex = 3` skips three junk vertices at the start of the buffers; a driver that ignored either field would draw wrong triangles (or none), so the
  rendered pixels prove that all five fields of the command that the shader wrote were used.
- **`glMemoryBarrierByRegion` is a narrower promise.** It only orders *fragment shader* accesses, and only within a small region of the framebuffer (possibly a single pixel), so it applies to pixels that depend on nothing but the same pixel of an earlier
  pass: "scatter/gather-like algorithms may or may not" be completed (section 7.11.2). Consequently only six bits are legal: `ATOMIC_COUNTER`, `FRAMEBUFFER`, `SHADER_IMAGE_ACCESS`, `SHADER_STORAGE`, `TEXTURE_FETCH`, `UNIFORM`;
  every other bit is `INVALID_VALUE`. A full-screen triangle (no inner diagonal) guarantees one fragment invocation per pixel, so "same pixel" is well defined.
- **Fragment shaders may lack images and storage blocks.** `MAX_FRAGMENT_IMAGE_UNIFORMS` and `MAX_FRAGMENT_SHADER_STORAGE_BLOCKS` can be 0 in ES 3.1 (4 in ES 3.2). The demo reads both and skips the corresponding ByRegion pipelines with a note (Mesa: 16 or 32).
- **The order of atomically appended triangles is not defined.** The ring is built with `atomicAdd` slots, so which triangle is first varies between runs and drivers; the picture and every check are independent of the order.

## Shader concepts

`compact_and_count.comp` (one invocation compacts the active indices and writes `ceil(n / 64), 1, 1` as a `DispatchIndirectCommand`) and `process_active.comp` (64 invocations per group, counts launched groups and invocations with `atomicAdd`) form
the compaction chain. `write_records.comp` writes the table, `record_probe.comp` records for each indirect command the number of launched groups and invocations, `gl_NumWorkGroups` and how often every linear group id ran. `img_setup.comp` derives
group counts from `imageSize()` of a `readonly writeonly` image (allowed: nothing is read or written) and `img_fill.comp` paints the tiles. `gen_geometry.comp` appends triangles with an `atomicAdd` slot, `finalize_command.comp` completes the
command, `star.vert` reads `vec2` and `uint` attributes. `chain.frag`, `tex_a.frag` / `tex_b.frag` and `ssbo_a.frag` / `ssbo_b.frag` are the three ByRegion pipelines.

## Validation (`--validate`)

47 checks (the same on radeonsi, llvmpipe and zink, at es-max and es-min):

- **Star (9).** The command the shader wrote is `{15, 1, 2, 3, 0}` and the atomic counter says 5; the index buffer holds `0, 1, 2, ...` after the two untouched junk indices; every surviving sector appears once with three vertices carrying its sector number; positions
  equal the CPU model (the measured difference is below 3e-7 NDC, the check allows 1e-3 because `sin`/`cos` accuracy is not specified); the junk vertices are untouched; the five triangle centroids have their palette colors; the three culled sectors, the center and a gap show
  the clear color (8-bit result, +-1). A negative control requires a drawn centroid to differ from the clear color.
- **Indirectly filled image (3).** `img_setup.comp` wrote `(8, 8, 1)`; all 4096 texels equal the CPU model (+-1, 8-bit quantization); six probes of the top-right cell.
- **ByRegion pipelines (6).** All pixels of the image chain (16384 at 256 x 256), of the texel-fetch strip (8192) and of the buffer strip (8192) equal the CPU model within 1; a control shows that a model that skipped three of the five passes differs at every pixel; no error for the six legal bits and for
  `ALL_BARRIER_BITS`.
- **Compaction chain (6).** The GPU counted 761 active elements of 1000 and wrote `(12, 1, 1)`; `glDispatchComputeIndirect(0)` launched exactly 12 groups and 768 invocations; the 761 squares are in the output in list order and the 263 words behind them keep their sentinel; a
  direct dispatch of a single group, the mutation, leaves 697 words wrong, so the comparison can fail.
- **Offsets (10).** For each byte offset 0, 4, ..., 36: the launched group count equals the product of the three words (including 0), each linear group id ran exactly once, `gl_NumWorkGroups` equals the record, and four invocations per group ran.
- **Errors (13).** Offsets 2, 6 (not a multiple of 4) and -4 (negative): `INVALID_VALUE`; offsets `size - 8`, `size` and 1 MiB (the 12-byte command would extend beyond the end of the 64-byte buffer), no buffer bound, no current program: `INVALID_OPERATION`; none of the eight erroring calls launched an
  invocation; `size - 12` is accepted and launches one group of four invocations; `glMemoryBarrier(ALL_BARRIER_BITS)` and `glMemoryBarrierByRegion(ALL_BARRIER_BITS)` raise no error.

## Spec ambiguities and how they were resolved

- **Unaligned offset: INVALID_VALUE.** The task text for this demo said "multiple of 4"; the error the specification names for an offset that is negative or not a multiple of the size of a `uint` is `INVALID_VALUE` (section 17), and only an
  offset whose command reaches beyond the buffer is `INVALID_OPERATION`. Both are asserted as written.
- **ES 3.1 typo.** The ES 3.1 text of the `DispatchComputeIndirect` errors names `DRAW_INDIRECT_BUFFER` for the "zero is bound" error; ES 3.2 corrects it to `DISPATCH_INDIRECT_BUFFER` (section 17). The demo follows ES 3.2.
- **Indirect counts above the limits.** "If any of num groups x/y/z is greater than MAX_COMPUTE_WORK_GROUP_COUNT ... the results are undefined": not tested.
- **`glMemoryBarrierByRegion` with an illegal bit.** Section 7.11.2 requires `INVALID_VALUE` for bits other than the six legal ones. meta-gl's `MemoryBarrierByRegionMask` has no enumerator for the other bits, so the case cannot be written without casting a raw bitfield
  (forbidden outside `tests/negative`); it is reported as an observation, not tested.
- **What ByRegion guarantees.** The text promises ordering only within an implementation-dependent region "as small as one framebuffer pixel". Every dependency in the pipelines is on the same pixel, which is the one case the text calls out (deferred shading), so a conformant
  driver must produce the CPU result. The demo cannot show a failure caused by a *missing* barrier: that is undefined behavior and usually works.

## Pitfalls

- `DrawElementsIndirect` needs a non-default vertex array object with the element buffer bound, and no client-side vertex arrays; the command is read from the buffer bound to `DRAW_INDIRECT_BUFFER` *at the time of the call*, the `indirect` argument is a byte offset into it.
- The element indices of an indirect draw are relative to `baseVertex` and the first one is `firstIndex * sizeof(type)` bytes into the element buffer, not into the vertex buffer.
- The std430 layout of a `vec2` + `uint` + padding struct is 16 bytes; `vec3` members would be padded to 16 too. Padding must match the `stride` given to `glVertexAttribPointer`.
- `COMMAND_BARRIER_BIT` is not implied by `SHADER_STORAGE_BARRIER_BIT`: the buffer is later read by the command processor, not by a shader.
- A `glMemoryBarrierByRegion` between passes is only sufficient if no pixel reads what another pixel wrote; a blur or a gather needs the full `glMemoryBarrier`.
- A write-after-read hazard exists across frames too (the first pass of the next frame stores to the image that the last pass of this frame loaded); the demo ends every pipeline with a barrier.
