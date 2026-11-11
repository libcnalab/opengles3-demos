# es31-indirect-draw

**Minimum:** OpenGL ES 3.1 · **Shaders:** GLSL ES 3.00 (drawing) and 3.10 (compute) · **Run:** `es31-indirect-draw [--headless --validate]`

## Purpose

A normal draw call carries its arguments in the call: `glDrawArraysInstanced(mode, first, count, instances)`. An
*indirect* draw call carries only the primitive mode (and the index type) and fetches the other arguments from a
**command structure stored in a buffer object**. The buffer is bound to `DRAW_INDIRECT_BUFFER` and the `indirect`
parameter is not a pointer but a **byte offset** into it (ES 3.2 §10.3.9). Because the arguments live in a buffer, the
GPU itself can write them, for example a compute shader that decides how many instances survive a visibility test,
without the CPU ever reading the result back.

The two command layouts are fixed by the specification (§10.5), all members are tightly packed 32-bit values:

```c
typedef struct { uint count, instanceCount, first, reservedMustBeZero; } DrawArraysIndirectCommand;          // 16 bytes
typedef struct { uint count, instanceCount, firstIndex; int baseVertex; uint reservedMustBeZero; }
        DrawElementsIndirectCommand;                                                                          // 20 bytes
```

`glDrawArraysIndirect` is defined as `glDrawArraysInstanced(mode, cmd->first, cmd->count, cmd->instanceCount)`;
`glDrawElementsIndirect` as `DrawElementsInstancedBaseVertex(mode, cmd->count, type, cmd->firstIndex * size-of-type,
cmd->instanceCount, cmd->baseVertex)`. Three details are easy to get wrong and are what this demo exercises:

- `firstIndex` is an **index** count, the byte offset into the element buffer is `firstIndex * sizeof(type)`, so it
  depends on whether the indices are `UNSIGNED_BYTE`, `UNSIGNED_SHORT` or `UNSIGNED_INT`.
- `baseVertex` is a signed value that is *added to every index after it was read*. ES 3.1 has no `glDrawElementsBaseVertex`,
  but it does honour `baseVertex` inside `DrawElementsIndirectCommand` (§10.5 describes the command through the
  BaseVertex equivalent). A negative `baseVertex` is fine as long as no sum becomes negative.
- `reservedMustBeZero` must be zero; "results are undefined if it is non-zero". The demo always writes zero. (In
  desktop OpenGL the same word is `baseInstance`; in ES it is reserved.)

## What the image shows

256 × 256 default framebuffer, a 4 × 4 grid of 64 × 64 cells (row 3 is the top row). Cells marked "split" show a state
before (left or top) and after (right or bottom) a change of the command buffer. Each instance gets the next entry of
a color palette (red, green, blue, yellow, magenta, ...) through a per-instance attribute.

| Row | Cell | What it shows |
|---|---|---|
| 3 | A0 | `glDrawArraysIndirect(TRIANGLES)` of command {3, 1, 8}: one skewed triangle |
| 3 | A1 | `TRIANGLE_STRIP`, command {4, 3, 11}: three squares in the colors of instances 0, 1, 2 |
| 3 | A2 | `TRIANGLE_FAN`, command {6, 2, 15}: two fans (a pie with a missing wedge), instances 0 and 1 |
| 3 | A3 | the command of A0 drawn on the left, a command with `instanceCount` 0 in the middle (draws **nothing**, no error), the command of A0 again on the right |
| 2 | E0 | `glDrawElementsIndirect`, `UNSIGNED_SHORT`: a square from the index list at `firstIndex` 0 |
| 2 | E1 | split: left `firstIndex` 6 (the lower-left triangle of the square); right `baseVertex` **-2** with the indices {10, 11, 12} (which reach the skewed triangle only after the negative base is added) |
| 2 | E2 | `baseVertex` 4: the same index list now selects the diamond |
| 2 | E3 | three half-diamonds (`firstIndex` 9, `baseVertex` 4, `instanceCount` 3) from a VAO whose color attribute has **divisor 2**: colors red, red, green (element = floor(instance / 2)) |
| 1 | T0 | `UNSIGNED_BYTE` indices, `firstIndex` 12: two lower-right triangles |
| 1 | T1 | `UNSIGNED_INT` indices, `firstIndex` 15, `baseVertex` 4: two left half-diamonds |
| 1 | V0 | the vertex shader ignores the attribute and computes a point on a circle from `gl_VertexID`: `DrawArraysIndirect` with `first` 4 gives vertex IDs 4..9, two instances |
| 1 | V1 | the same for `DrawElementsIndirect` with `baseVertex` 7: vertex ID = 7 + index |
| 0 | D0 | split: the command is rewritten as a whole by `glBufferSubData` between two draws at the **same offset**; left triangle, right two stacked quads |
| 0 | D1 | split: only the 4-byte `instanceCount` word is rewritten (`glBufferSubData` at offset + 4): one quad, then three |
| 0 | D2 | split (top / bottom): **a compute shader wrote the command** (instance count = number of non-zero flags: 3, then 5 after the flags were changed and the shader dispatched again) |
| 0 | D3 | the same for a `DrawElementsIndirectCommand` written by the same shader: for up to 3 instances the whole diamond (count 6, firstIndex 0), for more its upper half (count 3, firstIndex 9) |

All commands live in one buffer of 340 bytes. A guard word after the four arrays commands shifts the following
offsets away from multiples of 16 and 20 (the elements commands start at 68, 88, 108, ...): the only alignment the
specification demands is a multiple of 4 bytes (`sizeof(uint)`).

## Central calls

| Call | Role |
|---|---|
| `glBindBuffer(DRAW_INDIRECT_BUFFER, buf)` | selects the buffer the commands are read from; the binding is context state, **not VAO state** |
| `glDrawArraysIndirect(mode, offset)` | draws from a `DrawArraysIndirectCommand` at `offset` |
| `glDrawElementsIndirect(mode, type, offset)` | draws from a `DrawElementsIndirectCommand` at `offset`; indices come from the element buffer of the VAO |
| `glBufferSubData` | rewrites a whole command or a single word between draws; the next draw uses the new values |
| `glBindBufferBase(SHADER_STORAGE_BUFFER, 0, buf)` + `glDispatchCompute` | lets a shader write the commands: the *same buffer object* is the SSBO of the compute shader |
| `glMemoryBarrier(COMMAND_BARRIER_BIT)` | required between the compute stores and a draw that sources the commands (§7.11.2). The demo also sets `BUFFER_UPDATE_BARRIER_BIT`, which covers the mapped read-back of the commands |
| `glGetIntegerv(DRAW_INDIRECT_BUFFER_BINDING)` | reads the binding back |

## Spec

- ES 3.2 §10.3.9 Indirect Commands in Buffer Objects and Table 10.4; §10.3.7, §10.3.8 (vertex and index data in buffers);
  §10.3.2 Vertex Attribute Divisors.
- ES 3.2 §10.5 Drawing Commands Using Vertex Arrays: `DrawArraysIndirect`, `DrawElementsIndirect`,
  `DrawElementsInstancedBaseVertex` and the error lists. ES 3.1 §10.5 has the same commands.
- ES 3.2 §7.11.2 Shader Memory Access Synchronization (`COMMAND_BARRIER_BIT`, `BUFFER_UPDATE_BARRIER_BIT`), §7.8 Shader
  Storage Blocks, §6.2 / §6.3 (`BufferSubData`, mapping).
- ES 3.2 §12.2.2 and ES 3.1 §12.1.2 (transform feedback and drawing commands); ES 3.2 §2.3.1 (errors);
  Table 21.4 (`DRAW_INDIRECT_BUFFER_BINDING`).
- GLSL ES 3.00 §7.1 (`gl_VertexID`, `gl_InstanceID`); GLSL ES 3.10 §4.3.7 (buffer blocks, runtime-sized arrays) and §4.4.1.1.

The errors of `DrawArraysIndirect` (ES 3.1 §10.5, unchanged in 3.2) and how the demo provokes each one in isolation:

| Rule | Provoked by |
|---|---|
| `INVALID_OPERATION` if zero is bound to `DRAW_INDIRECT_BUFFER` | binding buffer 0 |
| `INVALID_OPERATION` if zero is bound to `VERTEX_ARRAY_BINDING` | the default VAO (no client arrays for indirect draws) |
| `INVALID_OPERATION` if zero is bound to **any enabled vertex array** | an enabled attribute whose buffer binding is zero (possible on a non-default VAO because a NULL pointer is accepted); disabling that array removes the error |
| `INVALID_OPERATION` if the command would be read beyond the end of the buffer | offsets `size - 12` (arrays, 16 bytes needed), `size - 16` (elements, 20 needed), `size`, far beyond; the offsets `size - 16` / `size - 20` that fit exactly are valid |
| `INVALID_VALUE` if `indirect` is not a multiple of 4 | offsets 49, 50, 51 (arrays) and 69, 70, 71 (elements), each otherwise valid |
| `DrawElementsIndirect`: `INVALID_OPERATION` if `ELEMENT_ARRAY_BUFFER` is zero | a VAO without an element buffer; `DrawArraysIndirect` with the same VAO is fine |
| `DrawElementsIndirect`: `INVALID_ENUM` for an invalid `type` | `GL_FLOAT` |
| transform feedback active and not paused | see "Pitfalls" |

## Shader concepts

`shaders/indirect.vert` (GLSL ES 3.00) never sees the commands. It has a position input (`layout(location = 0) in vec2`),
a per-instance color input (location 1) and three uniforms that only *place* the result in the viewport:
`gl_Position = p * u_scale + u_origin + float(gl_InstanceID) * u_pitch`. The command decides which vertices are fetched,
how many and how often, so the same command can be drawn into different places (cell A3 does exactly that). With
`u_mode == 1` the position is computed from `gl_VertexID` alone (`0.8 * (cos, sin)(1.3 * id)`), which makes the vertex ID
the GL delivers directly visible: `first + i` for arrays and `baseVertex + index` for elements.

`shaders/commands.comp` (GLSL ES 3.10) has two shader storage blocks: `Commands { uint words[]; }`, the indirect buffer
itself, and `Flags { readonly uint visible[]; }`. One invocation counts the non-zero flags with `visible.length()` and writes
a `DrawArraysIndirectCommand` and a `DrawElementsIndirectCommand` at the word offsets given by two uniforms. One
invocation does all the work, so the result does not depend on invocation order.

## Validation

`validate()` records 73 checks:

1. **Pixels, 21 regions.** For every draw the CPU interprets the same command values: the vertex IDs of §10.5, instances,
   the divisor of the VAO, primitive assembly for `TRIANGLES`, `TRIANGLE_STRIP` and `TRIANGLE_FAN`. Each pixel of each region
   is compared with the model unless it is within 1 pixel of a triangle edge (where the rasterizer's tie-breaking rules
   decide). Palette colors are exact multiples of 1/255, so the tolerance is the 8-bit quantization of ±1; the measured
   maximum deviation is 0 on all three drivers.
2. **Compute output.** Both compute-written commands are mapped and compared word for word with a CPU emulation of the
   shader (for 3 and for 5 visible flags), including the guard words around them and the first word of the following
   command, which must not change.
3. **Negative controls.** Nine model mutations (wrong `first`, one instance fewer, the negative `baseVertex` ignored,
   divisor 1 instead of 2, `firstIndex` unscaled, the vertex ID off by one, the stale command from before the compute
   dispatch or before `glBufferSubData`, `instanceCount` 1 in the empty middle command of A3) must *not* match the pixels
   (at least 20 mismatching pixels each). That proves each comparison can fail and that the effect under test (e.g. the
   compute shader's write, the `glBufferSubData` update) really reached the picture.
4. **State.** `DRAW_INDIRECT_BUFFER_BINDING` is 0 initially, equals the buffer after the bind, is unchanged by switching
   between VAOs (it is not VAO state, Table 21.4) and is 0 after binding buffer 0; `BUFFER_SIZE` of the command buffer.
5. **Errors.** The table above, each case isolated and paired with a valid control call.

## Pitfalls

- **The barrier is not optional.** Without `COMMAND_BARRIER_BIT` the stores of the compute shader are not guaranteed to be
  visible to `Draw*Indirect` (§7.11.2). On a desktop-class driver the missing barrier often "works", so a correct picture
  is no proof that the barrier is there; the demo can only show that the barrier version works.
- **`indirect` is an offset, not a pointer.** `glDrawArraysIndirect(mode, nullptr)` means offset 0 only because a buffer is bound.
  With `DRAW_INDIRECT_BUFFER` 0 the call is an error, not a client-memory read (ES 3.1 has no client-side indirect commands).
- **`firstIndex` is not a byte offset.** For `UNSIGNED_INT` indices it is multiplied by 4, for `UNSIGNED_BYTE` by 1.
- **Transform feedback.** ES 3.1 says that *any* indirect draw while transform feedback is active and not paused is
  `INVALID_OPERATION`. ES 3.2 relaxed this: a draw is an error only if its mode is not allowed by Table 12.1 for the
  `primitiveMode` of `glBeginTransformFeedback`, and any mode may be used while paused. The demo asserts the rules that hold in
  both versions (POINTS while recording TRIANGLES; paused) and chooses the expectation for the matching mode from the context
  version. Mesa applies the ES 3.2 rule in every ES context, also a native ES 3.1 one (softpipe) and under `MESA_GLES_VERSION_OVERRIDE=3.1`; there the ES 3.1 expectation is
  reported as a known deviation (XFAIL) on "Mesa".
- **Out-of-range vertex data is not checked.** The "beyond the end of the buffer" error is about the command structure only; vertex or
  index data read past the end of their buffers is undefined without robust buffer access (§10.3.5).
- **The VAO matters.** `DrawElementsIndirect` takes its indices from the `ELEMENT_ARRAY_BUFFER` stored in the bound VAO, and the
  UBYTE/USHORT/UINT cells use three VAOs that differ only in that binding.
