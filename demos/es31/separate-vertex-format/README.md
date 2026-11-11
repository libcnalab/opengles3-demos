# es31-separate-vertex-format

**Minimum:** OpenGL ES 3.1 · **Shaders:** GLSL ES 3.00 · **Run:** `es31-separate-vertex-format [--headless --validate]`

## Purpose

In ES 3.0 one call, `glVertexAttribPointer(index, size, type, normalized, stride, pointer)`, describes everything about a
vertex attribute: what one element looks like (size, type, normalized), which buffer it is in (the current `ARRAY_BUFFER`),
where it starts (`pointer`) and how far apart the elements are (`stride`). Two attributes that live in the same buffer
repeat the buffer, the stride and a shifted pointer; pointing an attribute at a different buffer means describing it again.

ES 3.1 splits that call into four independent pieces (ES 3.2 §10.3.1, §10.3.2):

| Command | State it sets | Belongs to |
|---|---|---|
| `glVertexAttribFormat(attrib, size, type, normalized, relativeoffset)` and `glVertexAttribIFormat(attrib, size, type, relativeoffset)` | what **one element** is: size, type, normalized or integer, and its byte offset *inside one vertex* | the attribute |
| `glVertexAttribBinding(attrib, bindingindex)` | **which binding** the attribute reads from (initially binding *i* for attribute *i*) | the attribute |
| `glBindVertexBuffer(bindingindex, buffer, offset, stride)` | **where**: the buffer, the byte offset of the first element and the stride | the binding |
| `glVertexBindingDivisor(bindingindex, divisor)` | **how often** it advances (instancing) | the binding |

An element of an attribute is fetched from `binding.offset + attribute.relativeoffset + i * binding.stride` (§10.3.7). One
binding can feed several attributes; an attribute can be moved to another binding; a binding can be pointed at another buffer
**without touching a single format**. Because a divisor belongs to the binding, every attribute on a binding advances at the
same rate. The old commands still work and are *defined* as a sequence of the new ones: `glVertexAttribPointer` is
`Format(index, ..., 0)` + `Binding(index, index)` + `BindVertexBuffer(index, ARRAY_BUFFER, pointer, effectiveStride)`, and
`glVertexAttribDivisor(i, d)` is `Binding(i, i)` + `BindingDivisor(i, d)`. The two APIs therefore act on the same state and can
be mixed, which is where the surprises are.

All of this is vertex array object state: the new commands need a non-default VAO bound and raise `INVALID_OPERATION` otherwise.
That is a difference from the legacy commands, which on the default VAO manage client-memory arrays.

## What the image shows

256 × 256 default framebuffer, 4 columns × 3 rows of 64 × 85 cells (row 2 is the top row, one pixel row at the very top is
background). Every cell holds one triangle with a different color at each corner (smooth color gradients), or colored squares;
all positions and colors are fetched by the vertex fetch unit under test.

| Cell | What it shows |
|---|---|
| 0 | **one binding, three attributes.** Buffer A starts with 24 junk bytes; binding 0 has offset 24 and stride 16. Position (float ×2) at relative offset 0, color (normalized ubyte ×4) at 8, a small per-vertex displacement (normalized short ×2) at 12 |
| 1 | the **same VAO and formats**, only `glBindVertexBuffer(0, B, 0, 24)`: another buffer, offset and stride (the vertices of B have 8 bytes of padding). No `Format` or `Binding` call is made |
| 2 | **two bindings.** Binding 0 holds the quad (divisor 0), binding 1 holds a struct per *instance* {offset, color} (divisor 1); a color attribute at relative offset 8 and an offset attribute at 0 share binding 1. Four instances: red, green (bottom), blue, yellow (top) |
| 3 | the same, but `glVertexBindingDivisor(1, 2)`: the two attributes on binding 1 now advance every second instance, so instances 0 and 1 show element 0 (red) and instances 2 and 3 element 1 (green): only the bottom row remains |
| 4 | `glVertexAttribBinding(offset, 2)`: the offset attribute moves to binding 2, which reads another buffer (positions along a diagonal) with its own divisor; the color attribute stays on binding 1. Four squares on a diagonal with the colors of binding 1 |
| 5 | **pure integer formats** (`glVertexAttribIFormat`). Three bands; a band is green only if the integer the shader received is exactly the expected one: `uvec2` from `UNSIGNED_BYTE ×2`, an `int` from `INT` holding 2^24 + 1 (a float conversion would give 2^24 and turn the band red), an `int` from `SHORT` holding -3 (sign extension) |
| 6 | **legacy** `glVertexAttribPointer` defines attributes 0 and 1 (stride 12, offsets 16 and 24 in buffer L1) |
| 7 | the same VAO after `glBindVertexBuffer(0, L2, 0, 12)`: because the legacy call put attribute 0 on *binding 0*, the new command re-points it, while attribute 1 (binding 1) still reads the colors of L1: L2's positions with L1's colors |
| 8 | **legacy after new-style** for one attribute. The offset attribute was first set up the new way (binding 5, buffer X), then with `glVertexAttribPointer` and buffer Y. The legacy call re-maps the attribute to binding 2 and resets its relative offset, so the triangle is shifted by Y's offset (+0.15, +0.1), not by X's |
| 9 | **two bindings, one buffer.** Buffer S is interleaved {x, y, rgba}; binding 0 starts at offset 0 (positions), binding 1 at offset 8 (colors), both with stride 12 |
| 10 | the **highest binding indices**: attributes bound to `MAX_VERTEX_ATTRIB_BINDINGS - 1` (positions) and `- 2` (colors) |
| 11 | the **limits in use**: a vertex stride of `MAX_VERTEX_ATTRIB_STRIDE` (2048 bytes on all drivers tested) with the color at relative offset 2044 (the largest multiple of four not above `MAX_VERTEX_ATTRIB_RELATIVE_OFFSET - 3`, so the four color bytes end exactly at the stride) |

## Central calls

| Call | Role |
|---|---|
| `glVertexAttribFormat`, `glVertexAttribIFormat` | element format and relative offset |
| `glVertexAttribBinding` | attribute -> binding |
| `glBindVertexBuffer` | binding -> buffer, offset, stride |
| `glVertexBindingDivisor` | binding -> divisor |
| `glVertexAttribPointer`, `glVertexAttribIPointer`, `glVertexAttribDivisor` | the legacy equivalents, mixed with the new ones |
| `glGetVertexAttribiv(VERTEX_ATTRIB_BINDING / RELATIVE_OFFSET / ...)`, `glGetIntegeri_v(VERTEX_BINDING_BUFFER / STRIDE / DIVISOR / OFFSET)`, `glGetInteger64i_v(VERTEX_BINDING_OFFSET)` | read the state back |

## Spec

- ES 3.2 §10.3.1 (Table 10.3; the commands above and their errors; the equivalence of `VertexAttrib*Pointer`),
  §10.3.2 (divisors), §10.3.3 (missing components), §10.3.7 (the address formula).
- ES 3.2 §10.4 (VAOs), §10.5 (the instance element index `floor(instance / divisor)`), §10.6 (queries; the queries
  `VERTEX_ATTRIB_ARRAY_BUFFER_BINDING` and `VERTEX_ATTRIB_ARRAY_DIVISOR` are mapped through `VERTEX_ATTRIB_BINDING` to
  `VERTEX_BINDING_BUFFER` / `VERTEX_BINDING_DIVISOR`), §10.7 (initial values).
- ES 3.2 Table 21.3 (state per VAO: `VERTEX_BINDING_OFFSET` is a 64-bit value queried with `GetInteger64i_v`; the initial
  `VERTEX_BINDING_STRIDE` is 16), Table 21.42 / 21.44 (minimums: 16 bindings, relative offset 2047, stride 2048, 16 attributes).
- ES 3.2 §2.2.2 (a 64-bit value that does not fit the returned type is clamped), §2.3.1 and §20.1 (errors, indexed queries).
- ES 3.1 §10.3.1 contains the same commands and error lists; the only difference that matters here is the wording of the
  `BindVertexBuffer` name error, which is the same for names that were never generated or have been deleted.
- GLSL ES 3.00 §4.3.4, §4.3.8.1: `layout(location = n) in ...`. The shaders are `#version 300 es`: nothing in the *shader* is new in 3.1.

Errors of the new commands (all provoked on a non-default VAO unless noted, each with a valid control call):

| Command | `INVALID_VALUE` | `INVALID_ENUM` | `INVALID_OPERATION` |
|---|---|---|---|
| `glVertexAttribFormat` | `attribindex` = MAX_VERTEX_ATTRIBS; size 0 or 5; `relativeoffset` = MAX_VERTEX_ATTRIB_RELATIVE_OFFSET + 1 | `UNSIGNED_INT_10F_11F_11F_REV` | packed type with size 3; **default VAO** |
| `glVertexAttribIFormat` | `attribindex`, size 5, `relativeoffset` as above | `FLOAT`, `HALF_FLOAT`, packed | **default VAO** |
| `glVertexAttribBinding` | `attribindex` or `bindingindex` out of range | | **default VAO** |
| `glBindVertexBuffer` | `bindingindex` = MAX_VERTEX_ATTRIB_BINDINGS; negative offset or stride; stride = MAX_VERTEX_ATTRIB_STRIDE + 1 | | a name never returned by `glGenBuffers`, a deleted name; **default VAO** |
| `glVertexBindingDivisor` | `bindingindex` out of range | | **default VAO** |
| `glGetIntegeri_v(VERTEX_BINDING_*, index)` | index = MAX_VERTEX_ATTRIB_BINDINGS | | |

The values *at* the limits (relative offset = MAX, stride = MAX, binding = MAX - 1, attribute = MAX - 1) are valid. A name from
`glGenBuffers` that has never been bound is a valid argument of `glBindVertexBuffer` (and `VERTEX_BINDING_BUFFER` reports it).
An erroring command changes nothing (§2.3.1); the demo checks that for one `Format` and one `BindVertexBuffer` call.

## Shader concepts

`shaders/basic.vert` has three inputs and no knowledge of buffers: `layout(location = 0) in vec2 a_pos`,
`layout(location = 1) in vec4 a_color`, `layout(location = 2) in vec2 a_offset`, and computes
`gl_Position = a_pos * u_scale + a_offset`. When the offset array is disabled the shader sees the *current* generic attribute
value, which is the initial (0, 0, 0, 1), so the offset is zero. A normalized `UNSIGNED_BYTE` ×4 attribute arrives as four values
in [0, 1]; a normalized `SHORT` as c / 32767.

`shaders/integer.vert` declares `uvec2 a_cell`, `int a_big`, `int a_neg` (locations 3 to 5) and compares each with a uniform;
integer inputs must be fed with `glVertexAttribIFormat` (or `glVertexAttribIPointer`), a `glVertexAttribFormat` array would give
undefined values for them (§10.3.1).

## Validation

`validate()` records 94 checks:

1. **Pixels, 12 cells.** The CPU model holds each attribute as (format, relative offset, buffer bytes, binding offset, stride,
   divisor) and fetches element `floor(instance / divisor)` or the vertex index from the *same byte arrays* that were
   uploaded, using the address formula of §10.3.7. Triangles are rasterized with Gouraud shading on the CPU; pixels within 1 pixel
   of an edge are skipped. Tolerance: 3/255. The specification states no accuracy for the interpolation of colors, so 3/255 is a sanity bound and
   the measured maximum (1/255 on all drivers tested) is printed with every cell.
2. **Negative controls.** Eight mutations of the model must *not* match the pixels: binding offset 0 instead of 24 (the junk bytes are read), the color
   at relative offset 0, stride 16 instead of 24, divisor 1 instead of 2, the offset attribute left on binding 1, binding 0 not
   re-pointed to L2, buffer X instead of Y, the second binding of cell 9 at offset 0 (at least 30 mismatching pixels each); and one GL draw of the integer cell with the
   expected value 16777216, which must turn exactly the INT band red.
3. **State.** Limits against their minimums; the initial state of a fresh VAO (attribute *i* on binding *i*, relative offset 0;
   every binding with buffer 0, offset 0, **stride 16**, divisor 0); the effect of each of the five new commands on every
   state variable they set (and on those they must not touch); the shared divisor of two attributes on one binding; the mapped
   queries `VERTEX_ATTRIB_ARRAY_BUFFER_BINDING` and `..._DIVISOR`.
4. **Legacy interplay.** `glVertexAttribPointer` after the new commands on the same attribute resets `VERTEX_ATTRIB_BINDING`
   to the index, `RELATIVE_OFFSET` to 0, the binding's buffer, offset and effective stride, leaves `VERTEX_ATTRIB_ARRAY_STRIDE`
   at the *argument* (0) and does not change the divisor; `glVertexAttribDivisor` moves the attribute to its own binding and sets that binding's
   divisor, leaving the old binding alone; `glVertexAttribIPointer` sets the integer flag.
5. **Errors** (table above) and the isolation of the fresh VAO from all other changes.

Known deviation reported by the demo (Mesa, all three drivers): `VERTEX_BINDING_OFFSET` is a 64-bit state value (Table 21.3,
`GetInteger64i_v`) but a binding offset of 2^33 + 8 reads back as 8; the demo also asserts that `GetIntegeri_v` clamps it to
`INT_MAX` (§2.2.2). Both are `expect_or_known_deviation` checks on "Mesa".

## Pitfalls

- **The default VAO.** `glBindVertexBuffer`, `glVertexAttribFormat` and the rest need a non-default VAO bound. Code written for ES 3.0
  with the default VAO and `glVertexAttribPointer` client arrays keeps working; mixing it with the new commands does not.
- **Initial stride 16.** A fresh binding has stride 16, not 0, and offset 0, not "tight". The legacy call computes the effective stride from size and
  type when `stride` is 0; `glBindVertexBuffer` does not (stride 0 would mean something unspecified).
- **`VERTEX_ATTRIB_ARRAY_STRIDE` is not the binding stride.** It reports the argument of the last `glVertexAttribPointer` (0 for "tight"); the
  stride the fetch uses is `VERTEX_BINDING_STRIDE`.
- **The legacy call undoes the new commands.** `glVertexAttribPointer(i, ...)` forces attribute *i* onto binding *i*, resets its relative offset
  and takes the buffer from `ARRAY_BUFFER`; so does `glVertexAttribDivisor(i, d)` for the binding. After it, an earlier `glVertexAttribBinding` or
  `glBindVertexBuffer` for that attribute no longer matters (cell 8). Conversely `glBindVertexBuffer(i, ...)` re-points an attribute that was
  defined the legacy way (cell 7), because both name the same binding.
- **A divisor belongs to the binding.** Changing it changes every attribute on that binding at once (cell 3); `glVertexAttribDivisor` moves its
  attribute to its own binding first, so it does not.
- **Integer inputs need `IFormat`.** A shader `int`/`uvec` input fed by a float format reads undefined values.
- **No size checks.** `glBindVertexBuffer` accepts any non-negative offset (also one beyond the buffer); reading past the buffer is only
  checked at draw time, and is undefined without robust buffer access.
