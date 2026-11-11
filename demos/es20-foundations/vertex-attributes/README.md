# es20-vertex-attributes

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es20-vertex-attributes [--headless --validate]`

## Purpose

A vertex shader never sees buffers, only *generic vertex attributes*: numbered 4-component slots
that the GL fills before the shader runs. This demo takes the slot mechanism apart and checks every
step against numbers computed on the CPU:

1. which **data types** an array may use and how `glVertexAttribPointer`'s `normalized` flag turns
   stored integers into the floats the shader receives (BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, INT,
   UNSIGNED_INT, FIXED, FLOAT);
2. what the shader gets when an array has **fewer than four components** (the rest is `(0, 0, 0, 1)`);
3. how the same four quads can be stored **interleaved, planar, in one shared buffer, or packed** with
   `stride` and `offset`;
4. **constant (current) attribute values** set with `glVertexAttrib{1,2,3,4}f[v]` while the array is disabled;
5. **client-memory arrays**, which only the default vertex array object (VAO 0) may use;
6. how an attribute gets its **location**: `layout(location)`, `glBindAttribLocation` before the link, or
   the linker's choice read back with `glGetAttribLocation`;
7. **`glGetActiveAttrib`** enumeration and the **`glGetVertexAttrib*` / `glGetVertexAttribPointerv`**
   state queries.

## Reading the image

The 256 × 256 window is an 8 × 8 grid of 32-pixel cells (row 0 at the top). Each cell is one draw call
into its own viewport; `validate()` compares each cell's center pixel with the CPU value.

| Cells (col, row) | What they show |
|---|---|
| cols 0-1, rows 0-7 | One row per data type: UNSIGNED_BYTE, BYTE, UNSIGNED_SHORT, SHORT, UNSIGNED_INT, INT, FIXED, FLOAT. Left cell `normalized = TRUE`, right cell `normalized = FALSE` with the normalization done by the shader instead. For the integer types the pair is **identical**; for FIXED and FLOAT the flag is ignored, so those pairs are identical too. |
| (2, 1) | The edge of the signed-byte range: stored `-128, -127, -126` (magnified by the shader). Nearly black on purpose: ES 3.0 maps `-128` and `-127` both to `-1.0`. |
| col 3, rows 0-3 | FLOAT arrays with 1, 2, 3 and 4 components per vertex. |
| col 3, rows 4-6 | Client-memory arrays on VAO 0: (4) both arrays in CPU memory, (5) the same pointers after the CPU array was edited, (6) positions from a VBO and colors from CPU memory in one draw. |
| (3, 7), (4, 7) | The same program drawn with locations chosen by `glBindAttribLocation` (5 and 2) and with locations chosen by the linker. |
| cols 4-7, rows 0-3 | Four 2 × 2-cell panels showing the same picture - four colored quads - from four memory layouts: interleaved floats, planar in two buffers, planar in one buffer behind a header, packed `SHORT`/`UBYTE` vertices. |
| cols 4-7, rows 4-5 | Constant attribute values from `glVertexAttrib1f`, `1fv`, `2f`, `2fv`, `3f`, `3fv`, `4f`, `4fv`. |
| (4, 6), (5, 6) | An *enabled* array overrides the constant (green); after disabling it again the last constant is back (teal), proving that drawing from an array does not touch the current value. |

The alpha channel carries the attribute's `w`. Four cells have `w < 1` (0.5, 0.5, 0.25, 0.25), so `validate()`
makes alpha opaque after all other checks, to keep the PNG capture readable (the capture is taken after `validate()`); a
`glColorMask`-restricted clear does it, and a final check confirms that RGB was left untouched.

## Central calls

| Concern | Calls |
|---|---|
| Describe an array | `glBindBuffer(ARRAY_BUFFER)`, `glVertexAttribPointer(index, size, type, normalized, stride, offset)`, `glEnableVertexAttribArray` / `glDisableVertexAttribArray` |
| Constants | `glVertexAttrib1f` ... `glVertexAttrib4fv` |
| Locations | `glBindAttribLocation` (before `glLinkProgram`), `glGetAttribLocation`, `layout(location = N)` |
| Enumerate | `glGetProgramiv(ACTIVE_ATTRIBUTES / ACTIVE_ATTRIBUTE_MAX_LENGTH)`, `glGetActiveAttrib` |
| Query | `glGetVertexAttribiv`, `glGetVertexAttribfv`, `glGetVertexAttribPointerv`, `glIsVertexArray` |

The buffer that was bound to `ARRAY_BUFFER` **when `glVertexAttribPointer` ran** is copied into the
attribute's state (ES 3.0 §2.10.6). Binding another buffer later changes nothing; the demo checks it.

### Conversions (ES 3.0 §2.1.6, §2.9)

With `normalized = TRUE` unsigned values become `c / (2^b - 1)` (equation 2.1) and signed values become
`max(c / (2^(b-1) - 1), -1)` (equation 2.2) - an ES 3.0 *behaviour change*: `-128` and `-127` both map
to `-1.0` and `0` is exact (ES 2.0 used `(2c + 1) / (2^b - 1)`). With `normalized = FALSE` the integer is
converted directly (`255` arrives as `255.0`). `FIXED` is 16.16 two's complement (`c / 65536`) and
`FIXED`/`FLOAT` ignore the flag. In the picture the shader only maps the received value into `[0, 1]`
(`a_value * u_gain + u_bias`); it never repeats the GL's conversion, except in the right-hand column where
that is the point.

## Shader concepts

- `in vec4 a_value` is fed by arrays of 1, 2, 3 or 4 components; the missing components are
  `y = z = 0`, `w = 1` (ES 3.0 §2.9.1) - that is how `w` reaches the framebuffer alpha.
- `layout(location = N)` on a vertex input (GLSL ES 3.00 §4.3.8.1) fixes the slot in the shader text and
  **overrides** `glBindAttribLocation` (ES 3.0 §2.12.5); `named.vert` omits it on purpose.
- An attribute that is declared but never read is not *active*: no location, not enumerated
  (`a_unused`). A scalar still costs a whole slot; `mat2` costs two consecutive slots.
- Two active attributes bound to the same slot are a **link error** in GLSL ES 3.00.

## Validation (`--validate`)

All numbers are computed independently on the CPU; pixel tolerances are ±1 (8-bit quantization).

- **Formats (17 cells):** the spec formulas applied to the stored values, then `* gain + bias`, clamped and
  rounded. This includes `-128, -127, -126`, whose result differs by 32 and 64 levels between the
  ES 3.0 formula and the old ES 2.0 one, so the test cannot pass with the wrong formula.
- **Sizes (4):** `(x, 0, 0, 1)`, `(x, y, 0, 1)`, `(x, y, z, 1)`, `(x, y, z, w)` including alpha.
- **Layouts (4 pixel checks + 4 state checks):** four quad centers and the empty gap per panel; stride,
  type, normalized flag, pointer offset and buffer-binding equality (same / different buffers) per layout.
- **Constants (8 + 8 + 3):** center pixels, `CURRENT_VERTEX_ATTRIB` exactly equal to the expanded vector
  after each call (read inside `render()` right after the call), the enabled-array/restore pair, the initial
  `(0, 0, 0, 1)` of an untouched attribute, and the unchanged current value after drawing from an array.
- **Client arrays (10):** three pixels, then state and errors: pointer returned by `glGetVertexAttribPointerv` is the CPU address,
  buffer binding 0, mixed sources; on a **non-zero VAO** the same call is `INVALID_OPERATION`, with a `NULL`
  pointer it is not (ES 3.0 §2.9, footnote 5).
- **Locations (21):** bound locations reported back; unused / non-existent names give `-1`; linker locations
  distinct and in range and usable for a draw; rebinding a linked program changes nothing until the next
  link; a shader `layout(location)` beats `glBindAttribLocation`; aliasing fails to link and then
  `glGetAttribLocation` is `INVALID_OPERATION`; a reserved `gl_` name and an index `>= MAX_VERTEX_ATTRIBS` are
  `INVALID_OPERATION` / `INVALID_VALUE`.
- **Enumeration (15):** `ACTIVE_ATTRIBUTES == 7`, `ACTIVE_ATTRIBUTE_MAX_LENGTH`, every
  `(name, type, size)` from `glGetActiveAttrib` (FLOAT, FLOAT_VEC2/3/4, FLOAT_MAT2, INT_VEC2,
  UNSIGNED_INT) cross-checked with `glGetAttribLocation`, no overlapping slots (the `mat2` bound to 4
  uses 4 and 5), index `== ACTIVE_ATTRIBUTES` is `INVALID_VALUE` and leaves all outputs untouched.
- **State queries (22):** the initial state of a fresh VAO (Table 6.2), `ENABLED`, `SIZE`, `STRIDE`, `TYPE`,
  `NORMALIZED`, `INTEGER`, `BUFFER_BINDING`, `POINTER` after `glVertexAttribPointer`, independence of two
  VAOs and of VAO 0, `ARRAY_BUFFER_BINDING` being context state, `CURRENT_VERTEX_ATTRIB` via `fv` and `iv`
  being context state, `glIsVertexArray` before/after bind and delete.
- **Index and size errors (6):** every entry point taking an index rejects `MAX_VERTEX_ATTRIBS` with `INVALID_VALUE`; size 5 and
  a negative stride are `INVALID_VALUE`.
- The runtime also requires `glGetError() == GL_NO_ERROR` after init, after the frame and after
  `shutdown()` deleted every object.

## Spec references

ES 3.0 §2.1.5 Fixed-Point Computation · §2.1.6 Fixed-Point Data Conversions (eq. 2.1, 2.2) ·
§2.5 Errors · §2.8 Vertex Specification · §2.9 Vertex Arrays (Table 2.4, footnote 5), §2.9.1 Transferring
Array Elements, §2.9.3 Drawing Commands · §2.10.6 Vertex Arrays in Buffer Objects · §2.11 Vertex Array
Objects · §2.12.5 Vertex Attributes · §6.1.10 Vertex Array Object Queries · §6.1.12 (GetVertexAttrib*,
GetVertexAttribPointerv) · Tables 6.2, 6.3, 6.22, 6.31.
GLSL ES 3.00 §4.3.4 Input Variables · §4.3.8.1 Input Layout Qualifiers.

## Pitfalls

- `normalized` is **not** a type conversion switch for floats: FIXED and FLOAT ignore it. And it is *not
  ignored* for integer types - passing `GL_FALSE` for color bytes gives `255.0`, not `1.0` (white, clamped).
- A client pointer needs VAO 0. In a real program the usual symptom is `INVALID_OPERATION` on the *pointer
  call*, not on the draw call.
- `glBindAttribLocation` after the link does nothing until the program is linked again; and never assume a
  linker-chosen location - ask `glGetAttribLocation` (it returns `-1` for inactive attributes).
- `glGetActiveAttrib` takes an **index** into the active-attribute list, not an attribute location
  (meta-gl observation O5 in `docs/META_GL_GAPS.md`); this demo uses the `ActiveAttribIndex` overload.
- Gaps between stride and component size are fine (the padded `stride` of 4/8/12 bytes here), but an
  offset must be a multiple of the component size.
- **Driver deviation (Mesa 25.0.7, observed on radeonsi and llvmpipe):** ES 3.0 §2.5 says a command that
  generates an error has no effect, yet the failed client-pointer call above still stores the pointer
  in the VAO, and a failing `glGetVertexAttribiv` overwrites its output with 0. The demo prints a note
  instead of asserting these two side effects; the error codes themselves are asserted.
