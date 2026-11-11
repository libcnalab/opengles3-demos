# es30-vertex-array-objects

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es30-vertex-array-objects [--headless --validate]`

## Purpose

A vertex array object (VAO) is the container for *everything the vertex processor needs to know about its
inputs*: which attribute arrays are enabled, where each one lives (buffer and byte offset), how it is laid out
(size, type, stride, normalized flag, integer flag, divisor) and which buffer supplies the indices of
`glDrawElements`. Binding a VAO swaps all of that at once. ES 3.0 made VAOs part of the core API (in ES 2.0
they were an extension) and also added the vertex formats this demo uses: *pure integer* attributes that reach an
`ivec4`/`uvec4` input unconverted, the packed `2_10_10_10` formats and `HALF_FLOAT`.

The demo does four things, each checked against numbers computed on the CPU:

1. **Switching.** Five VAOs with different buffers, formats, strides and element buffers are drawn in two different
   orders. If a VAO failed to carry some state, or state leaked from the previous VAO, a picture would change.
2. **What is and is not VAO state.** `ELEMENT_ARRAY_BUFFER` is part of the VAO; `ARRAY_BUFFER` is not (it only
   matters at the moment `glVertexAttribPointer` copies it into the attribute). The VAO name lifecycle
   (`glIsVertexArray`, deleting the bound VAO) is exercised too.
3. **Integer and packed attributes.** Values above 2^24, which a 32-bit float cannot hold, prove that integers
   really stay integers; the packed and half-float formats are decoded with the equations of the specification.
4. **Errors.** The rules for client-memory pointers on a non-default VAO, for packed types with the wrong size and
   for `glVertexAttribIPointer` with a non-integer type.

## What the image shows

256 × 256 default framebuffer, a 4 × 4 grid of 64 × 64 cells. Row 3 is the top row.

| Row | Cells (left to right) | What it shows |
|---|---|---|
| 3 | A, B, C, D | **VAO A:** positions and colors in two tightly packed float buffers (color is a `vec3`, so alpha defaults to 1). **VAO B:** one interleaved buffer, 8-byte stride: normalized `SHORT` positions and normalized `UNSIGNED_BYTE` colors. **VAOs C and D:** the *same* vertex buffer and attribute layout (12-byte stride); C has a `USHORT` element buffer (bottom and top wedge), D a `UBYTE` element buffer read from byte offset 2 (right and left wedge). |
| 2 | D, C, B, A | the same four VAOs again, in the opposite order |
| 1 | I1, I2, I3, I4 | integer inputs. Each cell is split into 4 columns (x, y, z, w) and 2 rows (upper: `ivec4`, lower: `uvec4`); a sub-area is green only if the shader received the exact integer the CPU expects. **I1:** `glVertexAttribIPointer` `INT` + `UNSIGNED_INT` with values above 2^24. **I2:** `SHORT` size 3 (sign-extended, `w` = 1) + `UNSIGNED_BYTE` size 2 (zero-extended, `z` = 0, `w` = 1). **I3:** arrays disabled, `glVertexAttribI4i` / `glVertexAttribI4ui`. **I4:** arrays disabled, `glVertexAttribI4iv` / `glVertexAttribI4uiv`. |
| 0 | UINT, INT, HALF, E | **UNSIGNED_INT_2_10_10_10_REV** (left half normalized, right half *not* normalized and divided by the shader; both halves show the same two colors) and **INT_2_10_10_10_REV** (the same pair, but the 2-bit `w` of the lower word is -2: the normalized half clamps it to -1, the raw half keeps -2, so the two halves differ in alpha). **HALF_FLOAT** positions and colors (four quads). **VAO E:** the color array is *disabled*; the orange comes from `glVertexAttrib4f`. |

Colors with alpha below 1 are stored in the framebuffer's alpha channel, which the PNG capture keeps; image viewers
composite such pixels over their own background, so the packed/half quads with `w` < 3 look lighter or translucent.

## Central calls

| Call | Role |
|---|---|
| `glGenVertexArrays`, `glBindVertexArray`, `glDeleteVertexArrays`, `glIsVertexArray` | VAO names and lifecycle |
| `glEnableVertexAttribArray`, `glVertexAttribPointer` | enable an array and describe it (float or float-converted data); records the current `ARRAY_BUFFER` |
| `glVertexAttribIPointer` | same for pure integer data (`BYTE`, `UBYTE`, `SHORT`, `USHORT`, `INT`, `UINT` only) |
| `glVertexAttrib4f[v]`, `glVertexAttribI4i[v]`, `glVertexAttribI4ui[v]` | *current* generic attribute values, used when the array is disabled |
| `glGetVertexAttribiv/fv/Iiv/Iuiv`, `glGetVertexAttribPointerv` | read the per-attribute state back |
| `glBindBuffer(ELEMENT_ARRAY_BUFFER)` | attaches an index buffer to the bound VAO |
| `glDrawArrays`, `glDrawElements` | draw; all vertex state comes from the bound VAO |

## Spec

- ES 3.0 §2.11 Vertex Array Objects; Table 6.2 (state stored in a VAO) and Table 6.3 (state that is *not*).
- ES 3.0 §2.9 Vertex Arrays and Table 2.4 (which command accepts which sizes and types); §2.9.1 (missing
  components are `y = z = 0`, `w = 1`); §2.9.2 Packed Vertex Data Formats; §2.9.3 (a disabled array reads the current value).
- ES 3.0 §2.8 Vertex Specification (current values, `VertexAttribI4*`); §2.10.6 and §2.10.7 (buffers for vertices and indices).
- ES 3.0 §2.1.6.1 equations 2.1 and 2.2 (normalized fixed-point to float); §2.5 (an erroring command is ignored).
- ES 3.0 §6.1.10 `IsVertexArray`; §6.1.12 `GetVertexAttrib*` and `GetVertexAttribPointerv`.
- GLSL ES 3.00 §4.3.4 (vertex inputs), §4.3.6 (outputs; integer outputs must be `flat`), §4.3.8.1 (`layout(location)`).

The rules the demo relies on, in the spec's words, shortened:

- *VAO state.* "All state related to the definition of data used by the vertex processor is encapsulated in a vertex
  array object." `ELEMENT_ARRAY_BUFFER_BINDING` is in Table 6.2, `ARRAY_BUFFER_BINDING` in Table 6.3. A newly bound
  VAO "is a new state vector ... with the same initial values listed in table 6.2" - not a copy of the previous one.
- *Deleting.* "If a vertex array object that is currently bound is deleted, the binding for that object reverts to
  zero and the default vertex array becomes current." `BindVertexArray` with an unused name is `INVALID_OPERATION`.
- *Signed normalized.* Equation 2.2 is `f = max(c / (2^(b-1) - 1), -1.0)`: for a 10-bit field `-512` and `-511`
  both give `-1.0`; for the 2-bit `w` the values `-2, -1, 0, 1` give `-1, -1, 0, 1`. (Older desktop GL versions used
  `(2c + 1) / (2^b - 1)`; the demo's mutation check shows the two differ visibly in `w`.)
- *Errors.* `INVALID_OPERATION` if the type is `INT_2_10_10_10_REV` / `UNSIGNED_INT_2_10_10_10_REV` and `size` is not 4,
  or if a non-zero VAO is bound, `ARRAY_BUFFER` is 0 and `pointer` is not NULL (a VAO cannot hold client arrays).
  `INVALID_VALUE` for a size outside Table 2.4, a negative stride or an index of `MAX_VERTEX_ATTRIBS` and above.

## Shader concepts

`attribs.vert` has `layout(location = 0) in vec2 a_pos` and `layout(location = 1) in vec4 a_attr`. Whatever the GL
delivers after converting the stored format arrives in `a_attr`; `u_scale` and `u_bias` only map that value to a
displayable color (`color = a_attr * u_scale + u_bias`). When the attribute-1 *array* is disabled, `a_attr` is the
current generic attribute value, read exactly as array data would be.

`integer.vert` declares `ivec4 a_i` and `uvec4 a_u`. They must be fed by `glVertexAttribIPointer` or `glVertexAttribI4*`
(a float source leaves them undefined, §2.9). The values leave the vertex shader through `flat` outputs: integer
varyings cannot be interpolated, and GLSL ES 3.00 requires `flat`. `integer.frag` compares each component with a
uniform holding the expected value, so the comparison is exact integer equality on the GPU.

## How it is validated

- **CPU model.** Each float cell is a list of Gouraud-shaded triangles built from the *same* source arrays that were
  uploaded, with the spec's equations applied on the CPU (normalized `UNSIGNED_BYTE` is `c/255`, normalized `SHORT` is
  `max(c/32767, -1)`, packed words are decoded bit by bit from the 32-bit value, half floats are decoded from their
  IEEE binary16 bits by a CPU routine that is itself checked against known bit patterns). Every pixel of a cell is read back and
  compared with the model, except pixels within 1 px of a triangle edge (the exact edge pixels depend on the rasterizer's
  tie-breaking rules, so they are skipped rather than guessed). The number of compared pixels is part of the check
  message and a minimum is required, so an empty model cannot pass.
- **Tolerances.** Flat-color cells (packed, half float, VAO E) allow ±1: the expected 8-bit value is a rounded float
  and a GPU may round a value like 127.5 either way. Gradient cells (A, B, C, D) allow ±3, like `es30-hello-triangle`:
  the vertex colors are exact, and interpolation uses the rasterizer's finite-precision barycentrics. The largest
  deviation seen on radeonsi, llvmpipe and zink is 1/255 and is printed with each cell.
- **Integer cells.** The shader decides; the CPU reads 8 pixels per cell and requires the exact "match" color of every
  sub-area (green shade per component), which means the shader saw exactly the expected `int`/`uint`.
- **Packed words.** Each quad's center pixel is also checked individually against the decoded color, for both the
  normalized and the raw variant, so a wrong conversion names itself in the output.
- **State.** `glGetVertexAttribiv` returns enabled/size/type/normalized/stride/integer/divisor/buffer of each attribute of
  each VAO and `glGetVertexAttribPointerv` the offset; the bindings are queried after every VAO switch. A fresh VAO must
  read the initial values of Table 6.2 even though a configured VAO was bound just before.
- **Lifecycle and errors.** Every `INVALID_*` case listed above is provoked on a scratch VAO and must produce exactly the
  error the spec names; cases that must *not* error (NULL pointer without a buffer, a client pointer on the default VAO,
  packed type with size 4) are checked as well.
- **Negative controls** (a check that cannot fail proves nothing):
  1. the shader is told to expect `16777216` (what a float conversion of `16777217` would give): the sub-area turns red
     while its neighbour stays green;
  2. VAO B is drawn into A's cell: the VAO-A model must now mismatch (> 100 pixels) and the "A then B" model must match exactly;
  3. the pre-ES-3.0 signed-normalized formula is applied to the INT_2_10_10_10_REV model and must *not* match the pixels.

## Run

```
./build/debug/bin/es30-vertex-array-objects --headless --validate --capture vao.png
```

## Pitfalls

- `ARRAY_BUFFER` is captured by `glVertexAttribPointer`, not by `glBindVertexArray`. Unbinding it afterwards (the demo draws
  with `ARRAY_BUFFER = 0`) is safe; forgetting to bind the right buffer *before* the pointer call is the classic bug.
- `glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)` while a VAO is bound *detaches the index buffer from that VAO*. Binding an
  element buffer while no VAO is bound edits the default VAO, which is not what you want in ES 3.0 code.
- `ivec4` / `uvec4` inputs need `glVertexAttribIPointer`. `glVertexAttribPointer` with `GL_INT` converts to float (and loses
  everything above 2^24); the shader would read undefined values.
- Missing components: a size-3 *integer* array gives `w = 1`, a size-2 gives `z = 0, w = 1` (§2.9.1).
- A signed 2-bit `w` only has the values -2..1; normalized, both -2 and -1 are -1.0 (eq. 2.2).
- **Driver deviation observed (not asserted).** On Mesa 25.0.7 (radeonsi, llvmpipe, zink) a rejected
  `glVertexAttribPointer` (non-default VAO, no `ARRAY_BUFFER`, non-NULL pointer) *does* raise `INVALID_OPERATION` but still stores the
  pointer in the VAO (`glGetVertexAttribPointerv` returns it afterwards). ES 3.0 §2.5 says the offending command "is ignored so
  that it has no effect on GL state". The demo checks the error code (which is correct) and does not assert the state
  afterwards; see the report.
