# es20-indexed-drawing

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es20-indexed-drawing [--headless --validate]`

## Purpose

`glDrawArrays` consumes vertices in order, so every corner of every triangle must be stored. With
`glDrawElements` the vertices are stored once and an **index array** says which of them each triangle uses.
This demo draws one 3 × 3-cell mesh - 16 vertices, 18 triangles, 54 indices - in 16 different ways and
checks every result against a CPU reference. It covers:

- index buffers (`ELEMENT_ARRAY_BUFFER`) with `UNSIGNED_BYTE`, `UNSIGNED_SHORT` and `UNSIGNED_INT` indices
  (the last one is core since ES 3.0) and the `indices` argument as a **byte offset** into the buffer;
- vertex reuse (the same picture from 374 bytes of indexed data or 1080 bytes of expanded vertices);
- `glBufferSubData` on an index buffer and on a vertex buffer, `glBufferData` re-specification with a new size
  and usage, and the "orphaning" idiom;
- `glGetBufferParameteriv` (`BUFFER_SIZE`, `BUFFER_USAGE`) and `glIsBuffer` before and after deletion;
- the fact that the **`ELEMENT_ARRAY_BUFFER` binding belongs to the vertex array object**;
- calls that must fail with a defined error and must change nothing.

## Reading the image

The window is a 4 × 4 grid of 64-pixel tiles (row 0 at the top). Each tile is one `glDrawElements` (or
`glDrawArrays`) call into its own viewport.

| | col 0 | col 1 | col 2 | col 3 |
|---|---|---|---|---|
| **row 0** | `UNSIGNED_BYTE` indices | `UNSIGNED_SHORT` indices | `UNSIGNED_INT` indices 70000..70015 | `UNSIGNED_SHORT`, byte offset 36, count 36: the bottom row of cells is skipped |
| **row 1** | initial `glBufferData` | after `glBufferSubData` on the **indices**: a hole in the center | after `glBufferSubData` on one **vertex**: a white, displaced vertex | after `glBufferData` with twice the size and a new usage: inverted colors |
| **row 2** | orphaning (`glBufferData(NULL)` + `glBufferSubData`): rotated colors | a second VAO with its own index buffer: checkerboard cells only | back to the first VAO, no `glBindBuffer`: the full mesh | client-memory indices on VAO 0 |
| **row 3** | no indices: 54 expanded vertices, `glDrawArrays` | `glDrawElements(count = -1)`: **nothing is drawn** | after a failed `glBufferSubData`: the unmodified mesh | offset 0, count 18: only the bottom row |

Tiles (3, 0) and (3, 3) together make up the full mesh: the same index buffer split into two batches.

## Central calls

| Concern | Calls |
|---|---|
| Index buffer | `glBindBuffer(ELEMENT_ARRAY_BUFFER, ibo)` **while the VAO is bound**, `glBufferData`, `glDrawElements(mode, count, type, offset)` |
| Change data | `glBufferSubData(target, offset, size, data)`, `glBufferData` again (also with `NULL` data) |
| Query | `glGetBufferParameteriv(target, BUFFER_SIZE / BUFFER_USAGE)`, `glIsBuffer`, `glGetIntegerv(ELEMENT_ARRAY_BUFFER_BINDING)` |

`glDrawElements(mode, count, type, indices)` reads `count` indices starting `indices` **bytes** into the
bound index buffer (ES 3.0 §2.10.7); element `i` selects the vertex `indices[i]` of every enabled attribute
array (§2.9.3). With no buffer bound (the initial state of VAO 0) `indices` is instead a pointer to CPU memory.

### Why the VAO matters

The `ELEMENT_ARRAY_BUFFER` binding is part of vertex array object state (ES 3.0 Table 6.2); `ARRAY_BUFFER`
is not (Table 6.3: the buffer is copied into each attribute by `glVertexAttribPointer`). So:

- binding an index buffer while a VAO is bound attaches it to **that** VAO; with VAO 0 bound it configures
  the default VAO instead;
- `glBindVertexArray(vao)` brings back the index buffer, so the draw call needs no `glBindBuffer` - tiles
  (1, 2) and (2, 2) draw with two different index buffers just by switching VAOs;
- `glBindBuffer(ELEMENT_ARRAY_BUFFER, 0)` while a VAO is bound **detaches** its index buffer. Unbind the VAO
  *first* during setup, and never touch `ELEMENT_ARRAY_BUFFER` while a foreign VAO is bound.

### Changing buffers (ES 3.0 §2.10.2, §6.1.9)

`glBufferSubData` overwrites a range in place: `BUFFER_SIZE` and `BUFFER_USAGE` do not change, the range must
lie inside the store (`INVALID_VALUE` otherwise) and a draw issued earlier still sees the old data - the demo
draws before and after each change. `glBufferData` deletes the data store and creates a new one: size and
usage follow the call, the contents are undefined when the pointer is `NULL`, but the buffer **name** - and
therefore every VAO that references it - stays valid. Passing `NULL` and then filling the used part is the
"orphaning" idiom: the GL may give the new data store fresh memory instead of waiting for draws that still
use the old one. A name from `glGenBuffers` is not a buffer object (`glIsBuffer` is `FALSE`) until its first
`glBindBuffer`.

## Shader concepts

`mesh.vert` has two inputs with explicit locations (`layout(location = 0) in vec2`, `= 1 in vec3`) and passes
the color through; the fragment shader writes the interpolated color to a user-declared `out vec4`. Because
every vertex has a different color, the picture reveals *which* vertices the indices selected.

## Validation (`--validate`)

- **Tiles (16 checks):** for every triangle that must be drawn, the pixel containing its centroid is compared
  with the color the rasterizer produces at that pixel's center - barycentric interpolation of the three
  vertex colors (ES 3.0 §3.6.1), tolerance ±3/255 for `mediump` varyings and 8-bit rounding. Every canonical
  triangle that must be **absent** (the skipped row, the hole, the unselected cells, the failed draw) must
  show the clear color at its centroid, and the tile corners must be clear. The checks fail loudly if a probe
  is too close to an edge or nothing was probed.
- **`UNSIGNED_INT`:** the indices are 70000..70015. A decoy copy of the mesh in magenta sits at index
  70000 mod 65536 = 4464, where an implementation that truncated indices to 16 bits would read.
  `MAX_ELEMENT_INDEX >= 2^24 - 1` (Table 6.28) is checked first.
- **Buffer state (17 checks):** `BUFFER_SIZE` / `BUFFER_USAGE` of all seven static buffers; after each step of
  the mutation sequence (initial `glBufferData`, `glBufferSubData` on the indices and on a vertex, 640-byte
  `DYNAMIC_DRAW` re-specification, orphaning to `STREAM_DRAW`), observed inside `render()` right after the
  call; name stability across re-specification; indexed data (374 B) smaller than expanded data (1080 B).
- **`glIsBuffer` / queries (14 checks):** generated-but-unbound `FALSE`, bound `TRUE`, `FALSE` after
  `glDeleteBuffers`, binding reset to 0 by the deletion, initial size 0 / `STATIC_DRAW`, all nine
  `BUFFER_USAGE` values round-trip, size 0 allowed, and the errors below.
- **VAO bindings (12 checks):** `ELEMENT_ARRAY_BUFFER_BINDING` queried after binding each of six VAOs; VAO 0 has
  its own binding; unbinding with a VAO bound affects only that VAO; `ARRAY_BUFFER_BINDING` survives every
  VAO switch.
- **Errors:** `glDrawElements(count = -1)`, `glBufferSubData` past the end, `glBufferSubData` with a negative
  offset, `glBufferData` with a negative size are `INVALID_VALUE` (ES 3.0 §2.5, §2.10.2); querying or writing
  an `ELEMENT_ARRAY_BUFFER` / `ARRAY_BUFFER` target with no buffer bound is `INVALID_OPERATION`
  (§2.10.1). The failed draw leaves its tile blank and the failed `glBufferSubData` leaves the mesh intact
  ("the command is ignored", §2.5).
- The runtime additionally requires `glGetError() == GL_NO_ERROR` after setup, after the frame (the negative
  calls' errors are read inside `render()` and asserted in `validate()`) and after `shutdown()`.

## Spec references

ES 3.0 §2.5 Errors · §2.9 Vertex Arrays, §2.9.3 Drawing Commands · §2.10.1 Creating and Binding Buffer
Objects · §2.10.2 Creating Buffer Object Data Stores · §2.10.4 Effects of Accessing Outside Buffer Bounds ·
§2.10.6 Vertex Arrays in Buffer Objects · §2.10.7 Array Indices in Buffer Objects · §2.10.8 Buffer Object
State · §2.11 Vertex Array Objects · §3.6.1 Basic Polygon Rasterization · §6.1.9 Buffer Object Queries ·
Tables 2.7, 2.8, 6.2, 6.3, 6.4, 6.28.
GLSL ES 3.00 §4.3.4 Input Variables · §4.3.8.1 Input Layout Qualifiers.

## Pitfalls

- `indices` is a **byte** offset. For 16-bit indices, element `n` is at byte `2 * n`.
- Indices past the end of the vertex data, or an index buffer read past its end, are undefined behaviour
  without robust buffer access (§2.10.4); the demo never does this. With `PRIMITIVE_RESTART_FIXED_INDEX`
  enabled (disabled by default) the largest value of the index type, e.g. 255 or 65535, is not a vertex.
- 8-bit indices are legal but limit a mesh to 256 vertices; use `UNSIGNED_SHORT` below 65536 vertices and
  `UNSIGNED_INT` beyond (the implementation limit is `MAX_ELEMENT_INDEX`).
- Client-memory indices (a CPU pointer with no index buffer) are meant for the default VAO; prefer buffers.
- `glBufferSubData` into data a pending draw still reads may make the driver wait or copy; orphaning with
  `glBufferData(NULL)` avoids that. Neither is a substitute for a defined update order - GL guarantees that
  a draw sees the data as it was **when the draw was issued**, which the demo verifies.
