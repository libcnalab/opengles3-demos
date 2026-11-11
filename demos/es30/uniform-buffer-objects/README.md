# es30-uniform-buffer-objects

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es30-uniform-buffer-objects [--headless --validate]`

## Purpose

A plain `uniform` lives in the program and is set one variable at a time with `glUniform*`. A **uniform block** is a
named group of uniforms whose values come from a **buffer object**: one `glBufferSubData` (or a mapping) updates
dozens of values, the same buffer can feed several programs, and the values survive a relink. The price is that the
application and the shader must agree on **where every member lives inside the buffer**. This demo shows all of it:

- the **std140** layout rules (ES 3.0 §2.12.6.4): scalars, `vec3` padding and packing, arrays, `mat2`/`mat3`/`mat4`
  and non-square matrices, `row_major`, structures and arrays of structures. A CPU layout calculator predicts every
  offset and stride, the GL is asked for the same numbers (`glGetUniformIndices`, `glGetActiveUniformsiv`), and the
  buffer is packed with the calculator's numbers and read back by the shader;
- a **packed** block, whose layout is the implementation's choice: the application must query it and pack with the
  queried numbers;
- **introspection** of blocks (`glGetUniformBlockIndex`, `glGetActiveUniformBlockiv`, `glGetActiveUniformBlockName`);
- the two-step **binding model**: block -> binding point (`glUniformBlockBinding`) and binding point -> buffer
  (`glBindBufferBase`, `glBindBufferRange` with `UNIFORM_BUFFER_OFFSET_ALIGNMENT`), the indexed state queries, one buffer
  serving two programs, and updates between draws (`glBufferSubData`, `glMapBufferRange`, `glCopyBufferSubData`,
  `glBufferData`);
- an **array of blocks**, the implementation **limits**, the defined **errors**, and a deliberate **layout mistake**.

## Reading the image

The 256 x 256 surface has two parts.

**Top, 64 pixels - the verification grid.** 16 columns x 4 rows of 16-pixel cells in reading order. The fragment
shader reads one member of a block per cell and compares it with the value the application packed:
**green** = equal, **red** = different, **magenta** = different in a panel that deliberately uses a wrong layout.
Dark cells test nothing. Magenta is the expected outcome in those panels; red anywhere would be a failure.

| Cells | What they test |
|---|---|
| 0-29 | block `Probe` (std140), 30 members in declaration order: `f`, `v3`, `after_v3`, `v2`, `v3b`, `i`, `iv2`, `u`, `flag_true`, `flag_false`, `v4`, `farr[3]`, `v2arr[2]`, `v3arr[2]`, `m2`, `m3`, `m4`, `m3x2`, `m3r` (row_major mat3), `m2x4r` (row_major mat2x4), `m3arr[2]`, `inner.a`, `inner.b`, `after_inner`, `lights[0].position/intensity/color`, `lights[1].position/intensity/color` - all green |
| 32-36 | block `Mistake` packed with the **correct** std140 layout (`vec3 a; float s; vec3 b; vec3 c; float t;`) - all green |
| 38-42 | the same block packed **like a C struct** (every member at the next multiple of its own size): `a`, `s`, `b` still agree (green), `c` and `t` do not (**magenta**) |
| 48-52 | the same block with every `vec3` **padded to 16 bytes**, so a float after a vec3 starts at +16: `a` agrees, `s`, `b`, `c`, `t` do not (**magenta**) |
| 54-59 | block `Queried` (`layout(packed)`): six members packed at the offsets and strides the GL reported - all green |

**Below - the scene.** 32-pixel cells, 8 per row, each drawn with the block `Palette` (`vec4 color[3]; vec2 offset;
float gain; int pick;`, 64 bytes). Program 1 draws a small square at `offset` with color `color[pick] * gain`;
program 2 reads the same block and draws the whole cell in the *inverse* color. Where the data comes from differs:

| Row | Cells |
|---|---|
| A | **A0** default binding 0 · **A1** `glUniformBlockBinding` moves the block to point 3 · **A2** another buffer at the same point, program untouched · **A3** program 2 reads the same buffer (big inverse square) · **A4/A5** `glBufferSubData` between draws, both programs follow · **A6** the same through `glMapBufferRange` (only offset, gain, pick change) · **A7** one buffer at two binding points, both programs drawn into one cell |
| B | **B0-B3** `glBindBufferRange` at four aligned offsets of one buffer · **B4** rejected ranges leave the binding intact (same picture as B3) · **B5** `glBindBufferBase` binds the whole buffer again (slot 0) · **B6** a slot is rewritten, then bound as a range · **B7** program 2 on that range |
| C | **C0** an **array of blocks** `Tint tints[2]` backed by two buffers (left half / right half) · **C1** ... backed by two ranges of one buffer · **C2** `glBufferData` fills the whole block · **C3/C4** `pick` and `gain` are updated alone, at the offsets `glGetActiveUniformsiv` reported · **C5** `glCopyBufferSubData` replaces the whole block with a slot of the ring buffer · **C6** `glBufferData` again ("orphaning") |

The lower part of the surface is intentionally empty.

## The binding model

```
program 1:  block "Palette"  --glUniformBlockBinding--> binding point 3  --glBindBufferBase/Range--> buffer (+ offset, size)
program 2:  block "Palette"  --glUniformBlockBinding--> binding point 3  ----------------------------^   (same buffer)
```

The binding **points** (an indexed array of `MAX_UNIFORM_BUFFER_BINDINGS` >= 24 slots, the `UNIFORM_BUFFER` indexed
target) are context state, independent of any program. A program only says "my block reads point N" and the context
says "point N is this range of this buffer". Consequences the demo exercises: after linking, every block is bound
to point 0 (`UNIFORM_BLOCK_BINDING`); changing the buffer at a point changes what every block bound there sees; a
second program on the same point shares the data; `glBindBufferBase` also resets `UNIFORM_BUFFER_START` and
`UNIFORM_BUFFER_SIZE` to 0 ("not specified"), whereas `glBindBufferRange` stores them (`glGetInteger64i_v`, or
`glGetIntegeri_v`: ES 3.0 §6.1.2 converts).

`glBindBufferRange(target, index, buffer, offset, size)` needs `offset` to be a multiple of
`UNIFORM_BUFFER_OFFSET_ALIGNMENT` (16 on the Mesa drivers used here, at most 256 by spec) and `size > 0`; otherwise
INVALID_VALUE. The ring buffer in row B therefore places its four palettes at multiples of
`align_up(64, alignment)`. A block must be backed by at least `UNIFORM_BLOCK_DATA_SIZE` bytes.

## std140 in one table

Rules (ES 3.0 §2.12.6.4): scalars align to their size; `vec2` to 8; `vec3` and `vec4` to **16**; the base offset of
the next member is the end of the *consumed* bytes, so a `vec3` occupies 12 and a float may follow at +12; an
**array** of scalars or vectors has a stride of 16 whatever the element; a matrix is an array of column vectors (row
vectors if `row_major`), each rounded up to 16 - **matrix stride 16**, even for `mat2`; a structure is aligned to 16
and its size rounded up to 16; an array of structures has the structure's size as stride.

| member of `Probe` | offset | why |
|---|---|---|
| `float f` | 0 | |
| `vec3 v3` | 16 | alignment 16, occupies 12 bytes |
| `float after_v3` | **28** | packs into the vec3's last slot |
| `vec2 v2` | 32 | |
| `vec3 v3b` | **48** | aligned up from 40 |
| `int i` | **60** | packs after the vec3 |
| `ivec2`, `uint`, two `bool` | 64, 72, 76, 80 | a `bool` is one `uint`; any non-zero value is true (the demo stores 0xDEADBEEF) |
| `vec4 v4` | 96 | |
| `float farr[3]` | 112, stride **16** | |
| `vec2 v2arr[2]`, `vec3 v3arr[2]` | 160, 192, stride 16 | |
| `mat2`, `mat3`, `mat4` | 224, 256, 304 | matrix stride 16 |
| `mat3x2` (3 columns of vec2) | 368 | 3 x 16 bytes |
| `row_major mat3`, `row_major mat2x4` | 416, 464 | 3 rows of vec3, 4 rows of vec2, each 16 bytes; `UNIFORM_IS_ROW_MAJOR` = 1 |
| `mat3 m3arr[2]` | 528, array stride **48** | |
| `Inner inner { float a; vec3 b; }` | 624 (`a`), 640 (`b`) | the structure is padded to 656 |
| `float after_inner` | **656** | right after the padded structure |
| `Light lights[2]` | 672, 704 | structure size 32, `intensity` packs at +12 |
| total | **736** | `UNIFORM_BLOCK_DATA_SIZE` |

Neither the shader nor the GL is trusted with these numbers alone: the CPU calculator in `main.cpp`
(~60 lines, rules 1-10 spelled out) computes them, `glGetActiveUniformsiv` must return the same, and the shader must
read the values the calculator's offsets put into the buffer. Only if both ends agree is the layout right.

### The layout mistake

`Mistake { vec3 a; float s; vec3 b; vec3 c; float t; }` is packed three ways. **std140** (the truth): `a` 0, `s` 12,
`b` 16, `c` 32, `t` 44, size 48. **A C struct with tightly packed floats**: 0, 12, 16, **28**, **40** - the first
three coincide with std140 (the float after a vec3 does pack into the vec3's slot, which is why the mistake hides
easily), but the second consecutive `vec3` is read from the wrong place. **Every vec3 padded to a vec4**: 0, **16**,
32, 48, 64 - everything after the first vec3 is shifted. The shader sees wrong values in exactly the members the
offsets differ for (cells 41, 42 and 49-52, magenta), and the offset comparison fails for them as well (negative
control, below).

### The packed block

`layout(packed) uniform Queried { float q_scalar; vec3 q_vec3; float q_after; mat3 q_mat3; vec2 q_arr[2];
layout(row_major) mat2x3 q_row; }` has no portable layout. The demo queries each member's offset, array stride, matrix
stride and `UNIFORM_IS_ROW_MAJOR`, packs with exactly those numbers and lets the shader verify. The Mesa drivers here
place `q_arr` at stride **8** and the row-major `mat2x3` with row stride **8** - not std140's 16 - and the cells are
still green, which is the point (the run prints a `[note]` line naming the differences).

## Introspection

`glGetUniformIndices(program, count, names, indices)` maps names to indices. Members of a block are named with
the **block** name (not the instance name): `"Probe.f"`, `"Probe.farr[0]"` (an array is one entry, `[0]`), `"Probe.lights[1].color"`
(each element of an array of structures is separate); a block without an instance name has bare member names
(`"color[0]"`, `"m_s"`); an array of blocks lists the member once (`"Tint.color"`, owned by the first block).
`glGetActiveUniformsiv(program, count, indices, pname, params)` returns `UNIFORM_TYPE`, `UNIFORM_SIZE` (array length),
`UNIFORM_BLOCK_INDEX`, `UNIFORM_OFFSET`, `UNIFORM_ARRAY_STRIDE` (0 for non-arrays), `UNIFORM_MATRIX_STRIDE` (0 for
non-matrices) and `UNIFORM_IS_ROW_MAJOR`. Members of named blocks have **no location**: `glGetUniformLocation` returns -1.
`glGetActiveUniformBlockiv` returns `UNIFORM_BLOCK_DATA_SIZE`, `_BINDING`, `_ACTIVE_UNIFORMS`, `_ACTIVE_UNIFORM_INDICES`,
`_NAME_LENGTH` and whether the vertex or fragment shader references the block: program 1's `Palette` is referenced by
both, program 2's and `Probe` only by the fragment shader (their vertex shaders declare no block), and the demo
checks all three. `glGetActiveUniformBlockName` honours `bufSize` and returns the length without the terminator.

## Limits and errors

`MAX_UNIFORM_BLOCK_SIZE` (>= 16384, `glGetInteger64v`), `MAX_VERTEX_UNIFORM_BLOCKS` / `MAX_FRAGMENT_UNIFORM_BLOCKS` (>= 12),
`MAX_COMBINED_UNIFORM_BLOCKS` (>= 24), `MAX_UNIFORM_BUFFER_BINDINGS` (>= 24), `UNIFORM_BUFFER_OFFSET_ALIGNMENT` (<= 256)
and the combined-component footnote of Table 6.33 are checked. "If the number of uniform blocks used by any shader in
the program exceeds its corresponding limit, the program will fail to link": a fragment shader that reads exactly
`MAX_FRAGMENT_UNIFORM_BLOCKS` blocks links, one with a block more does not.

Errors: invalid block index to `glGetActiveUniformBlockiv`, `glGetActiveUniformBlockName`, `glUniformBlockBinding`;
binding point >= `MAX_UNIFORM_BUFFER_BINDINGS` for `glUniformBlockBinding`, `glBindBufferBase`, `glBindBufferRange`,
`glGetIntegeri_v`, `glGetInteger64i_v`; uniform index >= `ACTIVE_UNIFORMS`; a misaligned range offset, a zero size and a
negative offset - all INVALID_VALUE and nothing written or rebound. Unknown names give `INVALID_INDEX`, **not** an error.

## Validation (`--validate`)

113 demo checks per run (plus 5 runtime checks):

- **Pixels, verification grid (6 checks):** every cell of the five groups must have the expected color - tolerance 0
  for exact green / red / magenta, 1/255 for the neutral gap cells; the shader compared exact values, so a green cell is
  an exact match of every component of every member.
- **Pixels, scene (23 checks):** per cell the probes (square centers, points inside, the cell corners the square does not
  reach, the corners of program 2's square) against the colors the CPU computes from the palette (`round(c * gain * 255)`,
  tolerance 1/255 for the highp float -> 8-bit conversion).
- **Introspection (about 35 checks):** the seven `glGetActiveUniformsiv` properties of all 30 `Probe` members against the
  CPU calculator, the block's data size / active uniforms / indices / referenced-by flags / names, hand-derived offsets
  that do not come from the calculator (16, 28, 48, 60, 640, 656, 716, 672), program 1/2/3 blocks, the array of blocks.
- **State (about 20 checks):** `UNIFORM_BLOCK_BINDING` after link and after `glUniformBlockBinding`; `UNIFORM_BUFFER_BINDING`,
  `_START`, `_SIZE` after `glBindBufferBase` / `glBindBufferRange` through both `glGetIntegeri_v` and `glGetInteger64i_v`.
- **Errors (about 25 checks):** the cases above with the exact error code, plus "nothing written" where the spec says so.
- **Limits (9 checks)** including the two link experiments.
- **Negative controls:** (1) a wrong `v3b.y` in the buffer must turn verification cell 4 red while its neighbour stays
  green, and restoring it must turn it green again; (2) the offset comparison must report 2 mismatches for the C layout
  and 4 for the padded layout of `Mistake`; (3) the magenta panels themselves, which *must* disagree with the shader.

### Known driver deviation

`glBindBufferRange(GL_UNIFORM_BUFFER, i, buffer, offset, size)` with a **negative** `offset` that is a multiple of the
alignment is INVALID_VALUE in ES 3.0 §2.10.1.1, but Mesa's llvmpipe, radeonsi and zink raise no error and store the
invalid binding; **drawing with that binding crashes** (segmentation fault on llvmpipe, GPU context loss on radeonsi). The
check stays strict and is labelled XFAIL on those drivers; because the call may succeed, the demo re-binds a valid range
immediately and never draws with the rejected one.

## Spec references

ES 3.0 §2.10.1.1 Binding Buffer Objects to Indexed Targets · §2.12.3 Program Objects (names of active resources) ·
§2.12.6 Uniform Variables, §2.12.6.2 Uniform Blocks, §2.12.6.3 Uniform Buffer Object Storage, §2.12.6.4 Standard
Uniform Block Layout, §2.12.6.5 Uniform Buffer Object Bindings · §6.1.1 Simple Queries (indexed), §6.1.2 Data
Conversions, §6.1.9 Buffer Object Queries · Tables 6.21, 6.25, 6.31 - 6.33. GLSL ES 3.00 §4.3.7 Interface Blocks,
§4.3.8.3 Uniform Block Layout Qualifiers.

## Pitfalls

- **Never assume a layout for `shared` (the default!) or `packed` blocks** - declare `layout(std140)` or query. The
  default of a block without a qualifier is `shared`, which is consistent across programs but not predictable.
- A `vec3` has alignment 16 but size 12; a `float` may follow it. Two `vec3`s in a row are 16 bytes apart. Arrays
  of `float` and `vec3` have stride 16 - a `float[4]` takes 64 bytes. `mat3` is three 16-byte columns.
- `row_major` changes the **memory layout only**; the shader still indexes `m[column][row]`. Upload the transposed
  data, or declare the matrix `row_major` and upload the matrix as written.
- Offsets and strides are invalidated by every relink; `UNIFORM_BLOCK_BINDING` is reset to 0 by it.
- The offset of `glBindBufferRange` must be a multiple of `UNIFORM_BUFFER_OFFSET_ALIGNMENT` - 256 is the worst case
  the spec allows, so pad ring-buffer slots accordingly.
- Reading a block from a buffer smaller than `UNIFORM_BLOCK_DATA_SIZE` is undefined and may crash the GPU context.
- `glBindBufferBase` and `glBindBufferRange` also change the **general** `UNIFORM_BUFFER` binding that `glBufferSubData`
  and `glMapBufferRange` use: after binding point 9 the next `glBufferSubData(GL_UNIFORM_BUFFER, ...)` writes into *that* buffer.
- Instance names matter for the shader only; the GL uses the block name in every query.
