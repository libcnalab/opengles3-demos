# es30-buffer-mapping

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es30-buffer-mapping [--headless --validate]`

## Purpose

`glBufferData` and `glBufferSubData` copy client memory into a buffer object. `glMapBufferRange` goes the other
way round: the GL hands the application a **pointer** into the buffer's data store, the application reads or
writes through it, and `glUnmapBuffer` gives the store back. Used well it avoids a copy and lets a program stream
data into a buffer that earlier draws are still reading. The price is a set of flags with precise rules about what
the application may assume afterwards. This demo exercises every one of them:

- `MAP_READ_BIT`, `MAP_WRITE_BIT` and the four modifiers `MAP_INVALIDATE_RANGE_BIT`, `MAP_INVALIDATE_BUFFER_BIT`,
  `MAP_FLUSH_EXPLICIT_BIT` and `MAP_UNSYNCHRONIZED_BIT` - alone, combined, and (for the combinations the spec
  forbids) rejected with the defined error;
- `glFlushMappedBufferRange` with sub-ranges that are **relative to the start of the mapping**;
- the mapped-state queries `BUFFER_MAPPED`, `BUFFER_ACCESS_FLAGS`, `BUFFER_MAP_OFFSET`, `BUFFER_MAP_LENGTH`,
  `BUFFER_SIZE` (32- and 64-bit) and `glGetBufferPointerv(BUFFER_MAP_POINTER)`;
- `glCopyBufferSubData` through `COPY_READ_BUFFER` / `COPY_WRITE_BUFFER`, between two buffers and inside one;
- the calls that interact with a mapped buffer: `glBufferData` (unmaps it), `glBufferSubData` and
  `glCopyBufferSubData` (rejected), `glDeleteBuffers` (unmaps it).

Every result is observed twice: **byte for byte** by mapping the buffer again with `MAP_READ_BIT`, and **visually**
by drawing from it.

## Reading the image

The 256 x 256 surface is a 4 x 4 grid of 64-pixel cells (reading order, row 0 on top). Every cell draws the same
kind of content - four colored quads in a 2 x 2 arrangement, six 12-byte vertices (`vec2` float position + four
normalized unsigned bytes of color) per quad, 288 bytes in all - but the bytes get into the buffer in a different
way. Bottom-left of a cell is quad 0, bottom-right quad 1, top-left quad 2, top-right quad 3.

| | col 0 | col 1 | col 2 | col 3 |
|---|---|---|---|---|
| **row 0** | **0** `MAP_WRITE_BIT` into a fresh `glBufferData(NULL)` store | **1** `WRITE \| INVALIDATE_BUFFER` over the whole store (the "orphan and refill" idiom) | **2** `READ \| WRITE`: the old bytes are read through the pointer and every color is inverted in place | **3** `WRITE \| INVALIDATE_RANGE` on quads 1-2: quads 0 and 3 keep their old (dark) colors |
| **row 1** | **4** `WRITE \| FLUSH_EXPLICIT`, everything flushed by three `glFlushMappedBufferRange` calls | **5** mapping starts at byte 72; only mapping ranges `[0,72)` and `[144,216)` are flushed. **Quad 2 was written but not flushed, so it is not drawn** (empty top-left) | **6** `WRITE \| INVALIDATE_RANGE \| UNSYNCHRONIZED`: four quads written one after the other while the draw of the previous quad is still pending | **7** `glCopyBufferSubData` from a second buffer: the two top quads are copies |
| **row 2** | **8** a copy inside **one** buffer (adjacent ranges): the top row repeats the bottom row | **9** a long series of rejected `glMapBufferRange` / `glFlushMappedBufferRange` / `glUnmapBuffer` calls: the data is untouched | **10** `glBufferData` on a **mapped** buffer unmaps it; the new data is drawn | **11** one buffer bound to two targets: mapped through `COPY_WRITE_BUFFER`, queried and unmapped through `ARRAY_BUFFER` |
| **row 3** | **12** `WRITE \| INVALIDATE_BUFFER` on quads 1-2: quads 0 and 3 become undefined and are **not drawn** | **13** a series of rejected `glCopyBufferSubData` calls: the destination keeps its data | **14** `glBufferSubData` over a mapped range is rejected; after unmapping it works | **15** `READ \| WRITE` at buffer offset 144: the two top quads are read and their colors swapped in place |

The thin lighter lines are the 2-pixel cell borders; the dark blue is the cell background. Cell 5, cell 12 and
the "not drawn" slots are empty on purpose - see "What is not asserted".

## Central calls

| Concern | Calls |
|---|---|
| Map | `glMapBufferRange(target, offset, length, access)` -> `void*` (NULL on error) |
| Write visibility | `glFlushMappedBufferRange(target, offset, length)` (needs `MAP_FLUSH_EXPLICIT_BIT`), `glUnmapBuffer(target)` -> `GLboolean` |
| State | `glGetBufferParameteriv` / `glGetBufferParameteri64v` (`BUFFER_MAPPED`, `BUFFER_ACCESS_FLAGS`, `BUFFER_MAP_OFFSET`, `BUFFER_MAP_LENGTH`, `BUFFER_SIZE`), `glGetBufferPointerv(BUFFER_MAP_POINTER)` |
| Copy | `glCopyBufferSubData(readTarget, writeTarget, readOffset, writeOffset, size)` |

`glMapBufferRange` returns the pointer "once all pending operations on that buffer have completed" (ES 3.0
§2.10.3), which is why a `MAP_READ_BIT` mapping straight after a draw sees consistent data, and why
`MAP_UNSYNCHRONIZED_BIT` exists: it skips that wait. The pointer is only good until `glUnmapBuffer`.

### The rules (ES 3.0 §2.10.3)

`glMapBufferRange` fails and returns NULL with **INVALID_OPERATION** if

- `length` is zero (note: INVALID_OPERATION, not INVALID_VALUE),
- the buffer is already mapped,
- neither `MAP_READ_BIT` nor `MAP_WRITE_BIT` is set,
- `MAP_READ_BIT` is combined with any of `INVALIDATE_RANGE`, `INVALIDATE_BUFFER`, `UNSYNCHRONIZED`,
- `MAP_FLUSH_EXPLICIT_BIT` is set without `MAP_WRITE_BIT`;

and with **INVALID_VALUE** if `offset` or `length` is negative or `offset + length` exceeds `BUFFER_SIZE` (also if
undefined access bits are set; meta-gl's typed `MapBufferAccessMask` cannot express that, so it is not tested -
see below). The demo checks **all 64 combinations of the six flags** against exactly this table, and for each of
the 19 valid ones the whole state of Table 2.9 (`BUFFER_MAPPED` TRUE, `BUFFER_ACCESS_FLAGS` = the bits passed,
`BUFFER_MAP_OFFSET`, `BUFFER_MAP_LENGTH`, `BUFFER_MAP_POINTER` = the returned pointer) with both the 32- and the
64-bit query, then the return to Table 2.8 after `glUnmapBuffer`.

`glFlushMappedBufferRange`: INVALID_VALUE for a negative offset/length or a range beyond the **mapping** (not the
buffer); INVALID_OPERATION when nothing is bound, the buffer is not mapped, or it is mapped without
`FLUSH_EXPLICIT`. `glUnmapBuffer` on an unmapped buffer returns FALSE and raises INVALID_OPERATION.

`glCopyBufferSubData`: INVALID_VALUE for negative values, ranges beyond either buffer, or - when the same buffer
is bound to both targets - **overlapping** ranges (`[0,144)` -> `[144,288)` only touches and is legal);
INVALID_OPERATION when either target has no buffer or either buffer is mapped. The `COPY_READ_BUFFER` and
`COPY_WRITE_BUFFER` targets exist so that a copy does not disturb `ARRAY_BUFFER` or `ELEMENT_ARRAY_BUFFER`; the demo
verifies that `ARRAY_BUFFER_BINDING` survives a copy.

`glBufferData` and `glDeleteBuffers` on a mapped buffer unmap it silently (no error); `glBufferSubData` over any
part of a mapped range is INVALID_OPERATION.

## What is not asserted

The specification makes only conditional promises, and the demo asserts nothing beyond them. A CPU copy of each
buffer records, byte by byte, **whether the value is defined**; the exact-bytes checks compare only defined bytes
and print how many were skipped.

- `MAP_INVALIDATE_RANGE_BIT`: the old contents of the **range** may vanish - data outside it must survive
  (cell 3 checks quads 0 and 3). `MAP_INVALIDATE_BUFFER_BIT`: the old contents of the **entire buffer** may vanish -
  after cell 12 only the written range is compared, quads 0 and 3 are skipped and not drawn.
- `MAP_FLUSH_EXPLICIT_BIT`: only flushed ranges are guaranteed to arrive (cells 4 and 5); an unflushed modified
  range is undefined, so cell 5 never reads or draws quad 2 and **does not claim that it did not arrive**.
- `MAP_UNSYNCHRONIZED_BIT`: legal as long as pending operations do not touch the mapped range. Cell 6 maps four
  disjoint ranges; the draw of each is queued before the next range is written.
- **Drawing from a mapped buffer**: ES 3.0 §2.10.3.2 says a command that reads a mapped buffer "may generate an
  INVALID_OPERATION error ... only commands which explicitly describe this error are required to do so"; draws do
  not. Cell 14 therefore draws from the mapped buffer once, **prints** what the driver did (`[note]` line) and then
  clears the cell and redraws it from the unmapped buffer. On the Mesa drivers of the test matrix the draw raised no
  error. The commands that *do* require INVALID_OPERATION (`glBufferSubData` over a mapped range, `glCopyBufferSubData`
  on a mapped buffer, a second `glMapBufferRange`) are asserted.
- Writing through a pointer whose mapping lacks `MAP_WRITE_BIT`, using a pointer after `glUnmapBuffer`, and accessing
  memory outside the mapped range are undefined and never done.

## Shader concepts

`quads.vert` has two inputs with explicit locations (`layout(location = 0) in vec2 a_position`,
`layout(location = 1) in vec4 a_color`; the byte color is a normalized attribute) and an optional `uniform vec2
u_shift`. `u_shift` moves a draw in NDC: cells 7 and 8 copy a *template* quad (bottom-row geometry) into other slots of
a buffer and draw those slots shifted up, so the copy shows up as a visible, separately positioned quad.
`quads.frag` writes the interpolated color with `highp` precision, so equal vertex colors give exactly the stored byte.

## Validation (`--validate`)

142 demo checks per run (plus 5 runtime checks); on radeonsi three of them are reported as XFAIL, see below:

- **Exact bytes (19 comparisons, plus 2 reads through a mapped pointer):** after each cell's operation the whole buffer is mapped through `COPY_READ_BUFFER`
  with `MAP_READ_BIT` and compared with the CPU copy (defined bytes only, count printed).
- **Pixels (16 checks):** for each cell, 24 probes - the center and four points 6 pixels inside the corners of each
  quad, the cell center (the gap between quads) and three background points - must equal the color the CPU says (the
  bytes that were written, inverted, swapped, copied ...) within **1/255** (the byte becomes `c/255` as a normalized
  attribute, equal colors interpolate to the same value, and ES 3.0 §2.1.2 allows 1/255 in the conversion back to
  8 bits). Absent quads, the gaps and the borders must show the background **exactly**.
- **Flag matrix (3 checks):** all 64 combinations of the six access flags against the spec's table, mapped state
  and 32/64-bit queries of the 19 valid ones, the unmapped state afterwards.
- **Errors (about 40 checks):** every case listed above, each requiring the returned pointer to be NULL and
  the exact error code; failed calls must change no state (the first mapping survives a second `glMapBufferRange`;
  the buffer data survives every failed copy).
- **Negative controls:** a wrong color is written through a mapping and the pixel check must see it (and the restored
  value must pass again); the byte comparison must report a flipped byte and must skip a byte declared undefined.
- The runtime additionally requires `glGetError() == GL_NO_ERROR` after setup, after the frame and after
  `shutdown()`.

### Known driver deviation

Mesa's `glthread` (enabled by default on radeonsi, the "gpu" test) answers `glUnmapBuffer` immediately with TRUE
without looking at the buffer's state. The INVALID_OPERATION error is still raised, but the **return value FALSE**
required by ES 3.0 §2.10.3.1 for an unmapped buffer is not delivered. The three affected checks
(`... returns FALSE`) stay strict and are labelled XFAIL on `radeonsi`; with `mesa_glthread=false` they pass.

## Spec references

ES 3.0 §2.5 Errors · §2.10.1 Creating and Binding Buffer Objects · §2.10.2 Creating Buffer Object Data Stores
(Table 2.8) · §2.10.3 Mapping and Unmapping Buffer Data (Table 2.9, §2.10.3.1 Unmapping Buffers, §2.10.3.2 Effects
of Mapping Buffers on Other GL Commands) · §2.10.5 Copying Between Buffers · §2.10.6 Vertex Arrays in Buffer
Objects · §6.1.9 Buffer Object Queries · Table 2.7. ES 3.2 §6.3 - §6.6 carry the same rules.
GLSL ES 3.00 §4.3.4 Input Variables · §4.3.8.1 Input Layout Qualifiers.

## Pitfalls

- A mapping is a loan: do not hand the pointer to the GL, do not keep it past `glUnmapBuffer`, and do not
  assume `BUFFER_MAP_POINTER` stays valid across a `glBufferData` (that unmaps).
- `MAP_WRITE_BIT` without `MAP_READ_BIT` gives **no promise about the bytes you do not write** - write the whole
  range, or ask for `READ | WRITE`.
- `MAP_FLUSH_EXPLICIT_BIT` offsets are relative to the **mapping**, not to the buffer: flushing the second quad of a
  mapping that starts at byte 72 means offset 72 - 72 = 0.
- `MAP_UNSYNCHRONIZED_BIT` is only safe when you know no pending draw reads the range, typically a ring buffer where each
  region is written once per cycle; `MAP_INVALIDATE_BUFFER_BIT` (orphaning) is the safe way to start a new cycle.
- `glMapBufferRange(length = 0)` is INVALID_OPERATION in OpenGL ES, and `MAP_WRITE_BIT | MAP_FLUSH_EXPLICIT_BIT`
  without any flush call leaves the data undefined - no error tells you.
- Whether a draw from a mapped buffer fails is implementation-dependent: do not rely on it either way.
- Not tested: an access bitfield with undefined bits (INVALID_VALUE per ES 3.0 §2.10.3) cannot be formed with meta-gl's
  typed `MapBufferAccessMask` without a raw cast.
