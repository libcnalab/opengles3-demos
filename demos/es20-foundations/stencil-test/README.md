# es20-stencil-test

**API introduced in:** OpenGL ES 2.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

The stencil buffer is a per-pixel 8-bit integer that you can test against, and update as a side
effect of drawing. The rules are small but easy to misremember: which operand is on which side of
the comparison, what the value mask and the write mask each do, where an operation saturates and
where it wraps, which of the three operations (`sfail`, `dpfail`, `dppass`) fires when the depth
test is involved, and how front- and back-facing triangles pick their state. This demo exercises all
of them and checks every outcome against a small CPU model of the specification.

OpenGL ES cannot read the stencil buffer back (`glReadPixels` returns color only, ES 3.0 §4.3.2).
The demo therefore **measures the stencil value with the stencil test itself**:

1. The experiments run with the color mask off, so they change only stencil (and depth).
2. A *readout* then draws a full-window triangle 255 times. Draw number `v` uses
   `glStencilFunc(GL_EQUAL, v, 0xFF)`, `glStencilOp(KEEP, KEEP, KEEP)` and writes the color
   `(v/255, 0.4, 0.4)`. A pixel whose stencil value is `v` passes exactly one of the 255 tests, so
   its red channel ends up equal to `v` (value 0 needs no draw: the cell was cleared to the color
   for 0). Reading the red channel therefore reads the stencil value, exactly, for all 256 values.

## The image

Top part: a grid of cells (16 x 14 pixels at the 256 x 256 design size). Each cell is cleared to a start value, receives one or more
draws with the state under test, and is then read out; **red = the stencil value left behind**
(teal = 0, salmon = 255). Bottom part: two classic uses.

| Region | Content |
|---|---|
| A (top left, 8 x 5) | The eight stencil operations (columns: KEEP, ZERO, REPLACE, INCR, DECR, INVERT, INCR_WRAP, DECR_WRAP) applied to the start values 0, 1, 0x55, 254, 255 (rows). `REPLACE` writes the reference 42. |
| B (top right, 8 x 7) | The eight stencil functions (columns: NEVER, LESS, EQUAL, LEQUAL, GREATER, NOTEQUAL, GEQUAL, ALWAYS) against seven stored/reference/mask combinations (rows). A failed test zeroes the pixel, a passed test keeps it. |
| C (left, sixth row) | `glStencilMask`: only masked bits change. Six draws with different masks and operations, then two `glClear(STENCIL)` calls (masked clear, and a front/back mask pair showing the clear uses the front mask). |
| D (left, seventh row) | `sfail` / `dpfail` / `dppass` together with the depth test, the depth mask and a disabled stencil test. |
| E (row below) | Separate front/back state: winding order, `glFrontFace`, a point, `glStencilFuncSeparate`/`OpSeparate`/`MaskSeparate`, `FRONT_AND_BACK`, and the non-separate calls reaching the back face. |
| Bottom left | A stencil-masked *window*: a disc is written to the stencil buffer only, the "inner world" (warm stripes) is drawn where stencil == 1, the "outer world" (cool stripes) everywhere else, and a white *outline* ring is drawn where stencil != 1 with a larger disc. |
| Bottom right | An *even-odd fill*: a pentagram drawn as a triangle fan with `GL_INVERT`, then filled where the stencil bit is set. The five arms (covered once) are filled; the inner pentagon (covered twice) stays empty. |

### Expected values of panel A

| start | KEEP | ZERO | REPLACE | INCR | DECR | INVERT | INCR_WRAP | DECR_WRAP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 42 | 1 | **0** | 255 | 1 | **255** |
| 1 | 1 | 0 | 42 | 2 | 0 | 254 | 2 | 0 |
| 85 | 85 | 0 | 42 | 86 | 84 | 170 | 86 | 84 |
| 254 | 254 | 0 | 42 | 255 | 253 | 1 | 255 | 253 |
| 255 | 255 | 0 | 42 | **255** | 254 | 0 | **0** | 254 |

The bold entries are the boundary cases: `INCR` and `DECR` saturate, the `_WRAP` forms wrap.

### Panels B, C, D and E in one place

- **B**, rows (stored value, reference, mask): `100/50/0xFF` (ref < stored), `100/100/0xFF` (equal),
  `100/150/0xFF` (ref > stored), `0x65/0x35/0x0F` (masked low nibbles equal although ref < stored),
  `0x1A/0xF3/0x0F` (masked ref < stored although unmasked ref > stored), `255/263/0xFF` (the
  reference 263 is *clamped* to 255, not masked to 7) and `1/-1/0xFF` (-1 clamps to 0).
- **C**: `0xAA` through mask `0x0F` with REPLACE 0x35 gives `0xA5`; `0xAA` through `0x3C` with INVERT
  gives `0x96`; `INCR` of `0x0F` through `0xF0` gives `0x1F` (the carry leaves the masked nibble);
  mask `0x00` changes nothing; `glClear` to 0xFF through `0x81` gives `0x81`, and with front mask
  `0x0F` / back mask `0xF0` gives `0x0F`.
- **D** (start 0x10, depth 0.5, ops sfail=INCR, dpfail=ZERO, dppass=REPLACE 0x7E): stencil fails ->
  17; stencil passes and depth fails -> 0; both pass -> 126; depth test disabled -> 126 (treated as
  passed); a stencil-failed fragment stores no depth, so a later 0.4 fragment passes -> 126; a
  passing fragment does store its depth, so the later one fails -> 0; `DepthMask(FALSE)` keeps
  the depth out but the stencil still updates -> 127; stencil test disabled -> 16 (untouched).
- **E** (start 100 unless noted): CCW triangle with front INCR / back DECR -> 101, CW -> 99, both
  swapped by `glFrontFace(CW)`; a point always uses the front state -> 101; INCR_WRAP then DECR_WRAP
  from 255 -> 255; front `EQUAL 100` passes (INVERT) -> 155, back `EQUAL 7` fails (sfail ZERO) -> 0;
  front mask `0x0F` -> `0x0F`, back mask `0xF0` -> `0xF0`; `FRONT_AND_BACK` and the non-separate
  calls reach the back face.

## Central calls

`glEnable/glDisable(GL_STENCIL_TEST)`, `glStencilFunc`, `glStencilFuncSeparate`, `glStencilOp`,
`glStencilOpSeparate`, `glStencilMask`, `glStencilMaskSeparate`, `glClearStencil` + `glClear`,
`glFrontFace`, `glColorMask`, and the queries `glIsEnabled` and `glGetIntegerv`
(`STENCIL_FUNC`, `STENCIL_REF`, `STENCIL_VALUE_MASK`, `STENCIL_FAIL`, `STENCIL_PASS_DEPTH_FAIL`,
`STENCIL_PASS_DEPTH_PASS`, `STENCIL_WRITEMASK`, the `STENCIL_BACK_*` variants, `STENCIL_CLEAR_VALUE`,
`STENCIL_BITS`).

## Spec sections

- ES 3.0 §4.1.4 Stencil Test: the test passes if `(ref & mask) <func> (stored & mask)` - the
  *reference* is the left operand. `ref` is clamped to [0, 2^s - 1] for comparisons and queries.
  `INCR`/`DECR` saturate at the maximum and 0; `INCR_WRAP`/`DECR_WRAP` wrap. Front state is used for
  front-facing polygons and for points and lines; back state for back-facing polygons. The
  non-separate commands set both. Initial state: test disabled, func ALWAYS, ref 0, mask all ones,
  all operations KEEP.
- ES 3.0 §4.1.5 Depth Buffer Test: a fragment that fails the stencil test is discarded (so it
  updates no depth); `dpfail` applies when the depth test fails, `dppass` when it passes **or is
  disabled**.
- ES 3.0 §4.2.2 Fine Control of Buffer Updates: `StencilMask` writes only the set bits; "the clear
  operation always uses the front stencil write mask".
- ES 3.0 §4.2.3 Clearing the Buffers: `ClearStencil` is masked to the bit planes; the stencil *test*
  does not apply to a clear, the write mask and scissor do.
- ES 3.0 §3.6.1 Basic Polygon Rasterization: facing from the sign of the window-space area and
  `FrontFace`; the edge rule (a pixel on an edge shared by two polygons is produced exactly once).
- ES 3.0 §3.4 Points; §4.3.2 Reading Pixels; §4.4.5 (`STENCIL_BITS`); §6.1.1 Simple Queries.
- GLSL ES 3.00 §7.1 (`gl_VertexID`, `gl_PointSize`) and §7.2 (`gl_FragCoord` in `stripes.frag`).

## Shader concepts

All geometry is generated from `gl_VertexID`, with no vertex attributes. `fullscreen.vert` emits **one
triangle** larger than the viewport. A quad would have a diagonal, and `INCR`/`DECR`/`INVERT` act once
per fragment, so double coverage of a pixel would change the result; one triangle has no internal
edge and covers every pixel exactly once. A uniform flips the vertex order to get a clockwise
(back-facing) triangle. `point.vert` writes `gl_PointSize`; `circle.vert` and `star.vert` are
triangle fans. `stripes.frag` colors diagonal stripes from `gl_FragCoord` with a power-of-two period,
which keeps the stripe index exact so that the CPU can predict every pixel.

## Validation

199 demo checks:

- **125 cell probes** (A 40, B 56, C 8, D 8, E 13). The CPU model (`model_run()`) applies the rules
  quoted above to one pixel and returns the expected stencil value; the probe pixel must equal it
  exactly (tolerance 0 - a stencil value off by one is a real failure, and the readout colors are
  exact 8-bit values).
- **20 visual probes** (window/outline 9, star 11). The expected color comes from the geometry on the
  CPU: distance from the disc center for the window; for the star, the number of fan triangles that
  contain the pixel center (barycentric tests), whose parity decides filled or empty. Probes avoid the
  fan's internal edges and the disc boundaries, and use pixel centers (where the rasterizer samples).
- **54 state checks:** initial state, `STENCIL_BITS >= 8`, every function and operation round trip
  (front and back), `glStencilFuncSeparate`/`OpSeparate`/`MaskSeparate` touching only the named face,
  `ref` clamping on query, `glIsEnabled`, `glClearStencil`, plus a final `glGetError`.

Run: `es20-stencil-test --headless --validate --capture out.png`. The layout is designed on a
256 x 256 grid and stretched to the surface, independently in x and y: every design edge maps to
`round(edge * size / 256)`, so neighbouring cells share their edges and the picture fills any surface
(on a wide or tall surface the cells, the portal and the star are stretched; the probes are computed
in the panel's NDC, so they follow). **Minimum surface: 64x64**; below that a cell is smaller than
4 x 3.5 pixels and the visual panels are too coarse for the probe margins, and the demo reports
`SKIP_UNSUPPORTED`.

**Limits.** The readout is not independent of the stencil test: it measures with `EQUAL` and `KEEP`.
If `EQUAL` were broken, the 125 cells would disagree with the CPU model in obvious ways, and the
window and star panels check the stencil test through ordinary color outcomes. Still, a run
validates one implementation at a time, and radeonsi and llvmpipe are both Mesa drivers sharing a
large part of the state handling.

## Pitfalls

- **Operand order.** The stencil test compares `ref` *to* the stored value (`ref < stored` for
  `GL_LESS`); the depth test compares the incoming depth to the stored one. Reusing the habit from
  depth testing inverts every ordered stencil function.
- **Two masks.** The `mask` argument of `glStencilFunc` selects which bits take part in the
  comparison. `glStencilMask` selects which bits can be written. They are independent.
- **`ref` is clamped, not masked.** With an 8-bit buffer `glStencilFunc(.., 263, ..)` uses 255. `-1`
  uses 0, and `GL_STENCIL_REF` reports the clamped value.
- **`glClear` obeys `glStencilMask`** and uses the *front* mask only. A mask left at `0xF0` after
  drawing makes the next `glClear(GL_STENCIL_BUFFER_BIT)` clear half the bits. The stencil test does
  not apply to a clear.
- **Saturate versus wrap.** `GL_INCR`/`GL_DECR` stop at 255/0. Use `GL_INCR_WRAP`/`GL_DECR_WRAP` for
  counting algorithms (shadow volumes, nested clip regions), where a wrap is harmless because
  front and back faces cancel.
- **Which operation fires.** `sfail` when the stencil test fails; otherwise `dpfail` if the depth
  test fails, `dppass` if it passes. With the depth test *disabled* the fragment counts as
  passing depth. A fragment that fails the stencil test never updates depth, and `glDepthMask`
  does not affect stencil updates.
- **Disabled test, no updates.** With `GL_STENCIL_TEST` disabled, no stencil operation runs, whatever
  `glStencilOp` says. With no stencil buffer the test always passes.
- **Facing.** Polygons use `glFrontFace` (default CCW); points and lines always use the *front*
  state. `glStencilFunc`/`glStencilOp`/`glStencilMask` set both faces; if you only want one, use the
  `*Separate` forms.
- **Overdraw counts.** Any pixel covered by two triangles gets the operation twice. The edge rule
  keeps a shared edge from doing that, but overlapping geometry does - which is exactly how the
  even-odd star works.
