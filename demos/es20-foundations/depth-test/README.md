# es20-depth-test

**API introduced in:** OpenGL ES 2.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

The depth buffer decides which of two fragments that land on the same pixel survives. Every
mistake in this area looks the same on screen ("the wrong thing is in front") but has a different
cause: the wrong comparison function, a write mask left off, a stale clear value, a depth range
that does not match the projection, or two surfaces that sit at exactly the same depth. This demo
isolates each cause in its own small experiment, predicts the outcome on the CPU from the
specification, and compares it with what the GPU drew.

OpenGL ES cannot read the depth buffer back (`glReadPixels` only returns color, ES 3.0 §4.3.2), so
every experiment is built so that **the depth values decide which color survives**. A base quad is
drawn first and leaves a known depth behind; a second quad (or several stripes) is drawn on top
with the state under test; the color of a probe pixel then tells which fragment won.

## The image

The window is a 4 x 4 grid. Each cell is an independent experiment with its own viewport and its
own cleared color and depth. Reading left to right, top to bottom:

| Cell | Experiment | What it shows |
|---|---|---|
| 0-7 | `glDepthFunc` NEVER, LESS, EQUAL, LEQUAL, GREATER, NOTEQUAL, GEQUAL, ALWAYS | The blue base quad is drawn with ALWAYS and stores window depth 0.5. Three stripes follow with the function under test: green is nearer (0.25), yellow is equal (0.5), magenta is farther (0.75). A stripe is visible only where its depth test passes, so the cell is a truth table. |
| 8 | depth test disabled | A farther fragment is still drawn, and the stored depth is **not** updated: a later fragment at 0.4 (orange) passes against the old 0.5. |
| 9 | `glDepthMask` | Left stripe (yellow) is drawn with the mask off, right stripe (green) with it on. An orange fragment at 0.4 then wins over the left one (depth was not stored) and loses to the right one (0.25 was stored). |
| 10 | `glClear` and the depth mask | The depth buffer is cleared to 1.0 in both halves, the left one with `glDepthMask(GL_FALSE)`. A 0.75 fragment (green) fails on the left (clear did nothing) and passes on the right. |
| 11 | `glClearDepthf` | Four bands cleared to -3, 0.4, 0.6 and 5; the out-of-range values are clamped to 0 and 1. A fragment at 0.5 passes LESS only in the right two bands. |
| 12 | `glDepthRangef` | The same NDC depth is drawn with four ranges (default, `[0.5, 1]`, `[0, 0.5]`, reversed `[1, 0]`); the range moves the window depth across the stored 0.5 and flips the result. |
| 13 | `gl_FragDepth` | Stripe by stripe: a shader depth overrides an interpolated one in both directions, the *written* depth is what is stored, a depth range is not applied to it, and a depth that ramps with `gl_FragCoord.x` gives a boundary inside one primitive. |
| 14 | `glPolygonOffset` on coplanar quads | Four decals drawn exactly on top of the base quad: with LESS they lose (equal is not less) unless a negative offset is enabled; with LEQUAL they win unless a positive offset pushes them back. |
| 15 | `glPolygonOffset` factor | The base is a tilted plane; the decals are the same plane pushed back by two depth slopes. Only a large negative `factor` brings a decal forward; `units` alone (a few depth-buffer LSBs) cannot. |

## Central calls

`glEnable/glDisable(GL_DEPTH_TEST)`, `glDepthFunc`, `glDepthMask`, `glClearDepthf` + `glClear`,
`glDepthRangef`, `glEnable(GL_POLYGON_OFFSET_FILL)` + `glPolygonOffset`, `glViewport`, `glScissor`,
and the queries `glIsEnabled`, `glGetBooleanv`, `glGetIntegerv`, `glGetFloatv`.

## Spec sections

- ES 3.0 §4.1.5 Depth Buffer Test: the comparison is *incoming* `<func>` *stored*. With the test
  disabled "the depth comparison and subsequent possible updates to the depth buffer value are
  bypassed". Initial state: function LESS, test disabled.
- ES 3.0 §4.2.2 Fine Control of Buffer Updates: `DepthMask(FALSE)` disables depth writes.
- ES 3.0 §4.2.3 Clearing the Buffers: `ClearDepthf` clamps its argument to [0,1], and "the masking
  operations described in section 4.2.2 are also applied" to a clear, as is the scissor test.
- ES 3.0 §2.13.1 Controlling the Viewport: `z_w = (f - n) / 2 * z_d + (n + f) / 2`; `n` and `f`
  are clamped to [0,1] but may be given in either order.
- ES 3.0 §3.6.2 Depth Offset: `o = m * factor + r * units` (equation 3.11) where `m` is the depth
  slope of the polygon and `r` the smallest resolvable depth difference. Applied only to polygons
  and only while `POLYGON_OFFSET_FILL` is enabled. Initial `factor` and `units` are 0.
- ES 3.0 §3.9.2.3 Shader Outputs and GLSL ES 3.00 §7.2: `gl_FragDepth` replaces the fragment depth.
  The value is clamped to [0,1] and converted to fixed point "as if it were a window z value";
  "the depth range computation is not applied here".
- ES 3.0 §6.1.1 Simple Queries, and ES 3.0 §4.4.5 for `DEPTH_BITS`.

## Shader concepts

`quad.vert` has no vertex attributes. The four corners of a triangle strip come from `gl_VertexID`;
the rectangle (`u_rect`) and the depth (`u_depth = z0, slope`) are uniforms, so a quad can be given
any NDC depth, including a tilt along x for cell 15. With slope 0 the depth is the same constant at
every vertex, so every fragment of every such quad gets the same depth. The EQUAL tests and the
coplanar decals depend on that, and they use an NDC depth of exactly 0.0 (window depth 0.5) so no
rounding can make two copies of "the same" depth differ.

`solid.frag` leaves the depth to the fixed-function pipeline. `fragdepth.frag` writes
`gl_FragDepth = a + b * gl_FragCoord.x`: a constant when `b = 0`, a ramp otherwise. It writes the
variable unconditionally, because a shader that assigns `gl_FragDepth` on some path only leaves the
depth undefined on the others (GLSL ES 3.00 §7.2).

## Validation

`validate()` runs 92 checks: 56 probed pixels, 35 state checks and a final `glGetError` check.

- **Pixels.** The expected color of every probe is computed on the CPU from the rules above (a
  `depth_test_passes()` function and the §2.13.1 / §3.6.2 formulas) and compared with a tolerance of
  1 (8-bit quantization; all colors are exact multiples of 1/255). Depths are chosen far apart
  (0.25 / 0.5 / 0.75, or 2 depth slopes) so that no result depends on depth-buffer precision. The
  EQUAL cases compare a depth with a bit-identical copy of itself. Probes sit in the middle of
  stripes, at least 3 pixels (usually 7) from any primitive edge.
- **Initial state** (read in `init()` before anything is touched): test disabled, function LESS,
  mask TRUE, range (0, 1), clear value 1, offset (0, 0), `DEPTH_BITS >= 16`.
- **Round trips:** every depth function, the mask, depth range including the clamp of (-1, 2) to
  (0, 1) and the reversed range, clear value clamping, `glPolygonOffset` with either sign,
  `glIsEnabled` for both enables.

Run: `es20-depth-test --headless --validate --capture out.png`.

### Known driver deviation: `gl_FragDepth` and the depth range

ES 3.0 §3.9.2.3 (unchanged in ES 3.1 and 3.2) says the depth range is **not** applied to a depth
written by a fragment shader. Stripe 3 of cell 13 tests this: with `glDepthRangef(0.5, 1)` and
`EQUAL`, a shader depth of 0.25 must stay 0.25 and fail against the stored 0.5. Mesa's llvmpipe
behaves like that. Mesa's radeonsi (25.0.7, reference machine) clamps the exported depth into the
depth range, turning 0.25 into 0.5, so the test passes. `validate()` accepts either of those two
outcomes (anything else fails), prints a `NOTE` for the second, and names the deviation in the check
text. It is deliberately not listed as a verified feature in `demo.json`.

## Pitfalls

- **`glClear(GL_DEPTH_BUFFER_BIT)` obeys `glDepthMask`.** If the mask was left `FALSE` after drawing
  transparent geometry, the next frame's depth clear silently does nothing (cell 10). Always set the
  mask to `TRUE` before clearing.
- **A disabled depth test also disables depth writes**, whatever the mask says (cell 8).
- **`GL_LESS` is strict.** It passes when the incoming depth is smaller; an equal depth fails it.
  Coplanar geometry therefore keeps whichever surface was drawn first (cell 14).
- **Equal depths are only reliable for identical arithmetic.** Two different shaders, or the same
  surface transformed by different matrices, may differ in the last bit. Use `invariant gl_Position`
  or a single shared transform if you rely on `GL_EQUAL`/`GL_LEQUAL` (GLSL ES 3.00 §4.6).
- **Polygon offset needs `POLYGON_OFFSET_FILL`**, applies to polygons only (not lines or points),
  and its sign matters: negative values move toward the viewer. `units` is a multiple of an
  implementation-dependent smallest step, so tune it per depth format; `factor` scales with the
  polygon's slope and is what rescues surfaces seen at grazing angles (cell 15).
- **`gl_FragDepth` is a window-space depth.** It is not passed through `glDepthRangef`, and a
  shader that writes it for some fragments must write it for all (and loses the driver's ability
  to reject fragments early).
- **A reversed or compressed depth range is legal.** `glDepthRangef(1, 0)` flips the sense of the
  depth test; `n` and `f` are clamped to [0,1] but not reordered.
