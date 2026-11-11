# es20-clear-viewport-scissor

**API introduced in:** OpenGL ES 2.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

Before a frame draws anything it has to be prepared, and every pixel that gets drawn passes through
a few small stages that decide *where* it may land. This demo shows those stages one by one and
verifies each of them:

- the **clear values** (`glClearColor`, `glClearDepthf`, `glClearStencil`) and `glClear` for each buffer;
- the **write masks** (`glColorMask`, `glDepthMask`, `glStencilMask`) and how a clear honours them;
- the **scissor test** (`glScissor` + `GL_SCISSOR_TEST`) and how it limits clears and draws;
- the **viewport transform** (`glViewport`, `glDepthRangef`) that maps normalized device coordinates
  to window coordinates.

The surprise for most newcomers is that these are *different mechanisms*. A clear ignores the
viewport and obeys the scissor box. A draw obeys both, but the viewport only moves and scales the
geometry while the scissor box actually cuts pixels away. The demo makes that visible.

OpenGL ES cannot read the depth or stencil buffer back: `glReadPixels` reads color buffers only
(ES 3.0 §4.3.2). The demo therefore *observes* them indirectly: after the clear under
test it draws a probe quad with the depth or stencil test enabled, and the color buffer shows
which pixels passed. A panel thus displays the content of a buffer you cannot read.

## The gallery

The image is a 4x4 grid of 64x64-pixel panels. Each panel is one experiment, isolated from its
neighbours by its own scissor box and viewport. Panels are named A-P in reading order.

| | col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|---|
| **row 1** | **A** `glClearColor` + `glClear(COLOR)` fills the box | **B** a scissored clear changes only a 40x30 box | **C** `glColorMask`: a white clear writes only R, G, B or A in four strips | **D** the same masks applied to a draw |
| **row 2** | **E** depth "ruler": four bands cleared to depth 0.125/0.375/0.625/0.875; 16 probe columns at `z_w = (i+0.5)/16` pass `GL_LESS` where `z_w` is below the band's value | **F** depth clear under a scissor box, and a second clear with `glDepthMask(FALSE)` that changes nothing | **G** `glDepthRangef(n, f)`: four ranges x probe depths of +-0.3 against a clear value of 0.6 | **H** a small viewport does not limit `glClear`; the NDC square still lands exactly on the viewport |
| **row 3** | **I** stencil "bit ruler": four bands cleared to 0x35, 0xC9, 0x1AB (masked to 8 bits: 0xAB) and 0x00; eight columns show the bits | **J** `glStencilMask`: partial clears of 0xA5 give 0xAF, 0x05, 0xA5 and 0x3C (front mask rules the clear) | **K** one `glClear(COLOR\|DEPTH\|STENCIL)`, each buffer then observed in its own strip | **O** the scissor box is ignored while the test is disabled, and cuts a full-panel quad when enabled |
| **row 4** | **L** viewport `(8,12,40,24)` maps NDC `[-1,1]^2` and `[-0.5,0.5]^2` to exact pixel rectangles | **M** one NDC triangle drawn through four 32x32 viewports | **N** a viewport whose lower-left corner is outside the window (`-32,-32`) | **P** a triangle far larger than the clip volume is clipped to exactly the viewport rectangle |

Panel I reads left to right as the stencil value in binary, most significant bit first, so band
0x35 is `00110101`. Panels E and G show a lit pixel where the probe passed the depth test.

## Central calls and the rules behind them

| Topic | Calls | Rule (ES 3.0) |
|---|---|---|
| Clear | `glClear(mask)`, `glClearColor`, `glClearDepthf`, `glClearStencil` | Each buffer is set to its clear value (§4.2.3). Color is clamped to [0,1] and converted to fixed point; depth is clamped to [0,1] when specified; the stencil value is masked to the number of bit planes. A mask of zero clears nothing and is not an error. |
| Clear and per-fragment ops | `glScissor`, `glEnable(GL_SCISSOR_TEST)` | Only the pixel ownership test, the scissor test, sRGB conversion and dithering apply to a clear. A fragment passes the scissor test iff `left <= x < left + width` and `bottom <= y < bottom + height` (§4.1.2). With the test disabled the box is ignored. A box of width or height 0 rejects everything. Negative sizes give `INVALID_VALUE`. |
| Write masks | `glColorMask`, `glDepthMask`, `glStencilMask`, `glStencilMaskSeparate` | Masks apply to clears as well as draws (§4.2.2). A clear always uses the *front* stencil write mask. |
| Viewport transform | `glViewport(x, y, w, h)` | `x_w = (w/2) x_d + (x + w/2)`, likewise for y (§2.13.1). Width and height are clamped to `GL_MAX_VIEWPORT_DIMS`; negative sizes give `INVALID_VALUE`; the origin may be negative. |
| Depth range | `glDepthRangef(n, f)` | `z_w = (f-n)/2 z_d + (n+f)/2`; `n` and `f` are clamped to [0,1]; `n > f` is allowed (§2.13.1). The clear depth is a window-space value and is not affected by the range. |
| Clipping | | Geometry is clipped to the clip volume, which the viewport transform maps onto the viewport rectangle (§2.18). |
| Initial state | | Viewport and scissor box equal the surface size, scissor test off, clear color (0,0,0,0), clear depth 1, clear stencil 0, all masks enabled, depth range (0,1). |

Other sections used: §4.1.4 stencil test and §4.1.5 depth test (as observers), §4.3.2 reading
pixels, and the state tables 6.5, 6.11, 6.12 and 6.28 for the queries.

## Shader concepts

There is no vertex buffer. `shaders/rect.vert` takes its (up to four) clip-space corners from a
`uniform vec2 u_vertex[4]` indexed by the built-in `gl_VertexID`, and the depth from `u_z`; drawn as
a 4-vertex `GL_TRIANGLE_STRIP` it is a quad, drawn as 3 vertices a triangle. `w` is 1, so clip
coordinates are NDC and the window position is exactly the viewport transform of §2.13.1.
`shaders/solid.frag` writes one flat `u_color` to its user-declared output (GLSL ES 3.00 has no
`gl_FragColor`). A fragment shader must declare a default float precision (`precision mediump
float;`), otherwise it does not compile.

## Expected image

Sixteen distinct, flat-colored panels (see the table). Panel E is a staircase growing towards the
top; panel F is green with a dark square; panel I and J show eight-column bit patterns; panel K is
purple with two yellow strips; the bottom row shows teal/pink rectangles, four triangles, a mint
and a coral square, and a violet rectangle.

## Validation

`es20-clear-viewport-scissor --headless --validate` runs 78 checks.

- **Gallery (36 checks).** For every panel the CPU paints the expected 64x64 image from the rules
  above - for example "pixel is lit iff `z_w < clear depth`", "value after masked clear = `(old &
  ~mask) | (clear & mask)`", "pixel is inside the triangle by at least 1 px" - and compares it with
  a `glReadPixels` of the framebuffer. Colors are compared to +-1 (the spec allows a difference in
  rounding of `c * 255`). Probe pixels are exact: scissor boxes and viewports are on integer window
  coordinates, so pixel centers are never on an edge. Only the triangle panel (M) skips pixels
  within 1 px of an edge. Named checks additionally probe the four corner pixels of a scissor box
  and the pixels just outside it.
- **State (25 checks).** Initial values of viewport, scissor box and test, clear values, masks and
  depth range (10); round trips of every setter through `glGet*` and `glIsEnabled`, clamping of
  `glClearDepthf` and `glDepthRangef`, and the independent back stencil mask (15).
- **Limits and errors (9 checks).** `GL_MAX_VIEWPORT_DIMS` covers the surface and an oversized
  viewport is clamped to it without an error; `INVALID_VALUE` for negative viewport and scissor
  sizes, after which the state is unchanged.
- **Corner cases (8 checks).** An empty `glClear` mask, a 0x0 and a zero-width scissor box, a
  disabled scissor test with a tiny box, an all-false color mask, and clamping of out-of-range
  clear colors; no GL error is raised by any of them.

The corner cases draw into a scratch panel; `validate()` re-renders the gallery
afterwards so the capture shows the canonical image.

## Pitfalls

- **The viewport is not a clip rectangle for clears.** `glClear` ignores it. To limit a clear use the scissor test.
- **The scissor box does nothing until `GL_SCISSOR_TEST` is enabled.** It is also in window pixels,
  not NDC, and `glScissor` is not scaled by the viewport.
- **`glDepthMask(GL_FALSE)` also protects against `glClear(GL_DEPTH_BUFFER_BIT)`** (same for color and stencil masks). A frame whose depth buffer never clears is often a forgotten mask.
- **Initial viewport and scissor are the size of the surface when the context is first made current.**
  They do not follow a later window resize; call `glViewport` yourself.
- **`glClearDepthf`, not `glClearDepth`.** ES has only the float version. `glClearStencil` takes an
  `int` and silently drops bits above the stencil depth.
- **Integer color buffers are cleared with `glClearBuffer{i,ui}v`, not `glClear`** (the result of `glClear` is undefined for them); this demo only clears normalized fixed-point buffers.
- **Dithering is part of a clear.** `GL_DITHER` is enabled by default, so on low-precision buffers a
  clear may be dithered. With an 8-bit-per-channel buffer the difference is at most one level, which the checks allow.
- **Depth and stencil cannot be read back in ES.** Probing them with a test-gated draw, as done here, is the portable method.
