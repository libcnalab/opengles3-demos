# es30-framebuffer-blit-msaa

**API introduced in:** OpenGL ES 3.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 ·
**Run:** `es30-framebuffer-blit-msaa [--headless --validate]`

## Purpose

ES 2.0 could only get pixels from one framebuffer into another by rendering them (sample a texture,
draw a quad). ES 3.0 adds `glBlitFramebuffer`, a copy between a **read framebuffer** and a **draw
framebuffer** that can scale, mirror and filter, and that is the only way to **resolve** a multisample
renderbuffer into an image you can sample, read back or show. This demo takes the command apart:

- the two separate bindings, `READ_FRAMEBUFFER` and `DRAW_FRAMEBUFFER`, and `glReadBuffer` choosing which
  color attachment of the read framebuffer is the source;
- the copy rule: which destination pixels are written and what they receive, for 1:1 copies, sub-rectangles,
  `NEAREST` and `LINEAR` scaling, mirroring (swapped coordinates) and clipping;
- what a blit ignores (write masks, depth/stencil tests) and what it obeys (the scissor test);
- depth and stencil copies, which allow `NEAREST` only;
- multisample renderbuffers: asking `glGetInternalformativ` which sample counts exist, allocating with the
  largest, rendering into them and resolving by a blit;
- the errors section 4.3.3 defines.

## What the image shows

The 256 x 256 window is four 128 x 128 panels, each one a `glBlitFramebuffer` into the default framebuffer:

```
 top-left:     64x64 chart, 2x NEAREST        top-right:    the same chart, 2x LINEAR
 bottom-left:  copied depth/stencil, shown     bottom-right: a triangle rendered with 4-8 samples
               through tested draws                         and resolved by a blit
```

The chart is a 64 x 64 image in which red grows by 4 per pixel in x, green by 4 per pixel in y (so every pixel is
unique) and blue is an 8 x 8 checkerboard. The NEAREST panel is blocky; the LINEAR panel is smooth between the
texel centers but keeps the checkerboard edges soft. In the bottom-left panel, a depth+stencil scene of four regions
was copied 2x by one blit and then painted over by four draws that each pass only where a test passes
on the *copied* buffers: white where the copied depth is greater than 0.5, cyan where the copied stencil is 2,
gray where the depth is greater than 0.8, black where the stencil is 3. What remains of the original red is the
region whose copied depth (0.25) failed every depth test, so the picture is a direct view of the copied depth and
stencil values. In the bottom-right panel the triangle's edges are not stair-stepped: pixels on the edge are blends of
the two colors.

## Central calls

| Call | Role | Spec |
|---|---|---|
| `glBindFramebuffer(READ_FRAMEBUFFER, a)` / `(DRAW_FRAMEBUFFER, b)` | source and destination of a blit; `glReadPixels` uses READ, `glClear` and draws use DRAW | §4.4.1 |
| `glReadBuffer(COLOR_ATTACHMENTi)` | which attachment of the read framebuffer is copied | §4.3.1 |
| `glBlitFramebuffer(sx0,sy0,sx1,sy1, dx0,dy0,dx1,dy1, mask, filter)` | the copy | §4.3.3 |
| `glRenderbufferStorageMultisample(RENDERBUFFER, samples, format, w, h)` | a multisample image | §4.4.2.1 |
| `glGetInternalformativ(RENDERBUFFER, format, NUM_SAMPLE_COUNTS / SAMPLES, ...)` | which sample counts a format supports | §6.1.15 |
| `glGetRenderbufferParameteriv(SAMPLES)`, `glGetIntegerv(SAMPLES / SAMPLE_BUFFERS)` | what was actually allocated | §4.4.2.1, §4.4.4.2 |

## Key ideas

- **A destination pixel is written when its centre lies in the destination rectangle**, and it receives the source
  at the point its centre maps to. Everything else follows from this one rule (the CPU model in `main.cpp` is exactly
  this): an unscaled copy moves pixels, 2x NEAREST repeats each source pixel, 3:1 NEAREST picks the central
  texel, `x1 < x0` mirrors, and a destination pixel whose source point lies outside the source buffer is not
  written ("clipping"; the scale is computed as though nothing were clipped). The demo never uses 2:1 NEAREST
  downscaling in a check: the destination centres then fall exactly on source pixel borders, where the spec leaves
  the choice open.
- **Both rectangles reversed means no reversal.** Swapped coordinates mirror only when exactly one of the two
  rectangles is swapped in that direction.
- **`LINEAR` is for color only.** With `DEPTH_BUFFER_BIT` or `STENCIL_BUFFER_BIT` in the mask the call is
  `INVALID_OPERATION` and *nothing* is copied (the color part of the same call is not copied either). `LINEAR`
  on integer buffers is also `INVALID_OPERATION`.
- **A blit bypasses the fragment pipeline.** The color, depth and stencil write masks and the depth and stencil
  tests are ignored; the scissor test is not. The demo masks all color writes, or enables `DEPTH_TEST` with `NEVER`,
  and the blit still copies everything.
- **Missing buffers are skipped silently.** A depth or stencil bit for a framebuffer without that buffer is
  ignored, the rest of the mask is still copied.
- **Format conversion.** RGBA8 to RGB565 quantizes ("one of the two integers closest", equation 2.3), but fixed-point
  to integer, unsigned to signed and signed to unsigned are `INVALID_OPERATION`. Depth/stencil formats must match.
- **Resolve.** If the read framebuffer is multisample (`SAMPLE_BUFFERS = 1`) and the draw framebuffer is not, the samples
  of each pixel are combined into one; the filter argument is ignored. ES 3.0 allows it only when the formats are
  identical **and the source and destination rectangles are the same rectangle**: scaling, translating or mirroring a
  resolve is `INVALID_OPERATION`. A multisample *draw* framebuffer is always an error, and so is `glReadPixels` from a
  multisample framebuffer, so a resolve blit is the only way out.
- **Sample counts are discovered, not assumed.** `NUM_SAMPLE_COUNTS` and `SAMPLES` list the counts of a format in
  descending order, `bufSize` limits how many values are written, and the largest is at least `MAX_SAMPLES`. A request
  gets at least the requested count and at most the next larger supported one, so asking for the largest gets exactly
  the largest. (ES 3.0 gives integer formats no multisample support at all: count 0 and `INVALID_OPERATION`. ES 3.1
  allows them; the demo asserts the rule of the context's version.)
- **The resolve of a normalized color is implementation-dependent** (§4.3.3). The demo therefore does not assert a sample
  pattern; it asserts what every resolve must satisfy (see Validation).

## Shader concepts

`chart.frag` writes three outputs with `layout(location = N)` (three attachments in one pass) from `gl_FragCoord` and
integer shifts and masks; `quad.vert` builds a rectangle from `gl_VertexID` and uniforms; `tri.vert` indexes a
uniform array `u_tri[3]` with `gl_VertexID`. All fragment shaders declare `precision highp float`.

## Validation (`--validate`)

215 checks, all against CPU models or defined errors; each blit check compares the *whole* destination, so pixels
that must not change are checked as well. All images read back with `glReadPixels` as `RGBA/UNSIGNED_BYTE`.

- **Color blits** (14 cases): the generic model of the blit rule is applied to the destination's previous contents.
  NEAREST cases are compared exactly. LINEAR cases use +-2: the spec gives no precision for the filter weights; an
  8-bit weight is off by at most 1/256 of a channel difference (153 for the checkerboard, 0.6) plus the rounding of the
  8-bit result. Observed: 0 (radeonsi, llvmpipe) and 1 (zink). Each case first checks that the model predicts written
  pixels, so an empty region cannot pass.
- **Separate bindings:** the binding queries; `glReadPixels` returns a chart pixel while `glClear` turns the *draw*
  framebuffer magenta and leaves the chart untouched.
- **`glReadBuffer`:** with draw buffers `{C0, NONE, C2}` and read buffers `C1`, then `C2`, destinations C0 and C2 equal
  the selected chart and C1 keeps its sentinel; `READ_BUFFER` is per framebuffer object.
- **Scissor and masks:** the scissor box limits a 2x blit to exactly that box; `glColorMask(FALSE x4)` has no effect on
  the blit. Mask 0 copies nothing and raises no error.
- **Conversion and integers:** RGBA8 to RGB565 gives for every channel one of the two 5/6/5-bit levels next to the
  source value; RGBA8UI to RGBA8UI copies exactly; LINEAR on integer data and the four mismatches (unsigned to signed,
  signed to unsigned, fixed-point to unsigned, unsigned to fixed-point) raise `INVALID_OPERATION` and change nothing.
- **Depth and stencil:** a scene with depth 0.25 / 0.75 / 0.9 / 1.0 and stencil 1 / 2 / 3 / 0 is blitted 1:1, 2x, mirrored,
  as depth only, as stencil only, and while masks and failing tests are active. Depth and stencil cannot be read back
  in ES, so each result is probed: draws at window depth 0.5 and 0.8 with `DEPTH_TEST LESS` pass where the stored depth is
  greater, draws with `STENCIL_TEST EQUAL k` pass where the stored stencil is k; the passes are read back and compared
  per pixel with the model. LINEAR with depth or stencil, `DEPTH24_STENCIL8` to `DEPTH_COMPONENT16`: `INVALID_OPERATION`
  with no copy; a stencil bit for a destination without stencil and depth/stencil bits for a color-only destination are
  silently ignored while the color is copied.
- **glGetInternalformativ:** 15 renderbuffer formats: `NUM_SAMPLE_COUNTS >= 1`, `SAMPLES` strictly descending and
  positive, largest `>= MAX_SAMPLES`; `bufSize` 0, 1 and larger write exactly what they should; negative `bufSize` is
  `INVALID_VALUE`, `RGB9_E5` is `INVALID_ENUM`. Every reported count of RGBA8 is allocated and `RENDERBUFFER_SAMPLES` lies
  between the request and the next larger count; the largest count of five formats gives exactly that many samples.
- **Multisample resolve:** `SAMPLES` / `SAMPLE_BUFFERS` of the draw framebuffer. A triangle with corners at fractional
  pixel positions is rendered with the largest count (8 on radeonsi, 4 on llvmpipe and zink) and resolved. Every pixel is
  classified geometrically (square inside, square outside, or crossing an edge). Interior pixels equal the triangle
  color and exterior pixels the background (tolerance 1, equation 2.3). Every edge pixel lies in the closed range between
  the two colors in all channels. Edge pixels whose centre is within 0.1 pixel of an edge line and 3 pixels from
  a corner (70 of them) are *strictly* between: an edge splits such a pixel into two nearly equal parts, so any
  sample pattern spread around the pixel centre has samples on both sides (the spec does not promise this, which is why
  only these pixels are asserted). The interior count plus the sum of the resolved coverage of the edge pixels equals the
  triangle's area within 0.5% (observed 0.01%; the per-pixel errors cancel). `LINEAR` resolves identically to `NEAREST`; a
  sub-rectangle with identical bounds resolves just that rectangle. A multisample depth+stencil+color scene resolves
  to the scene. **Negative control:** the same triangle in a single-sample framebuffer has exact interior and exterior pixels
  and *no* blended pixel, so the strictly-between check rejects it.
- **Errors:** resolve with scaling, with translation and with mirroring; multisample to RGB565 and to SRGB8_ALPHA8;
  multisample to multisample; single-sample to multisample; read and draw the same framebuffer; incomplete draw and read
  framebuffers (`INVALID_FRAMEBUFFER_OPERATION`); `glReadPixels` from a multisample framebuffer. Rejected resolves are
  shown to have copied nothing.
- **Negative controls:** the NEAREST and LINEAR models differ in 16380 of 16384 pixels; a mirrored result differs from the
  unmirrored one; the scissored blit differs from the unscissored model; the converted copy differs from the chart; a
  cleared depth/stencil buffer probes differently from a blitted one; single-sample versus multisample (above).

## Known driver deviations and limits of the evidence (Mesa 25.0.7: radeonsi, llvmpipe, zink)

- **`RGBA8` to `SRGB8_ALPHA8` multisample resolve raises no error.** ES 3.0 §4.3.3 (and ES 3.2 §16.2.1): "an INVALID_OPERATION
  error is generated if the formats of the read and draw framebuffers are not identical"; the formats differ, but all three
  drivers resolve. The check keeps the strict expectation and is reported as XFAIL on these drivers
  (`Check::expect_or_known_deviation`).
- `es-min` runs use `MESA_GLES_VERSION_OVERRIDE`, which only changes the reported version, but the driver *does*
  report the ES 3.0 integer-format sample counts (0) there, so the ES 3.0 branch of the integer checks is exercised.
- Not reachable through meta-gl's typed enums, hence not checked: `INVALID_VALUE` for mask bits other than the three buffer
  bits and `INVALID_ENUM` for a filter other than `LINEAR` / `NEAREST` (a raw cast would be needed).
- A multisample *resolve* result is implementation-dependent by the spec; the checks describe properties of any averaging resolve and
  are not a proof for a driver that resolves some other way (for example by picking one sample).

## Pitfalls

- A blit is not a draw: it ignores write masks and tests but obeys the scissor test. A forgotten `SCISSOR_TEST` clips it.
- Blit coordinates are pixel *borders*: a 64-pixel rectangle is `(0, 0, 64, 64)`, not `(0, 0, 63, 63)`.
- `READ_BUFFER` and `DRAW_BUFFERi` are per-framebuffer state; select them for the framebuffer that is bound to the
  corresponding target.
- A multisample renderbuffer cannot be sampled or read directly. Resolve it first, with identical bounds and format.
- A resolve needs *identical* formats on both sides (§4.3.3), which includes the default framebuffer's; resolving into an
  intermediate texture of the same format (as here) and blitting that to the window avoids the question.
- Depth and stencil buffers cannot be read back in ES: test them by drawing against them.
