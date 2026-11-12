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

The window is four quarters (split at `width / 2` and `height / 2`, so they fill it at any size), each one a
`glBlitFramebuffer` that stretches an offscreen image onto the quarter of the default framebuffer:

```
 top-left:     64x64 chart, NEAREST            top-right:    the same chart, LINEAR
 bottom-left:  copied depth/stencil (128x128),  bottom-right: a triangle rendered with 4-8 samples
               shown through tested draws,                    and resolved by a blit (128x128),
               NEAREST                                        NEAREST
```

Only these four display blits depend on the window size; every other blit of the demo works between offscreen targets
of fixed size (64, 128, 32 pixels), so the pixel-exact experiments are the same at every window size. In a 256 x 256
window the quarters are 128 x 128, which makes the top quarters exact 2x stretches and the bottom ones 1:1 copies. At
other sizes the stretch factor is not an integer: a destination pixel centre `d + 0.5` maps to `(d + 0.5) * src / n` in
the source, and where that lands within 0.05 pixel of a source pixel edge (e.g. 64 -> 160 pixels gives exact edges) the
implementation's rounding decides, so the NEAREST checks accept either neighbouring source pixel there and require an
exact match everywhere else. The LINEAR quarter is compared with the bilinear model at the quarter's own scale. The
smallest window is 64 x 64; below that the demo reports SKIP.

The chart is a 64 x 64 image in which red grows by 4 per pixel in x, green by 4 per pixel in y (so every pixel is
unique) and blue is an 8 x 8 checkerboard. The NEAREST panel is blocky; the LINEAR panel is smooth between the
texel centers but keeps the checkerboard edges soft. In the bottom-left panel, a depth+stencil scene of four regions
was copied 2x into a 128 x 128 target by one blit and then painted over by four draws that each pass only where a test passes
on the *copied* buffers: white where the copied depth is greater than 0.5, cyan where the copied stencil is 2,
gray where the depth is greater than 0.8, black where the stencil is 3. What remains of the original red is the
region whose copied depth (0.25) failed every depth test, so the picture is a direct view of the copied depth and
stencil values. In the bottom-right panel the triangle's edges are not stair-stepped: on all three Mesa drivers the pixels
on the edge are blends of the two colors (what a resolve makes of the samples is up to the implementation, see below).

## Central calls

| Call | Role | Spec |
|---|---|---|
| `glBindFramebuffer(READ_FRAMEBUFFER, a)` / `(DRAW_FRAMEBUFFER, b)` | source and destination of a blit; `glReadPixels` uses READ, `glClear` and draws use DRAW | §4.4.1 |
| `glReadBuffer(COLOR_ATTACHMENTi)` | which attachment of the read framebuffer is copied | §4.3.1 |
| `glBlitFramebuffer(sx0,sy0,sx1,sy1, dx0,dy0,dx1,dy1, mask, filter)` | the copy | §4.3.3 |
| `glRenderbufferStorageMultisample(RENDERBUFFER, samples, format, w, h)` | a multisample image | §4.4.2.1 |
| `glGetInternalformativ(RENDERBUFFER, format, NUM_SAMPLE_COUNTS / SAMPLES, ...)` | which sample counts a format supports | §6.1.15 |
| `glGetRenderbufferParameteriv(SAMPLES)`, `glGetIntegerv(SAMPLES / SAMPLE_BUFFERS)` | what was actually allocated | §4.4.2.1, §4.4.4.2 |
| `glTexStorage2DMultisample(TEXTURE_2D_MULTISAMPLE, n, RGBA8UI, ...)` + `texelFetch(usampler2DMS, p, s)` | ES 3.1+ contexts only: an integer multisample image whose samples can be read one by one | ES 3.2 §8.8, §11.1.3.3 |

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
- **What a resolve guarantees, and what it does not.** §4.3.3 (ES 3.2 §16.2.1) defines the resolve by format:
  > If the source formats are integer types or stencil values, a single sample's value is selected for each pixel. If the
  > source formats are floating-point or normalized types, the sample values for each pixel are resolved in an
  > implementation-dependent manner. If the source formats are depth values, sample values are resolved in an
  > implementation-dependent manner where the result will be between the minimum and maximum depth values in the pixel.

  So an integer pixel is one of its samples, a stencil pixel likewise, a depth pixel lies between its smallest and largest
  sample. An RGBA8 pixel can be *anything*: the specification does not say it lies between the sample colors, that it is
  their average, or even that a pixel whose samples all hold one color resolves to that color. (The text for the window,
  §4.1.10, is the same: "The method of combination is not specified"; an average is only *recommended*.) The demo asserts
  the first three rules and only prints what it measures for the RGBA8 resolve.

## Shader concepts

`chart.frag` writes three outputs with `layout(location = N)` (three attachments in one pass) from `gl_FragCoord` and
integer shifts and masks; `quad.vert` builds a rectangle from `gl_VertexID` and uniforms; `tri.vert` indexes a
uniform array `u_tri[3]` with `gl_VertexID`. All fragment shaders declare `precision highp float`.
`flat_uint.frag` writes a `uvec4` output into the RGBA8UI multisample texture; integer outputs are stored without
conversion. `fetch_ms.vert` / `fetch_ms.frag` are `#version 310 es` and are compiled only in ES 3.1+ contexts: they read
one sample of that texture per pass with `texelFetch(usampler2DMS, ivec2(gl_FragCoord.xy), u_sample)` (GLSL ES 3.10
§8.9.2). Multisample samplers have no default precision, hence `highp usampler2DMS` (GLSL ES 3.10 §4.7.4). Both shaders
of that program are 3.10: "It is an error to link a vertex shader and a fragment shader if they are written in different
versions of the language" (GLSL ES 3.10 §1.1).

## Validation (`--validate`)

218 checks in an ES 3.1+ context, 209 in an ES 3.0 context (the integer resolve needs ES 3.1), all against CPU models,
per-sample readback or defined errors; each blit check compares the *whole* destination, so pixels
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
- **Multisample resolve, what is asserted.** `SAMPLES` / `SAMPLE_BUFFERS` of the draw framebuffer and `RENDERBUFFER_SAMPLES`.
  Rules that hold for *any* resolve:
  - `LINEAR` resolves identically to `NEAREST` ("The filter parameter is ignored"; Appendix A.1 repeatability makes the
    same command on the same samples give the same result);
  - a sub-rectangle with identical bounds leaves every pixel outside the rectangle untouched ("Only pixels whose centers lie
    within the destination rectangle are written by BlitFramebuffer"), checked exactly on the whole RGBA8 target. Inside the
    rectangle the RGBA8 values are implementation-dependent like every RGBA8 resolve, and nothing requires a sub-rectangle
    resolve to reproduce the whole-image resolve bit for bit (it is a different command, and an implementation may take
    another resolve path for it), so how many inside pixels equal the whole-image resolve is printed, not asserted. The
    inside is asserted with integer samples instead (next item but one);
  - **depth and stencil**: the four-region scene is written into the multisample depth/stencil buffer by *scissored clears*.
    The scissor test selects whole pixels and "masking and scissoring affect clearing the multisample buffer in the same way
    as they affect clearing the corresponding color, depth, and stencil buffers" (§4.2.3.1), so all samples of a pixel hold
    one value whatever the sample locations are. Stencil ("a single sample's value is selected") and depth ("between the
    minimum and maximum depth values in the pixel") must then resolve to exactly that value; probed with tested draws.
    Negative control: the resolved buffer differs from a cleared one in over 1500 probed pixels;
  - **integer color (ES 3.1+ contexts)**: the triangle is drawn into an RGBA8UI multisample *texture* (`glTexStorage2DMultisample`,
    the largest count `glGetInternalformativ(TEXTURE_2D_MULTISAMPLE, RGBA8UI, SAMPLES)` reports). Every stored sample is read
    back, one pass per sample index, with `texelFetch(usampler2DMS, ...)`; no resolve is involved. Every sample must be exactly
    the cleared background or the triangle value, and both must occur. The blit resolve into an RGBA8UI texture must give,
    in **every pixel, the value of one of that pixel's own samples**. Negative control: the resolved values shifted by one are
    rejected in every pixel. The same is repeated for a sub-rectangle resolve (20,30)-(100,90) into a target cleared to a
    sentinel: every pixel inside is one of its own samples and every pixel outside keeps the sentinel. (A per-channel average would also be rejected in each pixel whose samples differ: 301 of 301 on
    radeonsi with 8 samples, 244 of 244 on llvmpipe and zink with 4; printed, since how many such pixels exist depends on
    the sample locations.) In an ES 3.0 context this part is skipped with a note: ES 3.0 has no multisample integer images
    (`RenderbufferStorageMultisample` with an integer format is `INVALID_OPERATION`, §4.4.2.1).
  - The single-sample render of the triangle is fully specified (point sampling at pixel centres, §3.6.1): interior pixels
    equal the triangle color, exterior pixels the background (tolerance 1, equation 2.3), and no pixel is a blend.
- **Multisample resolve, what is only observed.** The RGBA8 resolve of the triangle (largest count: 8 on radeonsi, 4 on llvmpipe
  and zink) is printed as a `note:` line and never asserted: interior pixels equal to the triangle color (all of them on the
  three drivers), exterior pixels equal to the background (all), edge pixels inside the range of the two colors (all), edge
  pixels strictly between them (301 / 244, including all 70 pixels centred on an edge), and the summed coverage against the
  triangle's area (-0.015 % on radeonsi, +0.002 % on llvmpipe and zink). Likewise the resolved RGBA8 color of the uniformly
  cleared depth/stencil scene (exact on all three drivers), and which sample index the integer resolve picks (always sample 0
  on all three).
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
- The RGBA8 resolve is implementation-dependent (§4.3.3), so no RGBA8 resolved value is asserted, not even for pixels whose samples
  all hold one color. A conformant driver that resolves normalized colors by some other rule passes. What the demo does assert
  about resolves (filter ignored, nothing written outside an identical-bounds sub-rectangle, uniform depth/stencil, integer
  resolve = one of the samples, also for a sub-rectangle) follows from the quoted text alone.

## Known driver deviation: ANGLE

- **A multisample resolve of a sub-rectangle writes outside the destination rectangle.** ANGLE 2.1.17841 (Vulkan backend,
  RADV) resolves (20,30)-(100,90) as if the rectangle reached the far edges of the 128 x 128 framebuffer: all 5784 pixels of
  (20,30)-(128,128) outside the rectangle are written, with resolved values (RGBA8 and RGBA8UI alike). ES 3.0 §4.3.3: *"Only
  pixels whose centers lie within the destination rectangle are written by BlitFramebuffer."* Both "outside" checks are
  reported as XFAIL on "ANGLE"; the inside of the integer sub-rectangle resolve conforms. The original check also required the
  RGBA8 inside to equal the whole-image resolve; that asserted more than §4.3.3 guarantees and is now an observation (on ANGLE
  4758 of 4800 inside pixels are equal, on the Mesa drivers all of them).

## Pitfalls

- A blit is not a draw: it ignores write masks and tests but obeys the scissor test. A forgotten `SCISSOR_TEST` clips it.
- Blit coordinates are pixel *borders*: a 64-pixel rectangle is `(0, 0, 64, 64)`, not `(0, 0, 63, 63)`.
- `READ_BUFFER` and `DRAW_BUFFERi` are per-framebuffer state; select them for the framebuffer that is bound to the
  corresponding target.
- A multisample renderbuffer cannot be sampled or read directly. Resolve it first, with identical bounds and format. To see the
  individual samples, render into a multisample *texture* (ES 3.1) and `texelFetch` it.
- Do not build exact expectations on a resolved RGBA8 image, not even where all samples agree: the specification leaves the
  combination to the implementation. Integer formats are the exception (one of the samples is selected).
- A resolve needs *identical* formats on both sides (§4.3.3), which includes the default framebuffer's; resolving into an
  intermediate texture of the same format (as here) and blitting that to the window avoids the question.
- Depth and stencil buffers cannot be read back in ES: test them by drawing against them.
