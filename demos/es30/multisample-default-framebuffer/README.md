# es30-multisample-default-framebuffer

**API introduced in:** OpenGL ES 2.0 (multisampled window surfaces), ES 3.0 (`glBlitFramebuffer` resolve) ·
**Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 ·
**Run:** `es30-multisample-default-framebuffer [--headless --validate] [--width W --height H] [--frames N]`

## Purpose

Most multisampling examples render into a multisample *renderbuffer* and resolve it with a blit. The older and
simpler path is a multisampled **default framebuffer**: you ask the window system (EGL, SDL) for a config with
`EGL_SAMPLES = 4`, and from then on every draw into the window is antialiased without any further GL call. This
demo shows that path and what the ES 3.0 specification says about it:

- `SAMPLE_BUFFERS` and `SAMPLES` describe the framebuffer that is bound for drawing; for the default framebuffer
  they come from the EGL config;
- the window's color buffer holds **resolved** pixels: the resolve "occurs each time a pixel is updated" (§3.3);
- `glReadPixels` **may** read the multisampled default framebuffer, whereas reading a multisample framebuffer
  *object* is an error (§4.3.2);
- `glBlitFramebuffer` can resolve the default framebuffer into a single-sample framebuffer object, under the
  strict resolve rules of §4.3.3 (identical rectangles, identical formats); the default framebuffer cannot be the
  *destination* of a blit while it is multisampled.

The runtime requests the config by setting `demo::Info::samples = 4`. The demo itself never calls an EGL
function.

## What the image shows

A cube rotating about a tilted axis on a dark blue background (`(20, 24, 40)`), drawn with back-face culling.
Every face has its own flat color, and the six colors are chosen so that **R + G + B = 450 for every face**,
against 84 for the background. With the default `--frames 1` the cube stands still at its first pose (three
faces visible: violet top, green left, cyan right). The cube's size follows the smaller side of the surface, so it
fills a square window and stays centred in a tall or wide one.

Viewed closely, the silhouette and the seams between faces are not stair-stepped: the pixels along them are
blends of the background and face colors. Away from the edges every pixel is exactly a face color or the
background.

## Central calls

| Call | Role | Spec |
|---|---|---|
| (EGL config with `EGL_SAMPLE_BUFFERS = 1`, `EGL_SAMPLES = 4`) | the default framebuffer gets a multisample buffer | §3.3 |
| `glGetIntegerv(SAMPLE_BUFFERS / SAMPLES)` | 1 / 4 for the default framebuffer, 0 / 0 for a single-sample FBO | §3.3, §4.4.4.2 |
| `glGetFramebufferAttachmentParameteriv(DRAW_FRAMEBUFFER, BACK, ...)` | the back buffer is RGBA8 (sizes, component type, encoding) | §6.1.13 |
| `glGetMultisamplefv(SAMPLE_POSITION, i)` | sample positions, ES 3.1+ contexts only | ES 3.2 §13.4 |
| `glReadPixels` with `READ_FRAMEBUFFER_BINDING` 0 | reads resolved pixels; no error | §4.3.2 |
| `glBlitFramebuffer(0,0,w,h, 0,0,w,h, COLOR_BUFFER_BIT, NEAREST)` | resolves the default framebuffer into an FBO | §4.3.3 |
| `glEnable(CULL_FACE)`, `glCullFace(BACK)`, `glFrontFace(CCW)` | the hidden faces of the convex cube are removed | §3.6.1 |

## Key ideas

- **Multisample rasterization.** With `SAMPLE_BUFFERS = 1`, a polygon produces a fragment for every pixel with at
  least one covered sample point, and the fragment carries a coverage bit per sample (§3.6.3). Depth/stencil tests
  and blending run per sample (§4.1.10). Then "the sample values for each color in the multisample buffer are
  combined to produce a single color value, and that value is written into the corresponding color buffer".
- **Reading is allowed.** ES 3.0 §4.3.2: "ReadPixels generates an INVALID_OPERATION error if
  READ_FRAMEBUFFER_BINDING is non-zero, the read framebuffer is framebuffer complete, and the value of
  SAMPLE_BUFFERS for the read framebuffer is one." The condition names framebuffer *objects* only. For the
  default framebuffer, `glReadPixels` reads the color buffer, which holds the resolved values.
- **Blit resolve rules.** ES 3.0 §4.3.3: when the read framebuffer has `SAMPLE_BUFFERS` one and the draw
  framebuffer zero, the samples "are converted to a single sample before being written", and the filter is
  ignored. "If SAMPLE_BUFFERS for the read framebuffer is one, no copy is performed and an INVALID_OPERATION
  error is generated if the formats of the read and draw framebuffers are not identical or if the source and
  destination rectangles are not defined with the same (X0, Y0) and (X1, Y1) bounds." And "if SAMPLE_BUFFERS
  for the draw framebuffer is greater than zero, an INVALID_OPERATION error is generated", which also applies
  when the draw framebuffer is the multisampled window.
- **Both rectangles mirrored.** Swapping X0 and X1 in *both* rectangles keeps the bounds identical, so this
  resolve is legal. "If both the source and destination rectangles specify a negative width or height for the same
  direction, no reversal is performed", so the result is not mirrored.
- **Equal channel sums.** A blend of face colors keeps R + G + B = 450, and a blend with the background lies
  between 84 and 450. A resolved pixel's channel sum therefore tells how much of it the cube covers, whatever
  the faces are: `(sum - 84) / 366`.

## Shader concepts

`cube.vert` transforms the 36 vertices (6 faces x 2 triangles, no index buffer) by a CPU-computed
`u_mvp` and passes the face color as a `flat` output. With `flat`, every fragment and every sample of a face gets
the provoking vertex's color unchanged. No interpolated value can drift by a rounding step, so interior pixels
are exactly the face color. `cube.frag` writes it out at `highp`. The colors are `k / 255`, which an 8-bit buffer
stores exactly. Dithering is disabled for clarity, although it could not change exactly representable values.

## Validation (`--validate`)

The CPU projects the 8 cube corners with the same `mvp` matrix, keeps the faces that are counter-clockwise in
window coordinates (the front faces), and takes the convex hull of the corners as the silhouette. Each pixel's
square, **grown by half a pixel** on every side, is then classified. The margin is needed because sample points
may lie outside the pixel's unit square (§3.3) and vertices are snapped to the subpixel grid. The classes are:

| class | where | expectation (8-bit, +-1 per channel) |
|---|---|---|
| exterior | grown square misses the silhouette | the background exactly |
| face | grown square inside one front face | that face's color exactly |
| seam | grown square inside the silhouette, touching 2-3 faces | R + G + B = 450 +-3, channels between those faces' colors |
| silhouette | the rest | channels between the background and the touching faces, sum in [81, 453] |

A seam pixel's sum of 450 relies on the shared-edge rule of §3.6.1 (and corollary 8 of Appendix B, "no pixel dropouts or duplicates"): each
sample on an edge shared by two faces is produced by exactly one of them. A crack would let background into the
sum, and a double hit cannot be seen with opaque colors, but a crack can. The sum tolerance is 3 because each
channel's conversion to 8 bits errs by less than one step.

The checks:

1. **Queries.** With the default framebuffer bound, `SAMPLE_BUFFERS = 1` and `SAMPLES >= 1` (4 on radeonsi). With a
   single-sample RGBA8 FBO as the draw framebuffer, both are 0. With a 4-sample renderbuffer FBO, 1 and >= 4.
   After rebinding 0, the default framebuffer's values return. The `BACK` attachment reports 8/8/8/8 bits,
   `UNSIGNED_NORMALIZED`, `LINEAR`, which is the format the RGBA8 resolve target must match exactly. In ES 3.1+
   contexts, `glGetMultisamplefv(SAMPLE_POSITION, i)` lies in [0, 1] for every i < SAMPLES ("x and y each lie in
   the range [0, 1]"), and i = SAMPLES is `INVALID_VALUE`. In an ES 3.0 context the call is ES 3.1 API and is not
   made.
2. **`glReadPixels` from the default framebuffer** raises no error, and `READ_FRAMEBUFFER_BINDING` is 0 and
   `READ_BUFFER` is `BACK`. Every pixel of the image passes its class check. Pixels whose centre lies within 0.1
   pixel of the silhouette, at least 3 pixels from any corner, must be **strictly** between background and cube
   (sum in (87, 447)): the edge halves the pixel, so samples lie on both sides. The **summed coverage** (face and
   seam pixels count 1, silhouette pixels `(sum - 84) / 366`) must equal the silhouette's area within 0.05 pixel
   per pixel of perimeter. That bound is a sanity tolerance, not a spec number. The measured value is printed:
   within 0.02 % on radeonsi.
3. **Negative control.** The same cube drawn into a single-sample RGBA8 FBO has identical exterior and face
   pixels, but **no** blended silhouette pixel, and the centred-pixel check rejects every one of its centred
   pixels. The multisampled image has hundreds. So the blend checks can fail and do detect multisampling.
4. **Blit resolve.** The FBO is first filled with a magenta sentinel. A blit of the whole surface with identical
   rectangles raises no error and passes the same class checks. Its exterior and face pixels equal the
   `glReadPixels` image exactly. A sub-rectangle with identical bounds writes exactly that rectangle: outside it
   the sentinel is kept, inside it the pixels are valid resolves. The blit with both rectangles mirrored in x
   raises no error and is not mirrored (the cube is not symmetric, so a mirrored image would fail).
5. **Errors (§4.3.3, §4.3.2).** These must raise `INVALID_OPERATION`:
   - a resolve to a same-size rectangle at another position;
   - a resolve with 2x scaling;
   - a resolve with only the destination mirrored;
   - a resolve into an RGB565 FBO (formats not identical).

   The rejected resolves leave the sentinel untouched. A blit *into* the multisampled default framebuffer is
   `INVALID_OPERATION`, and the window still reads back identically afterwards. `glReadPixels` from a complete
   4-sample FBO is `INVALID_OPERATION`, and the same call on the default framebuffer is not.

**Not asserted, only printed.** For edge pixels, the spec does not say that the blit resolve and the window's own
resolve combine the samples the same way: both are "implementation-dependent". The demo compares the two images
only where all samples of a pixel are equal, and prints how many edge pixels differ (0 on radeonsi). The class
checks also assume that a resolve is a weighted average of the samples. ES 3.0 says only that samples are
"combined" or "resolved in an implementation-dependent manner".

## Environment and results

| driver | outcome |
|---|---|
| gpu (radeonsi) | PASS at 256x256, 800x800, 640x480, 333x777 and 97x501, in ES 3.2 and ES 3.0 (override) contexts, across many rotation poses (`--frames N`). |
| llvmpipe, zink | BLOCKED_ENVIRONMENT. Mesa creates a 0x0 drawable for a multisampled surfaceless pbuffer, so the runtime refuses to run the demo. See `docs/KNOWN_LIMITATIONS.md`, row "environment (runtime check)". |

Surfaces smaller than 96 x 96 report `SKIP_UNSUPPORTED`: the edge statistics need enough silhouette pixels.

## Spec references

- ES 3.0 §3.3 Multisampling (multisample buffer of the window-system framebuffer, SAMPLE_BUFFERS, SAMPLES,
  sample points possibly outside the pixel)
- ES 3.0 §3.6.1 Basic Polygon Rasterization (point sampling, shared edges, face culling), §3.6.3 Polygon
  Multisample Rasterization
- ES 3.0 §4.1.10 Additional Multisample Fragment Operations (per-sample operations, combination into the color
  buffer)
- ES 3.0 §4.3.2 Reading Pixels (INVALID_OPERATION only for a multisample framebuffer *object*)
- ES 3.0 §4.3.3 Copying Pixels (resolve, identical bounds and formats, multisample draw framebuffer, mirroring)
- ES 3.0 §4.4.4.2 Whole Framebuffer Completeness (SAMPLE_BUFFERS and SAMPLES of framebuffer objects), §4.4.2.1
  (RENDERBUFFER_SAMPLES >= requested)
- ES 3.0 §6.1.13 Framebuffer Object Queries (default framebuffer `BACK` attachment)
- ES 3.2 §13.4 Multisampling (`glGetMultisamplefv`, positions in [0, 1], INVALID_VALUE for index >= SAMPLES)
- ES 3.0 Appendix B, corollary 8 (no pixel dropouts or duplicates along shared edges)
- GLSL ES 3.00 §4.3.9 Interpolation (`flat`)

## Pitfalls

- **The window's sample count is chosen at context creation.** No GL call can turn multisampling of the default
  framebuffer on or off. If the EGL config has no samples, `SAMPLE_BUFFERS` is 0 and every edge is aliased.
- **Do not resolve the window into a mismatched target.** The draw framebuffer of a resolve must have *exactly*
  the window's format. An RGBA8 window into RGB565 or RGBA4 fails, and so does any offset, scale or one-sided
  mirror. Query the `BACK` attachment's sizes before you choose the target format.
- **You cannot blit into a multisampled window.** Content made in an FBO has to be drawn into the window as a
  textured quad, or the window has to be single-sampled.
- **`glReadPixels` behaves differently on FBOs.** The same call that works on the multisampled window fails on a
  multisample FBO: resolve the FBO with a blit first.
- **Edge values are implementation-dependent.** Sample positions, the number of samples and the resolve weights
  are all up to the implementation. Test edge pixels with bounds and statistics, never against exact values.
- **Software Mesa drivers** (llvmpipe, zink on lavapipe) cannot create a multisampled surfaceless pbuffer of the
  requested size. Use multisample renderbuffers there.
