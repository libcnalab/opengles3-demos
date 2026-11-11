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

## Read this first

The technique itself is four short steps in `main.cpp`:

1. **Ask for samples.** `main()` passes `demo::Info{... .samples = kRequestedSamples}` (4); the runtime turns that into
   the EGL config request. Nothing else is needed to make window drawing multisampled.
2. **Check what you got.** `check_queries()` reads `SAMPLE_BUFFERS` and `SAMPLES` with the default framebuffer bound,
   then with single-sample and multisample framebuffer objects bound, to show that both values follow the draw
   framebuffer.
3. **Draw normally.** `render()` calls `draw_cube()`: one `glDrawArrays` of 36 vertices with back-face culling, no
   multisample-specific call.
4. **Get resolved pixels.** In `validate()`, `read_image()` does a `glReadPixels` straight from the window. Further down,
   after `bind_read(kDefaultFbo)` and `bind_draw(single_)`, `blit(all, all)` (a thin wrapper around `glBlitFramebuffer`)
   resolves the whole window into a single-sample RGBA8 framebuffer object.

The rest of this README, and most of `validate()` (stencil sample counting, the single-sample control, the error
cases), is reference material on what ES 3.0 does and does not promise about multisampling; read it once the four
steps are clear.

## What the image shows

A cube rotating about a tilted axis on a dark blue background (`(20, 24, 40)`), drawn with back-face culling.
Every face has its own flat color, and the six colors are chosen so that **R + G + B = 450 for every face**,
against 84 for the background. With the default `--frames 1` the cube stands still at its first pose (three
faces visible: violet top, green left, cyan right). The cube's size follows the smaller side of the surface, so it
fills a square window and stays centred in a tall or wide one.

Viewed closely, the silhouette and the seams between faces are not stair-stepped: on radeonsi the pixels along them
are blends of the background and face colors, and away from the edges every pixel is exactly a face color or the
background. That is what an averaging resolve produces; the specification leaves the combination of the samples to
the implementation (see Validation).

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
  between 84 and 450. If the resolve averages the samples, a resolved pixel's channel sum tells how much of it the
  cube covers, whatever the faces are: `(sum - 84) / 366`. The demo uses this for its printed observations only.
- **Per-sample facts without a resolve.** The window's samples cannot be read, but they can be *counted*: stencil
  `INCR` runs per sample, and an `ANY_SAMPLES_PASSED` query reports whether any sample passes a stencil test.

## Shader concepts

`cube.vert` transforms the 36 vertices (6 faces x 2 triangles, no index buffer) by a CPU-computed
`u_mvp` and passes the face color as a `flat` output. With `flat`, every fragment and every sample of a face gets
the provoking vertex's color unchanged. No interpolated value can drift by a rounding step, so interior pixels
are exactly the face color. `cube.frag` writes it out at `highp`. The colors are `k / 255`, which an 8-bit buffer
stores exactly. Dithering is disabled for clarity, although it could not change exactly representable values.

## Validation (`--validate`)

*Reference material: a conformance-level account of every check. Not needed to follow the technique.*

39 checks in an ES 3.1+ context, 36 in an ES 3.0 context (the `glGetMultisamplefv` checks are ES 3.1 API).

**What the specification does not promise.** Every pixel of the window, and every pixel a resolve blit writes, is a
combination of samples, and the specification leaves the combination open:

> ES 3.0 §4.1.10: "the sample values for each color in the multisample buffer are combined to produce a single color
> value, and that value is written into the corresponding color buffer [...] The method of combination is not specified.
> If the framebuffer contains sRGB values, then it is recommended that an average of sample values is computed in a
> linearized space [...]. Otherwise, a simple average computed independently for each color component is recommended."
>
> ES 3.0 §4.3.3: "If the source formats are floating-point or normalized types, the sample values for each pixel are
> resolved in an implementation-dependent manner."

An average is *recommended*, not required. A conformant implementation may return a resolved value outside the range of
the samples, and the text does not even say that a pixel whose samples all hold one color resolves to that color. So
**no value read from the multisampled window or from a resolve blit is asserted.** Everything the demo used to assert
about them (exact background and face pixels, seam sums of 450, bounded silhouette blends, strictly blended centred
pixels, coverage equal to the silhouette area) is now measured and printed as a `note: observed ...` line.

The hard checks are the ones that do not depend on how samples are combined:

1. **Queries.** With the default framebuffer bound, `SAMPLE_BUFFERS = 1` and `SAMPLES >= 1` (4 on radeonsi). With a
   single-sample RGBA8 FBO as the draw framebuffer, both are 0. With a 4-sample renderbuffer FBO, 1 and >= 4.
   After rebinding 0, the default framebuffer's values return. The `BACK` attachment reports 8/8/8/8 bits,
   `UNSIGNED_NORMALIZED`, `LINEAR`, which is the format the RGBA8 resolve target must match exactly. In ES 3.1+
   contexts, `glGetMultisamplefv(SAMPLE_POSITION, i)` lies in [0, 1] for every i < SAMPLES ("x and y each lie in
   the range [0, 1]"), and i = SAMPLES is `INVALID_VALUE`. In an ES 3.0 context the call is ES 3.1 API and is not
   made.
2. **`glReadPixels` from the default framebuffer** raises no error, `READ_FRAMEBUFFER_BINDING` is 0 and `READ_BUFFER`
   is `BACK`. The CPU projection yields exterior, single-face, seam and silhouette pixels (none of the checks below is
   vacuous).
3. **Per sample, on the window itself: stencil counting and occlusion queries.** The window's stencil buffer is part
   of its multisample buffer (§3.3), the stencil test and stencil operations run per sample (§4.1.10), and an
   `ANY_SAMPLES_PASSED` query is TRUE "if any fragment or sample passes the depth test" (§4.1.6). With color writes
   off, the depth test on with `ALWAYS` and no depth writes, the cube is drawn with stencil op `INCR`, so every sample
   counts the front faces that cover it. A polygon drawn with a stencil test inside a query then answers "does any of
   my samples hold such a count?" exactly, without any resolve. The probing polygons are the silhouette moved 0.75 px
   inwards ("inset") or outwards ("grown"); the margin absorbs vertex snapping (`SUBPIXEL_BITS` >= 4: at most 1/32 px)
   and CPU/GPU rounding of the projection. Because the cube and the probing polygon are rasterized at the *same*
   sample points, the checks hold wherever the sample points lie (§3.3: "inside or outside of the unit square";
   ES 3.2 §13.4: they "cannot be queried").
   - no sample is covered by two front faces (a full-screen query for stencil > 1 is FALSE): the shared-edge rule of
     §3.6.1, which §3.6.3 applies to every sample point;
   - no sample of the inset silhouette is uncovered (stencil = 0 is FALSE): no crack along the seams;
   - after zeroing the grown silhouette, no sample with a non-zero count remains: the cube covers nothing outside its
     silhouette;
   - **positive controls**, so that each query is shown to be able to return TRUE: covered samples exist; before the
     cube is drawn the inset silhouette query is TRUE; drawing the cube twice makes the "covered twice" query TRUE.
     Mutating the demo to probe the crack with the grown polygon, or to zero only the inset polygon, makes the crack
     and the outside check fail.
4. **Single-sample control, fully specified.** The same cube drawn into a single-sample RGBA8 FBO. There a pixel is
   produced when its centre lies in a polygon, and a centre on a shared edge is produced by exactly one of the two
   polygons (§3.6.1); with flat opaque colors every pixel is therefore exactly the background or exactly one face. The
   CPU classification (pixel square grown by half a pixel, as below) gives: exterior pixels equal the background,
   single-face pixels equal their face, every seam pixel is exactly one of the faces it touches (no crack, no blend), and
   every silhouette pixel is the background or exactly one face. Pixels within 1 px of a projected corner are left out
   of the seam and silhouette counts, because the shared-edge rule says nothing about a centre that falls exactly on a
   corner where several faces meet (19 of them at 256 x 256).
5. **Blit resolve.** The FBO is first filled with a magenta sentinel. A blit of the whole surface with identical
   rectangles raises no error; its result (`full`) is the reference for the next two blits. §4.3.3 says "the samples
   corresponding to each pixel location in the source are converted to a single sample before being written to the
   destination": a resolved pixel is a function of that pixel's samples, and Appendix A.1 (repeatability) makes it the
   same function each time, whatever that function is. So:
   - a sub-rectangle with identical bounds raises no error, keeps the sentinel outside the rectangle, and equals `full`
     inside it;
   - both rectangles mirrored in x raise no error and the result equals `full` ("If both the source and destination
     rectangles specify a negative width or height for the same direction, no reversal is performed"). The cube is
     not mirror-symmetric (its single-sample image differs from its mirror image in thousands of pixels), so a mirrored
     copy would be detected.
6. **Errors (§4.3.3, §4.3.2).** These must raise `INVALID_OPERATION`:
   - a resolve to a same-size rectangle at another position;
   - a resolve with 2x scaling;
   - a resolve with only the destination mirrored;
   - a resolve into an RGB565 FBO (formats not identical).

   The rejected resolves leave the sentinel untouched. A blit *into* the multisampled default framebuffer is
   `INVALID_OPERATION`, and the window still reads back identically afterwards (a command that generates an error has
   no other effect, §2.5). `glReadPixels` from a complete 4-sample FBO is `INVALID_OPERATION`, and the same call on the
   default framebuffer is not.

**Observed, not asserted** (printed as `note:` lines). For the window image and for the blit resolve the demo classifies
each pixel against the CPU projection. Each pixel's square is **grown by half a pixel** on every side, because sample
points may lie outside the pixel's unit square (§3.3) and vertices are snapped to the subpixel grid. The classes, and
what a weighted-average resolve would give in each:

| class | where | what an averaging resolve gives (8-bit, +-1 per channel) |
|---|---|---|
| exterior | grown square misses the silhouette | the background exactly |
| face | grown square inside one front face | that face's color exactly |
| seam | grown square inside the silhouette, touching 2-3 faces | R + G + B = 450 +-3, channels between those faces' colors |
| silhouette | the rest | channels between the background and the touching faces, sum in [81, 453] |

The demo prints how many pixels of each class deviate from that, how many pixels centred on the silhouette are strictly
blended, the summed coverage (face and seam pixels 1, silhouette pixels `(sum - 84) / 366`) against the silhouette's
area, how many pixels the blit resolve and the window's own resolve differ in (both are implementation-dependent and
need not agree), and how many blended pixels the single-sample control has (none) against the window (hundreds). On
radeonsi at 256 x 256: no deviation in any class, 90 of 90 centred pixels blended, coverage within 0.001 % of the area,
the two resolves identical.

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
- ES 3.0 §4.1.10 Additional Multisample Fragment Operations (per-sample stencil test and operations; "The method of
  combination is not specified"), §4.1.5 Depth Buffer Test, §4.1.6 Occlusion Queries, §2.5 GL Errors (an erroneous
  command has no effect), Appendix A.1 Repeatability
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
- **Resolved values are implementation-dependent.** Sample positions, the number of samples and the way samples are
  combined are all up to the implementation, for edge pixels and interior pixels alike. Test multisample rasterization
  per sample (stencil counts, occlusion queries, or `texelFetch` on a multisample texture in ES 3.1), and treat resolved
  colors as observations.
- **Software Mesa drivers** (llvmpipe, zink on lavapipe) cannot create a multisampled surfaceless pbuffer of the
  requested size. Use multisample renderbuffers there.
