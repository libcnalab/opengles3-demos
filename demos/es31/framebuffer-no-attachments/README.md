# es31-framebuffer-no-attachments

**API introduced in:** OpenGL ES 3.1 · **Minimum context:** ES 3.1 · **Shaders:** GLSL ES 3.10 (`#version 310 es`) ·
**Run:** `es31-framebuffer-no-attachments [--headless --validate]`

## Purpose

In ES 3.0 a framebuffer object is complete only if an image is attached to it, so a pass that wants to *compute* per-pixel (a
voxelizer, a coverage counter, a visibility pass that fills a buffer) still needs a dummy color target. ES 3.1 removes
that: a framebuffer with **no attachments** is complete as soon as `FRAMEBUFFER_DEFAULT_WIDTH` and
`FRAMEBUFFER_DEFAULT_HEIGHT` are non-zero, and the width, height and sample count that rasterization uses come from
`glFramebufferParameteri` instead of from an attachment. The fragment shader has nothing to write to except memory:
shader storage buffers and images.

The demo draws into such framebuffers and treats the fragment shader's *side effects* as the only observable result. It checks:

1. which fragments arrive (their number and their pixel), against a CPU model, **exactly**, under the viewport, the scissor box, the
   `(0,0)-(width,height)` rectangle of the framebuffer and the rasterization rules for rectangles and triangles;
2. the framebuffer parameters: initial values, completeness rules, limits and every error ES 3.2 section 9.2.1 defines;
3. `layout(early_fragment_tests)` in a depth-tested framebuffer: the shader runs for exactly the visible fragments, and no occluded one;
4. `gl_HelperInvocation`: helper invocations (extra invocations that exist only to supply derivatives to their neighbours) have no side effects.

## What the image shows

Eight tiles of 64 x 32 pixels, each the **hit map** that the GPU measured (one cell per framebuffer pixel, scaled 2x): dark = no fragment, white = one, yellow
= two (overlapping triangles), cyan = a fragment of an occluded rectangle that was shaded anyway; **red would be a pixel that disagrees with the CPU model**
(none do). Row 0 is at the bottom.

| Tile (row, column) | Run |
|---|---|
| 0,0 | rectangle [5,41) x [3,19) with scissor box (10,4) 20 x 12 |
| 0,1 | viewport (8,4) 32 x 16 and a quad that covers the whole viewport |
| 1,0 | viewport (-32,-16) 64 x 32, half outside the framebuffer |
| 1,1 | two triangles on the 1/16-pixel grid: a large one and a thin sliver that crosses it |
| 2,0 | a 16 x 8 color attachment on a framebuffer whose default size is 64 x 32: only the attachment's corner is rendered |
| 2,1 | early tests: rectangle B (behind) minus rectangle A (in front): an L shape |
| 3,0 | **no** early tests: B as shaded (the L plus a cyan block: occluded fragments that ran) |
| 3,1 | helper-invocation shader on the same two triangles |

## Central calls

| Call | Role | Spec (ES 3.2) |
|---|---|---|
| `glFramebufferParameteri(target, pname, param)` | default width, height, samples, fixed sample locations | 9.2.1 |
| `glGetFramebufferParameteriv(target, pname, &v)` | read them back | 9.2.3 |
| `glCheckFramebufferStatus` | complete when width and height are both non-zero | 9.4.2 |
| `glBindImageTexture`, `imageStore` | image side effects | 8.23, GLSL 8.12 |
| shader storage buffer + `atomicAdd/Min/Max` | counting side effects | 7.8, GLSL 8.11 |
| `glMemoryBarrier` | make shader writes visible to mapping and to `glReadPixels` | 7.11.2 |
| `layout(early_fragment_tests) in;` | depth test before the shader | 13.8.4, 14.2.4, GLSL 4.4.1 |
| `gl_HelperInvocation` | true for derivative-only invocations | GLSL 7.1.2 |

## How a fragment is observed

`count.frag` has **no output variable**. Each invocation does `atomicAdd(total, 1)`, `atomicMin/Max` on its own window position, and
`atomicAdd(cells[y * stride + x], 1)`: a hit map with one counter per pixel, plus bounds. The map has `kGuard` = 32 extra columns and rows, so a fragment
outside the framebuffer would land in the guard region and be counted instead of corrupting memory. After the draw the application calls
`glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT)`, maps the buffer and compares **every cell**.

The image variant (`count_image.frag`) uses `imageStore` on two `r32ui` images (GLSL ES 3.10 has no image atomics; they came with 3.20), writing the
constant 1 and the invocation's own `(y << 16) | x`. The images are read through a framebuffer with `glReadPixels` after
`glMemoryBarrier(FRAMEBUFFER_BARRIER_BIT | SHADER_IMAGE_ACCESS_BARRIER_BIT)`; every covered pixel must carry its own position code, every other pixel 0.

ES 3.2 section 7.11.1 states "the number of invocations per fragment is exactly one when the framebuffer has no multisample buffer", which is why exact
counts are legitimate here; a fragment that fails the scissor test is not shaded at all.

## The CPU model

* **Rectangles** have edges on pixel boundaries, so every pixel centre is strictly inside or outside: coverage is `rect ∩ scissor ∩ [0,width) x [0,height)`.
  The viewport does not clip (the quad is clipped to NDC), so a viewport larger than the framebuffer, or half outside it, still yields exactly the framebuffer's pixels.
* **Triangles** use vertices on the **1/16 pixel grid** (ES requires `SUBPIXEL_BITS` >= 4, so any implementation represents them exactly), in a 64 x 32
  viewport, where NDC values are exact floats. The model is the exact integer edge-function test on the pixel centre. A test-design check proves that no
  pixel centre lies exactly on an edge (the one case the rules leave to the implementation): 633 + 73 covered pixels, 0 on an edge. The two triangles
  overlap, so some pixels are hit twice, which the map shows.
* **A framebuffer of 33 x 17, then 40 x 8** (same object, resized) checks that the size is taken from the parameters at draw time.
* **MAX_FRAMEBUFFER_WIDTH x HEIGHT** (16384 x 16384 on these drivers): complete with no memory behind it; the viewport (bounded by `MAX_VIEWPORT_DIMS`) is
  covered by a full-NDC quad, and a 3 x 2 scissor box at its far corner must give exactly 6 fragments with those exact coordinates.
* **Attachments win.** With a 16 x 8 color renderbuffer attached and a default size of 64 x 32, only 16 x 8 = 128 fragments arrive; after detaching it the
  framebuffer is 64 x 32 again.

## Early fragment tests

The framebuffer has only a `DEPTH_COMPONENT24` renderbuffer (no color). Rectangle A (pixels [8,40) x [4,28), depth 0.25) is drawn first, then rectangle B
([24,56) x [10,31), depth 0.75), which A partly hides: B rasterizes 672 fragments, 384 of them visible. Two probe draws at depth 0.5 and 0.9, with depth writes
off, then show what the depth buffer holds.

* With `layout(early_fragment_tests)` the depth test runs **before** the shader and the shader does not run for a fragment that fails it (ES 3.2 13.8.4):
  B's invocations are **exactly** the 384 visible fragments, pixel for pixel.
* Without it the test runs after the shader. The spec still allows an implementation to skip a fragment that will never contribute
  (section 7.11.1), so the demo asserts only what is guaranteed: every one of the 384 visible fragments ran exactly once, nothing ran outside B, and the
  total lies between 384 and 672. It prints the measured count (672 on all tested drivers, 288 of them for occluded fragments).
* The result is the same either way: both variants leave the depth buffer holding A's 0.25, B's 0.75 where A is not in front and the cleared 1.0 elsewhere (the
  probes pass in exactly 1280 / 896 pixels). An occluded fragment shaded after the test must not write depth.
* A shader that statically writes `gl_FragDepth` together with `early_fragment_tests` must fail to compile (GLSL ES 3.10 section 4.4.1). Mesa accepts it
  (XFAIL, see below).

## Helper invocations

A fragment shader that uses derivatives (`helper.frag` calls `dFdx(gl_FragCoord.x)` and `dFdy(gl_FragCoord.y)`) needs the pixels of a 2 x 2 quad even where the
primitive covers only some of them; the others are **helper invocations**. GLSL ES 3.10 section 7.1.2: they execute the same code but their stores and
atomics "have no effect on the underlying image or buffer memory", and `gl_HelperInvocation` is true in them. The test cannot see a helper directly (it has no
effects, by definition), so it counts what a *leak* would change:

* `guarded` counts invocations that run `atomicAdd` behind `if (!gl_HelperInvocation)`;
* `unguarded` counts every invocation, with no guard at all;
* `helper_seen` is incremented only when `gl_HelperInvocation` is true;
* a per-pixel map of the unguarded atomic.

For two triangles (706 covered pixels, 125 uncovered pixels that share a quad with a covered one) and for a rectangle starting at an odd pixel (576 covered, 108 such pixels)
all of `guarded` and `unguarded` must equal the CPU coverage **exactly**, `helper_seen` must be 0, the map must equal the model, and `dFdx(gl_FragCoord.x)` and
`dFdy(gl_FragCoord.y)` must be 1 within 0.01 in every real invocation, also next to uncovered pixels (the derivative is that of a linear function, so any
approximation gives 1; the tolerance is a sanity bound because the spec leaves derivative precision open). The negative control is arithmetic: if the 125 uncovered pixels had
stored, `unguarded` would exceed the coverage, so the equality can fail.

## Parameters, completeness and errors

* A new framebuffer object has `FRAMEBUFFER_DEFAULT_WIDTH` = `HEIGHT` = `SAMPLES` = 0 and `FIXED_SAMPLE_LOCATIONS` = FALSE (table 21.15).
* No attachment and size 0 x 0, 16 x 0 or 0 x 16: `FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT`; a draw gives `INVALID_FRAMEBUFFER_OPERATION` and
  no fragment is shaded. 24 x 12: complete. Values read back as set.
* `MAX_FRAMEBUFFER_WIDTH` / `HEIGHT` >= 2048 and `MAX_FRAMEBUFFER_SAMPLES` >= 4 (table 21.40); the maxima themselves are accepted and complete; maxima + 1, -1 and `INT_MAX`
  give `INVALID_VALUE` and leave the parameters unchanged.
* The default framebuffer bound: `INVALID_OPERATION` for both `glFramebufferParameteri` and `glGetFramebufferParameteriv`. `READ_FRAMEBUFFER`, `DRAW_FRAMEBUFFER`
  and `FRAMEBUFFER` name the right object.
* `INVALID_ENUM` for a bad `target` or `pname` cannot be provoked through meta-gl's typed enums and is not tested.
* **Multisample.** `FRAMEBUFFER_DEFAULT_SAMPLES` = 4 is accepted and the framebuffer is complete; a covering triangle limited by a scissor box [5,41) x [3,19) produces
  between 1 and `MAX_FRAMEBUFFER_SAMPLES` invocations per pixel (section 7.11.1) inside the box. A single-sample control gives exactly 576. By section 9.2.1 the
  framebuffer "is considered to have sample buffers" if samples are non-zero, but Mesa reports `SAMPLE_BUFFERS` = 0 and `SAMPLES` = 0 for it (XFAIL below). Also
  on llvmpipe and zink (lavapipe) a multisample framebuffer shades the quad neighbours of a scissor box as ordinary invocations (XFAIL below).

## Validation summary

70 checks, every one against the CPU model or the specification text; none is "the call returned". Negative controls: the full-quad hit map judged against the sub-rectangle's
model differs in 1472 pixels and the two-triangle map against the first triangle alone in the 73 sliver pixels; the test-design checks prove that
B has occluded fragments (672 vs 384) and that helper candidates exist (125 and 108), so the exact equalities could fail; the single-sample control shows what the multisample
run is compared against.

## Known driver deviations (XFAIL, reported by `check.expect_or_known_deviation`)

| Driver | Rule | Observed |
|---|---|---|
| all Mesa drivers | GLSL ES 3.10 4.4.1: `early_fragment_tests` + a static write to `gl_FragDepth` is an error | compiles |
| all Mesa drivers | ES 3.2 9.2.1: no attachments and `FRAMEBUFFER_DEFAULT_SAMPLES` != 0: the framebuffer "is considered to have sample buffers" | `SAMPLE_BUFFERS` = 0, `SAMPLES` = 0 (rasterization is multisampled) |
| llvmpipe, zink on lavapipe | ES 3.2 7.11.1 (a fragment that fails the scissor test is not shaded) / GLSL ES 3.10 7.1.2 (helper stores have no effect) | in a **multisample** framebuffer (attachment-less or renderbuffer) the pixels outside a scissor box that share a 2 x 2 quad with it run with `gl_HelperInvocation` false and their atomics take effect (661 invocations for 576 pixels, 53 pixels outside the box hit); radeonsi and every single-sample run are exact |

## Pitfalls

* **GLSL ES 3.10 has no image atomics and no `gl_SampleMaskIn`.** They are GLSL ES 3.20 features; this demo is ES 3.1, so images are used with `imageStore` and
  counting is done with SSBO atomics.
* **`layout(early_fragment_tests)` forbids writing `gl_FragDepth`**, and the depth buffer, stencil buffer and occlusion counts may then be updated even for a fragment that is
  discarded after the shader (section 13.8.4). It is the switch that makes an SSBO/image write in a fragment shader mean "this fragment passed the depth test".
* **Without the qualifier the number of shader invocations is not what you see on screen.** Occluded fragments may run (and did on every driver tested), so
  never use a late-tested shader's side-effect count as a visibility count.
* **Helper invocations are not a problem for correctness, only for assumptions.** Do not count "fragments" with an image/SSBO write and expect extra pixels; the
  spec makes those writes vanish. Do not rely on `gl_HelperInvocation` to enumerate them.
* **A framebuffer with no attachments is incomplete until both default sizes are non-zero.** Setting only the width is not enough
  (`FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT`), and drawing into it raises `INVALID_FRAMEBUFFER_OPERATION`.
* **Bind the right target.** `glFramebufferParameteri(READ_FRAMEBUFFER, ...)` configures the framebuffer bound for *reading*; `FRAMEBUFFER` means `DRAW_FRAMEBUFFER`.
* **The viewport is not the framebuffer.** It can be larger or half outside; only the framebuffer rectangle limits rendering when there are no attachments.
* Reading an SSBO written by a draw needs `glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT)`; reading an image through `glReadPixels` needs `FRAMEBUFFER_BARRIER_BIT`.

## meta-gl notes

No gap blocked this demo. Ergonomics only: `glDrawArrays`-style draws into an attachment-less framebuffer are ordinary; the `FramebufferDefaultParameter::Layers`
enumerator is ES 3.2 and is not used (the layer count is a 3.2 parameter).
