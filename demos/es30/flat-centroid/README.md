# es30-flat-centroid — flat shading, centroid interpolation and sample coverage

Three rasterization features that act on single fragments and samples rather than on whole
primitives. They are easy to get subtly wrong:

* **Flat shading.** A `flat` output is not interpolated. Every fragment of a triangle receives
  the value of one vertex, the *provoking vertex*. In OpenGL ES that is always the **last**
  vertex of the triangle. ES 3.0 has no `glProvokingVertex`.
* **Centroid interpolation.** On a multisample target a pixel produces a fragment when any of
  its samples is covered, so its centre may lie outside the triangle. A default (`smooth`) input
  may then be evaluated outside the primitive, which extrapolates the attribute.
  `centroid in` forces evaluation at a point inside both the pixel and the primitive.
* **Multisample fragment operations.** `glSampleCoverage` and `SAMPLE_ALPHA_TO_COVERAGE` AND
  the fragment's per-sample coverage with a mask derived from a value or from alpha.

Minimum version: **OpenGL ES 3.0**, shaders `#version 300 es`.

## What you see

The surface is a 2 × 2 grid of panels at any window size:

| panel | content |
|---|---|
| top-left | A low-poly dusk landscape. A `TRIANGLE_STRIP` mountain ridge, a `TRIANGLE_FAN` sun and two kites made of independent `TRIANGLES`. Every vertex carries a palette index (`flat uint`) and a shade (`flat float`). Each facet is coloured by its last vertex. |
| top-right | Coverage bands for the values 0, 1/8, …, 1, rendered into a 4× multisample renderbuffer, resolved and magnified. Rows from the top: `glSampleCoverage(v, FALSE)` in cyan; `(v, TRUE)` in amber; both on top of each other (complementary masks, so no slot colour shows through); alpha-to-coverage with alpha = v in pink. With 4 samples the steps are visibly quantized, and alpha-to-coverage shows its dither pattern. |
| bottom-left (red frame) | A rotated tile from a 3 × 3 texture atlas, drawn into a 4× multisample renderbuffer **without** `centroid`, resolved and magnified. The neighbouring atlas cells are magenta, and magenta pixels bleed in along the edges. |
| bottom-right (green frame) | The same tile with `centroid in`. There is no bleed. |

## Central API and GLSL

* `flat out uint` / `flat out float` / `flat out int v_vertex_id = gl_VertexID` in
  `shaders/flat.vert`. Integer outputs *must* be `flat` (GLSL ES 3.00 §4.3.6).
* `glDrawArrays(GL_TRIANGLES / GL_TRIANGLE_STRIP / GL_TRIANGLE_FAN, first, count)` with
  `first > 0`, so `gl_VertexID` also identifies the provoking vertex absolutely.
* `glVertexAttribIPointer` for the integer palette attribute, and an `RGBA32UI` renderbuffer
  read back with `glReadPixels(..., GL_RGBA_INTEGER, GL_UNSIGNED_INT, ...)`.
* `centroid out` / `centroid in` (GLSL ES 3.00 §4.3.9) in `shaders/centroid.vert|frag`.
* `glRenderbufferStorageMultisample`, then `glBlitFramebuffer` to resolve into a texture.
* `glSampleCoverage(value, invert)`, `glEnable(GL_SAMPLE_COVERAGE)` and
  `glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE)`. Their state is read back with `glGetFloatv`,
  `glGetBooleanv` and `glIsEnabled`.
* `GL_ANY_SAMPLES_PASSED` occlusion queries and stencil `INCR` as resolve-free sample counters.

## Spec sections

* ES 3.0 §2.17 Flatshading and Table 2.12 (provoking vertex). Counting vertices from 1:
  independent triangle *i* → 3*i*, strip triangle *i* → *i*+2, fan triangle *i* → *i*+2.
* ES 3.0 §3.3 Multisampling, §3.6.1 and §3.6.3 Polygon (Multisample) Rasterization, §3.9.1
  ("When interpolating input variables declared using `centroid in`, the variable is sampled at
  a location within the pixel covered by the primitive").
* GLSL ES 3.00 §4.3.9 Interpolation. With multisampling and no `centroid`, the value is taken
  "to the pixel's center, or anywhere within the pixel, or to one of the pixel's samples". With
  `centroid`, it is taken "to a point that lies in both the pixel and in the primitive being
  rendered, or to one of the pixel's samples that falls within the primitive". With
  single-sampling it is taken at the pixel centre and `centroid` is ignored.
* ES 3.0 §4.1.3 Multisample Fragment Operations. The value is "clamped to [0,1] before being
  stored". Invert means "all bit values are inverted". The two masks are ANDed with the fragment
  coverage. The SAMPLE_COVERAGE mask "must have the same properties of proportionality and
  invariance" as the alpha mask, for which "all 1's [correspond] to the maximum of all alpha
  values, and all 0's [correspond] to all alpha values being 0". "No changes to the fragment
  alpha or coverage values are made at this step if the value of SAMPLE_BUFFERS is not one."
* ES 3.0 §4.1.6 Occlusion Queries (TRUE "if any fragment or sample passes the depth test"),
  §4.1.10 (per-sample stencil test, applied only to samples whose coverage bit is 1).
* ES 3.0 §4.3.2 (ReadPixels from a multisample FBO → `INVALID_OPERATION`), §4.3.3 (the resolve
  of normalized formats is *implementation-dependent*; a blit into a multisample draw
  framebuffer → `INVALID_OPERATION`), §4.4.2.1 (`RENDERBUFFER_SAMPLES` ≥ the request), §4.4.4.2.
* ES 3.0 Table 6.7 (initial multisample state) and Table 6.28 (`SUBPIXEL_BITS` ≥ 4).

## How it is validated

Run `./build/debug/bin/es30-flat-centroid --headless --validate`. The demo records 45 checks.

**Flat shading.** The landscape is drawn a second time into a 128 × 128 `RGBA32UI` target.
Each pixel stores `gl_VertexID`, the flat `uint`, the bits of the flat `float` and the bits of
the *smooth* copy of the float. For every triangle of every topology the CPU picks up to four
probe pixels: the centroid and the points halfway to each vertex, at least 1 px from every
edge. The first three values must equal those of the provoking vertex **bit for bit**. The vertex
data are checked at start-up, so a first-vertex or middle-vertex convention would fail: the
provoking vertex differs from the other two in palette index and shade, and `gl_VertexID` always
differs. The smooth copy is the **negative control**. At the centroid probe it must match the CPU
barycentric interpolation and differ from the provoking value. Its tolerance is the shade
gradient times 1/8 px, from vertex snapping at `SUBPIXEL_BITS` ≥ 4. The displayed panel is
probed too: every probed facet must show `palette[i] * shade` of its provoking vertex (±1,
8-bit quantization).

**Centroid.** Every conclusion comes from `GL_ANY_SAMPLES_PASSED`, not from the resolve. The
fragment shader discards every fragment that does *not* have the property under test. The tile
carries, besides its texture coordinate, the signed distance of each vertex to the tile's four
edge lines, in pixels. Interpolated, this is the distance of the evaluation point from each edge.
On the 4× multisample renderbuffer, with `centroid`:

* no fragment is evaluated more than 1/16 px outside the tile. Snapping to a 1/16 px grid can
  move an edge by about 1/32 px; the tolerance is twice that;
* no fragment samples a neighbouring atlas cell. The texture coordinates are inset by 1/8
  texel at about 0.92 texel per pixel;
* positive control: fragments evaluated inside the tile and within 2 px of an edge exist, so
  the query can return TRUE.

On the single-sample FBO, defined by the single-sample rule:

* no fragment is evaluated outside, with or without `centroid`;
* the centroid value equals the pixel-centre value;
* the same positive control applies.

Without `centroid` the evaluation point is allowed, but not required, to lie outside the
primitive. The demo therefore *prints* whether extrapolation and bleed happened. They did on
radeonsi, llvmpipe and zink. Replacing the centroid inputs with the default ones in the shader
makes both centroid checks fail, which shows that they discriminate.

**Sample coverage**, on a 4× multisample renderbuffer with `DEPTH24_STENCIL8`. A draw is observed
by an occlusion query: "no sample written" means the query is FALSE. "Every sample written
exactly once" is tested by drawing with stencil op `INCR`, then querying a full-screen draw with
stencil test `NOTEQUAL 1`, which must be FALSE. The checks:

* value 0 writes nothing; value 1 writes every sample; with invert the two swap;
* `(v, FALSE)` followed by `(v, TRUE)` writes every sample exactly once, for nine values of v;
* negative control: `(0.5, FALSE)` drawn twice does *not* write every sample exactly once;
* value 0.5 writes a partial mask (samples with stencil 1 and samples with stencil 0 both exist);
* alpha-to-coverage: alpha 0 writes nothing and alpha 1 writes everything. ANDed with coverage
  value 0, nothing is written. Alpha 0 ANDed with an all-ones coverage mask writes nothing;
* state: initial values (disabled, value 1, invert FALSE); `glSampleCoverage(1.75, TRUE)` reads
  back 1 and TRUE; `(-0.5, FALSE)` reads back 0; `(0.375, TRUE)` reads back exactly 0.375; the
  enable state through `glIsEnabled` and `glGetBooleanv`;
* the multisample FBO reports `SAMPLE_BUFFERS` 1 and `SAMPLES` = `RENDERBUFFER_SAMPLES` ≥ 4.
  The single-sample FBO reports 0 and 0, and there coverage value 0, value 1 inverted,
  alpha-to-coverage with alpha 0 and value 0.5 all change nothing. Every pixel is written,
  checked by exact readback and stencil count;
* the coverage panel and both zoom panels must show the resolved texels they magnify. They are
  compared with the resolved images read back from their textures, so the check holds whatever
  the resolve produced: on the single-sample default framebuffer the pixel centre selects exactly
  one texel (the demo confirms `SAMPLE_BUFFERS` 0 for the window first and skips the window
  probes otherwise);
* errors: `glBlitFramebuffer` into a multisample draw framebuffer and `glReadPixels` from a
  multisample FBO both generate `INVALID_OPERATION`.

**Observed, not asserted.** Every *resolved* value is printed as a `note:` line and never
asserted. ES 3.0 §4.3.3: "If the source formats are floating-point or normalized types, the
sample values for each pixel are resolved in an implementation-dependent manner." The
specification does not say that the result lies between the sample colours, that it is their
average, or even that a pixel whose samples all hold one colour resolves to that colour. So the
eight endpoint bands of the band image (value 0 and 1, inverted, complementary pairs,
alpha-to-coverage 0 and 1) are reported as "8 of 8 bands show the colour their samples were
given" (true on radeonsi, llvmpipe and zink), and the resolved coverage curves (17 values each)
are printed: on all three drivers they rise in steps of 1/4 for `glSampleCoverage` and in 1/16
steps (dithered) for alpha-to-coverage. What the endpoint bands show is asserted per sample, with
no resolve involved, by the occlusion-query and stencil checks above. (Earlier versions of this
demo asserted the endpoint bands as exact. That relied on an assumption the specification does
not make.)

## Pitfalls

* Not every implementation can create a multisampled *default* framebuffer. Here the
  surfaceless llvmpipe and zink pbuffers of a 4-sample EGLConfig come up as 0 × 0, unusable
  surfaces. So all multisampling happens in renderbuffers, which ES 3.0 guarantees up to
  `MAX_SAMPLES` ≥ 4.
* Reading back a multisample FBO is an error. Resolve with `glBlitFramebuffer` into a
  single-sample target first. The resolve of normalized formats is unspecified, so do not build
  exact expectations on resolved values, not even where all samples of a pixel agree. Count samples with stencil and occlusion
  queries instead.
* D3D-style "first vertex" flat shading does not exist in ES 3.0. Meshes ported from such APIs
  must rotate their index order.
* `centroid` fixes extrapolation, not derivatives. GLSL notes that derivatives of centroid
  inputs may be less accurate. The atlas uses NEAREST without mipmaps, so this does not matter
  here.
* A texture coordinate exactly on a cell boundary still selects the neighbouring texel with
  NEAREST, even when it is evaluated inside the primitive. Inset atlas coordinates slightly
  (here by 1/8 texel).
* The intermediate masks of `glSampleCoverage` and alpha-to-coverage are implementation-
  defined. Only 0, 1, invert and the AND are fully specified.
