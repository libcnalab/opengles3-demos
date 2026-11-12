# es31-texture-gather

**API introduced in:** OpenGL ES 3.1 · **Minimum context:** ES 3.1 · **Shaders:** GLSL ES 3.10 (one more shader, GLSL ES 3.20, runs only on an ES 3.2 context)

## Purpose

`textureGather` returns one component of each of the four texels that the LINEAR filter would blend, without
blending them. The demo asks one gather question per fragment, reads every answer back, and compares it with a CPU
model of the rules in the specification. It covers 2D, 2D array and cube textures, all four component selections, the
three wrap modes (including the footprints that reach past the edge of the image), `textureGatherOffset` up to the
limits the implementation reports, shadow samplers (which gather comparison results), integer samplers, the texture
swizzle, incomplete textures, and the compile-time rules for the `comp` and `offset` arguments.

## Expected image

The window is a 4 x 4 grid of cells that tiles the surface exactly (boundaries at `k * w / 4` and `k * h / 4`; 64 x 64 cells at
the default 256 x 256). Each cell shows a 9 x 9 (or 7 x 7) result image, one block per footprint, magnified with a NEAREST blit
by the largest integer factor per axis that leaves a one-pixel gap (7 x 7 pixels per block at 256 x 256; the blocks are
rectangular when the cells are) and centred in the cell. An integer factor keeps every block exactly one source pixel. Rows 0 to 2 (from the top) are `CLAMP_TO_EDGE`, `REPEAT` and
`MIRRORED_REPEAT`, the columns are `comp` 0 to 3. The source texture is made of four ramps (one per component) that
run in different directions, so each cell is a smooth gradient with a visible seam where the footprint wraps around
the edge. The red, green and blue channels of a cell are the x, y and z results of the same gather, i.e. the same ramp
shifted by one texel and one row, which is the figure 8.7 order made visible. The last row shows an array layer, one
cube map face, a shadow gather (a binary pattern: every channel is 0 or 1) and a gather with the offset (2, -3).

## Key ideas

- **The footprint.** The gather takes the same four texels as bilinear filtering. For a point at texel coordinates
  (u, v) they are i0 = floor(u - 0.5), j0 = floor(v - 0.5) and the three neighbours i1 = i0 + 1, j1 = j0 + 1, each
  passed through the wrap mode. The result is (i0 j1, i1 j1, i1 j0, i0 j0), that is **x = upper left, y = upper
  right, z = lower right, w = lower left** in figure 8.7 (t grows upwards). It is not the row-major order.
- **How the demo hits one footprint exactly.** The fragment at window pixel (a, b) queries the point (a, b) in texel
  units of an 8 x 8 texture, the corner shared by the four texels (a-1, b-1) to (a, b). That point is half a texel from
  every texel centre, so no rounding can select a neighbouring footprint. A 9 x 9 viewport therefore visits all 81
  footprints of the texture, among them the 32 that include texels outside the image, where the wrap mode decides.
- **Distinct texels.** Every (texel, component) pair of the 8-bit textures has its own value (a permutation of 0..255),
  and the 32-bit and signed textures follow the same scheme. A wrong texel, a wrong component, a wrong order or a wrong
  wrap result can therefore never give the right number by accident.
- **`comp`** must be a constant integer expression (0 to 3). It selects the component after the texture swizzle, so a
  swizzle of (A, ONE, ZERO, R) makes comp 1 return all ones and comp 2 all zeros. Without `comp` the x component is
  selected.
- **Filters are ignored**, completeness is not: the demo's textures use NEAREST filters so that they are complete (the
  default `NEAREST_MIPMAP_LINEAR` would make a texture without mipmaps incomplete); a LINEAR filter gives the same
  gather. An incomplete texture gathers `(0, 0, 0, 1)` from all four texels.
- **Offsets** in `textureGatherOffset` are added to the texel coordinates before the footprint is chosen, so they move
  the whole footprint by whole texels. They must lie in `[MIN_PROGRAM_TEXTURE_GATHER_OFFSET,
  MAX_PROGRAM_TEXTURE_GATHER_OFFSET]`, which section 8.14.2 requires to be at least as wide as the limits of
  `textureOffset` (-8 and 7). In GLSL ES 3.10 the offset must be a constant expression. The demo builds one program per
  offset and uses the queried extremes (-32 and 31 on the Mesa drivers of the test machine); they are larger than the
  8-texel image, so wrapping is part of the test. The offset never applies to the array layer.
- **Shadow samplers** compare each of the four depths with the reference value and return the four comparison
  results (1.0 or 0.0) in the same x, y, z, w order. The texture must be a depth texture with
  `TEXTURE_COMPARE_MODE = COMPARE_REF_TO_TEXTURE`.
- **Cube maps** are addressed by a direction. The shader builds the direction for a face coordinate (s, t) by inverting
  table 8.20 of the specification, so the test also checks the face selection. Only footprints inside a face are
  compared: the others reach the neighbouring faces, where seamless filtering applies.
- **`textureGatherOffsets`** (four independent offsets, constant array) belongs to GLSL ES 3.20 (section 8.9.4 there),
  not to 3.10. The demo compiles `shaders/gather_offsets.frag` (`#version 320 es`) only when the context reports ES 3.2
  and checks it on the 9 x 9 footprints. In the `es-min` runs, which report ES 3.1, that check is skipped and a note
  is printed. The same release of GLSL ES also allows a non-constant offset in `textureGatherOffset`; the 3.10 rule is
  tested as a compile error.

## Central calls

`textureGather`, `textureGatherOffset` (GLSL ES 3.10 section 8.9.3); `glTexStorage2D`, `glTexStorage3D`,
`glTexSubImage2D`, `glTexSubImage3D`, `glTexImage2D`, `glTexParameteri` (wrap, filter, swizzle, compare function),
`glGetIntegerv` (`MIN/MAX_PROGRAM_TEXTURE_GATHER_OFFSET`, `MIN/MAX_PROGRAM_TEXEL_OFFSET`), and the framebuffer calls that
render each result into a single-sample target (`RGBA8`, `RGBA32I`, `RGBA32UI`) for `glReadPixels`.

## Specification

- ES 3.2 section 8.14.2 (texel selection, figure 8.7, table 8.21, offsets, incomplete textures), 8.13 and 8.13.1 (cube
  map selection, table 8.20), 8.17 (completeness), 8.20.1 (table 8.22, depth comparison), 8.10 (texture parameters),
  8.5 and 8.18 (image specification, immutable storage), chapter 21 (the offset limits).
- GLSL ES 3.10 section 8.9.3 (gather), 8.9.2 (`textureOffset`: constant offsets), 4.7.4 (no default precision for most
  sampler types: every sampler in the demo is declared `highp`).
- GLSL ES 3.20 section 8.9.4 (`textureGatherOffsets`, non-constant offsets).

## Validation

88 checks on an ES 3.2 context (87 on a context that reports 3.1), all exact:

- the limit relations of section 8.14.2, and the guaranteed range [-8, 7];
- the order of figure 8.7 twice: once as the whole 9 x 9 image per wrap mode and comp (12 images), once from four
  independently written texel lookups at the footprint (5, 5) for every comp;
- comp omitted, a constant-expression comp, LINEAR filters, and the swizzle (A, ONE, ZERO, R) with all four comps;
- `isampler2D` on `RGBA8I` (negative values) and `usampler2D` on `RGBA32UI` (values above 2^24, which a float path would
  damage), every comp;
- array layers 0 to 2 and the clamped layer coordinates 7 and -3; all six cube faces with all four comps;
- 18 constant offsets (small, mixed signs and the four combinations of the queried extremes) with CLAMP_TO_EDGE and
  REPEAT, further comps, the array and the integer forms;
- shadow gathers with four comparison functions, with an offset, on array and cube textures, and the observation that LESS
  and GEQUAL are complementary;
- the incomplete texture, the three compile-time checks (a uniform as `comp` is rejected, a uniform-derived offset is
  rejected, constant expressions compile), and on ES 3.2 the `textureGatherOffsets` shader;
- the window: all 16 cells, every block (probed at its centre pixel), against the same model (alpha is made opaque with an
  alpha-only clear). The demo skips below 40 x 40 pixels, where a cell cannot hold a 9 x 9 image plus the gap.

Exactness is justified by the rules themselves. A gather does no filtering: it converts four stored texels to floating
point (equation 2.1, c / 255 for RGBA8) and the render target converts the value back (the float to fixed-point conversion of chapter 2), which is the
identity for 8-bit values; dithering is disabled. Integer results are the stored integers. Each result image is preceded
by a clear to a sentinel value, so a draw that wrote nothing cannot match. **Negative controls:** a model with the
row-major order must disagree with the observed image in 77 of 81 pixels, and the CLAMP_TO_EDGE and REPEAT images must
differ in exactly the 32 edge footprints and nowhere else. During development, shifting the sample point by one texel
made 70 checks fail.

## Pitfalls

- The default `TEXTURE_MIN_FILTER` is `NEAREST_MIPMAP_LINEAR`. A texture without mipmaps is then incomplete, and
  gathering from it returns `(0, 0, 0, 1)`; the filter itself is ignored by gather, but completeness is not.
- The default wrap mode is `REPEAT`, not `CLAMP_TO_EDGE`. The first version of this demo assumed clamping for the depth and
  array textures and failed on their edge footprints.
- A gather result is not a 2 x 2 block in row-major order: the first component is the upper-left texel.
- A shadow sampler gathers comparison results, not depths; the reference value is compared with each texel before they
  are packed.
- `comp` and (in GLSL ES 3.10) the offset must be constant expressions. A uniform is a compile error.
- `CLAMP_TO_BORDER` is an ES 3.2 wrap mode and is not used by this ES 3.1 demo.
- Cube map footprints at the edge of a face use texels of the neighbouring face (seamless filtering) and, at a corner,
  an implementation-defined fourth texel, so they are not asserted.
