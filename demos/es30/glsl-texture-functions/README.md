# es30-glsl-texture-functions

**Introduced in:** OpenGL ES 3.0 / GLSL ES 3.00 · **Minimum context:** ES 3.0 · **Shaders:** `#version 300 es`

## Purpose

GLSL ES 3.00 replaced the ES 2.0 `texture2D`/`textureCube` pair with one overloaded family of lookup functions
(§8.8): `texture` (with an optional bias), `textureProj`, `textureLod`, `textureOffset`, `textureLodOffset`,
`textureGrad`, `textureGradOffset`, the projective `textureProjLod`, `textureProjOffset`, `textureProjLodOffset`,
`textureProjGrad`, `textureProjGradOffset`, the integer-coordinate `texelFetch` and `texelFetchOffset`, and
`textureSize`. Each of them exists for several sampler types: float, signed and unsigned integer samplers of 2D, 3D,
2D array and cube textures, and the depth-comparing shadow samplers. §8.9 adds `dFdx`, `dFdy` and `fwidth`.

This demo calls every one of those forms on every sampler kind that has it, in the fragment stage and (where the
function exists without derivatives) in the vertex stage, and compares the result **texel by texel** with a CPU model
of the ES 3.0 level-of-detail and texel-selection rules. It complements `es30-texture-swizzle-lod`, which looks at how
texture parameters (swizzle, BASE/MAX_LEVEL, MIN/MAX_LOD) act on a few of these functions on a 2D texture; here the
subject is the functions themselves across all sampler types.

## The picture

The left part (the top part on a portrait window) is a mip-level visualisation: a ground plane in perspective sampled
with `texture()`; every level of the 32×16 test texture is shown in its own palette colour (level 0 red, 1 orange,
2 yellow, 3 green, 4 cyan, 5 blue), with a parity checkerboard of the texels. The bands are where the implementation's
implicit level of detail crosses a level; their exact position is implementation-dependent and is not checked.

The rest of the window is a grid with one cell per fragment-stage lookup (154 cells, row-major from the top left).
A cell is the 4×4 result block of that lookup, magnified: lookups on RGBA8 and integer textures show the palette colour
of the level they returned with the texel-parity shade; shadow lookups show green when the result pair is the expected
(1, 0); `textureSize` cells encode width/height/depth in red/green/blue; derivative cells show |value| / 8.
Layout and cell sizes are relative to the window; below 8×8 pixels per cell the demo reports `SKIP_UNSUPPORTED`.

## Central API calls

- `glTexImage2D` / `glTexImage3D` for every level (and every cube face) of ten textures: RGBA8 2D, 3D, 2D array and
  cube; RGBA32UI 2D, RGBA32I 3D, RGBA8UI cube; DEPTH_COMPONENT32F 2D, cube and 2D array.
- `glTexParameteri`: `NEAREST_MIPMAP_NEAREST` / `NEAREST`, wrap modes, `COMPARE_REF_TO_TEXTURE` with `LEQUAL`.
- An RGBA32UI colour attachment (`glFramebufferTexture2D`), `glScissor` per 4×4 block, `glReadPixels` with
  `RGBA_INTEGER` / `UNSIGNED_INT`.
- `glTransformFeedbackVaryings`, `glBeginTransformFeedback(POINTS)` with `RASTERIZER_DISCARD`, `glMapBufferRange` for
  the vertex-stage lookups.
- `glGetIntegerv(MIN_/MAX_PROGRAM_TEXEL_OFFSET)`, `glGetFloatv(MAX_TEXTURE_LOD_BIAS)`.

## Spec sections

- GLSL ES 3.00 §8.8 Texture Lookup Functions, §8.9 Fragment Processing Functions.
- ES 3.0 §2.12.9.2 Texel Fetches, §2.12.9.3 Texture Size Query, §2.12.9.4 Texture Access (vertex stage).
- ES 3.0 §3.8.9 Cube Map Texture Selection (Table 3.21), §3.8.9.1 Seamless Cube Map Filtering.
- ES 3.0 §3.8.10.1 Scale Factor and Level of Detail (eq. 3.16–3.20), §3.8.10.2 Coordinate Wrapping and Texel Selection
  (Table 3.22), §3.8.10.4 Mipmapping (eq. 3.23), §3.8.13 Texture Completeness.
- ES 3.0 §3.8.15.1 Depth Texture Comparison Mode (Table 3.23).
- ES 3.0 Table 6.28 (`MAX_TEXTURE_LOD_BIAS` ≥ 2.0), Table 6.32 (`MIN_PROGRAM_TEXEL_OFFSET` ≤ −8,
  `MAX_PROGRAM_TEXEL_OFFSET` ≥ 7).

## How the results are made exact

**Self-describing texels.** Every texel of every level stores its own address. RGBA8 texels hold
(32·level + 16, 8·i, 8·j, 255 − 16·a), where *a* is the 3D slice, the array layer or the cube face; the RGBA32UI texture
holds (level, i, j, 0xC0DE0000 + a), the RGBA32I texture (−1−level, −1−i, −1−j, −1−a) (so the sign survives), the
RGBA8UI cube (level, i, j, face), and depth texels a distinct odd multiple of 1/32768 per (face/layer, level, i, j).
A lookup result therefore *names* the level and texel the implementation chose.

**One texel per lookup.** All textures use `NEAREST_MIPMAP_NEAREST` and `NEAREST`. That is the only mipmapped filter
that is legal for every format here (integer and 32-bit depth formats are not filterable, §3.8.13), and it makes the
result a single texel of a single level: there are no filter weights to model. `texelFetch` with a non-zero lod also
*requires* a mipmap filter (§2.12.9.2 makes it undefined otherwise).

**Raw bits out.** The fragment shader writes `floatBitsToUint` of float results (and integer results unchanged) to an
RGBA32UI target, which is colour-renderable in core ES 3.0. RGBA32F would need `EXT_color_buffer_float`, an extension.
An RGBA8 lookup returns c/255 (§2.1.6.1); the comparison accepts a quarter of a byte step, which still separates
neighbouring texels (they differ by at least 8 steps). Shadow, integer and size results are compared exactly.

**Exact coordinates and derivatives.** In a 4×4 block the coordinate is `P = u_p + u_ddx·f.x + u_ddy·f.y`, with f the
fragment centre relative to the block (0.5 … 3.5). All inputs are short dyadic fractions, so P is exact in binary32
and its implicit derivatives are exactly `u_ddx` and `u_ddy`. Projective cases keep q constant over the block and use
q ∈ {½, 2, 4} so the division is exact.

**Level selection only where it is unambiguous.** The implementation latitude is in ρ and in the rounding of λ:

- §3.8.10.1 allows any approximation f of ρ that is continuous, monotonic and satisfies
  `max{mu, mv, mw} ≤ f ≤ mu + mv + mw`. Most cases put the whole derivative on one texel axis (e.g.
  `dPdx = (1/4, 0)`, `dPdy = 0` on a 32-wide texture: ∂u/∂x = 8); then both bounds coincide and λ = log2 8 = 3 exactly.
  One `textureGrad` case deliberately uses two axes (mu = 8, mv = 2) to show the latitude: λ ∈ [3, 3.32], level 3
  either way.
- Eq. 3.23 rounds λ with `⌈λ + ½⌉ − 1`, its footnote allows `⌊λ + ½⌋`. They only differ at λ = n + ½.

The CPU model evaluates λ at both ends of the ρ interval (plus the bias clamped to ±`MAX_TEXTURE_LOD_BIAS`, eq. 3.17)
and `init()` refuses the case table unless every case selects the same level at both ends, at least 0.1 away from any
n + ½. Explicit lods are integers or clearly fractional (2.4 → 2, 2.6 → 3, 0.3 → 0, −1 → 0 (magnified), 9 → the last
level). Every sample position is at least 1/32 texel away from a texel boundary, and array layer coordinates are away
from x.5 (where RNE and ⌊r + ½⌋ may differ). The one case whose outcome depends on the implementation limit — a bias
of +100, clamped to `MAX_TEXTURE_LOD_BIAS` — is dropped with a note if that limit happened to make it ambiguous.

**Cube maps.** Directions have |ma| = 1 and their derivatives have no component along the major axis. Then the chain
rule gives exactly ∂s = ½·∂sc (Table 3.21), which is what eq. 3.19 needs for `u = wt·s` — for implicit derivatives as
well as for `textureGrad`, whose GLSL text says the gradients are taken "before texture coordinates are projected onto
the appropriate cube face". With NEAREST filtering a cube map always clamps to the edge (§3.8.9.1).

## What is covered

| sampler | functions (fragment stage unless noted) |
|---|---|
| `sampler2D` | `texture` (no bias, +1, −3 → magnified, +100 → clamped), `textureProj` vec3 and vec4 (with bias; the vec4 form's z is ignored), `textureLod` (5 lods), `textureOffset` at (`MAX_`, `MIN_PROGRAM_TEXEL_OFFSET`) and with bias, `textureLodOffset`, `textureGrad` (x only, y only, both), `textureGradOffset`, `textureProjLod` vec3 and vec4 (q = ½), `textureProjOffset`, `textureProjLodOffset`, `textureProjGrad`, `textureProjGradOffset`, `texelFetch` (levels 0–2), `texelFetchOffset`; vertex stage: `texture`, `textureLod`, `textureOffset`, `textureProj`, `texelFetchOffset` |
| `sampler3D` | `texture` with bias (ρ from ∂r only), `textureLod`, `textureOffset`, `textureLodOffset`, `textureGrad` (∂r, ∂t), `textureGradOffset`, `textureProj`, `textureProjLod`, `textureProjOffset`, `textureProjLodOffset`, `textureProjGrad`, `textureProjGradOffset`, `texelFetch`, `texelFetchOffset`; vertex: `textureProjLod`, `texelFetch` |
| `sampler2DArray` | `texture` with bias (layer changing per pixel), `textureLod` with layers 2, 7.25 → 3 (clamped), −2 → 0, `textureOffset`, `textureLodOffset`, `textureGrad`, `textureGradOffset`, `texelFetch`, `texelFetchOffset`; vertex: `textureLodOffset` |
| `samplerCube` | `texture` with and without bias, `textureLod` on all six faces, `textureGrad` along sc and tc; vertex: `textureLod` |
| `usampler2D` | `texelFetch`, `textureLod`, `textureOffset` with bias, `textureGrad`, `textureProj`; vertex: `textureLod` |
| `isampler3D` | `texture` with bias, `texelFetchOffset`, `textureLodOffset`, `textureProjLod`; vertex: `texelFetch` |
| `usamplerCube` | `texture` with bias, `textureLod` |
| `sampler2DShadow` | `texture` with bias, `textureLod`, `textureOffset`, `textureLodOffset`, `textureGrad`, `textureGradOffset`, `textureProj`, `textureProjLod`, `textureProjOffset`, `textureProjGrad`, `textureProjLodOffset`, `textureProjGradOffset`; vertex: `textureLod` |
| `samplerCubeShadow` | `texture` with bias, `textureGrad` |
| `sampler2DArrayShadow` | `texture`, `textureGrad`, `textureGradOffset` |
| all ten | `textureSize` of every level (vertex stage: cube and 2D array at level 3) |

**Shadow lookups** return one bit, so each pixel does two comparisons (`LEQUAL`) with references just below and just
above the depth of the texel the model expects (±¼ of the depth spacing, passed per pixel in `u_ref[16]`). The pair is
(1, 0) only if the sampled depth equals the expected one; any other texel gives (1, 1) or (0, 0). In the projective
forms the reference is the third component and is divided by q like the coordinates.

**Vertex stage.** §2.12.9.4: a vertex shader has no derivatives; λ_base is the explicit lod, or 0 for `texture`,
`textureOffset` and `textureProj`, which therefore read the base level. Thirteen lookups run as 13 points with
`gl_VertexID` selecting the case and are captured with transform feedback. `textureGrad` is not run in the vertex stage:
GLSL ES 3.00 lists it for all stages, but §2.12.9.4 says a vertex lookup without an explicit lod value uses
λ_base = 0 and that ρ is ignored, so the two texts disagree about what the gradients do there.

**Derivatives.** §8.9 defines dFdx/dFdy by forward or backward differencing (or other methods), with no precision.
For a function that is affine in window coordinates every such difference *is* the slope, so the demo asserts
`dFdx`, `dFdy` and `fwidth` of `(3x − 5y + 7, −2x + ½y, ¾x + ¾y, ¼y)` (x, y = `gl_FragCoord`), of an interpolated
varying and of the uniform-built coordinate P, with a sanity tolerance of 10⁻³ (10⁻² for the varying), and checks
`fwidth = |dFdx| + |dFdy|` on the measured values. `dFdx(x²)` is only printed: it is 2x ± 1 depending on the method,
and all Mesa drivers here show one difference per pixel pair of a 2×2 quad (+1, −1, +1, −1).

**Compile-time rules.** Five shaders must fail to compile, each next to a positive control that differs only in the
rule: a bias argument in a vertex shader (§8.8: "The bias parameter is not accepted in a vertex shader"), `dFdx` in a
vertex shader (§8.9: fragment only), a non-constant offset (§8.8: "The offset value must be a constant expression"),
`textureOffset` on a cube sampler and `texelFetch` on a cube sampler (no such overloads).

## Validation

- 161 lookups (148 fragment blocks of 16 pixels, 13 vertices) compared with the CPU model; a failing case reports the
  first pixel with the value it got and the level/texel the model expected.
- 207 negative controls: for every case one or more mutated models must disagree with the measurement somewhere —
  the model without the texel offset, ignoring the bias, one level off, without the projective divide, counting the
  array layer in ρ, or with s mirrored on the cube face. `init()` also verifies that each mutation changes the
  prediction, so none of these controls is vacuous.
- Limits from Tables 6.28 and 6.32; the queried offset limits are compiled into the offset shader and used.
- Derivative checks and compile-time rules as above.
- The result cells of the picture are probed (one pixel inside texel (1, 1) of each lookup cell) against the palette
  colour of the model's texel.
- Unwritten blocks are detected: the target is first cleared to 0xDEADBEEF with `glClearBufferuiv`.

Results on the reference machine (radeonsi, llvmpipe, zink; ES 3.2 and the ES 3.0 override): 392/392 checks pass on
every driver. All report `MIN/MAX_PROGRAM_TEXEL_OFFSET` = −32/31 and `MAX_TEXTURE_LOD_BIAS` = 16.

## Run

```
./build/debug/bin/es30-glsl-texture-functions                    # window
./build/debug/bin/es30-glsl-texture-functions --headless --validate --capture out.png
ctest --preset debug -R es30-glsl-texture-functions
```

## Pitfalls

- **Default sampler precision.** Only `sampler2D` and `samplerCube` have a default precision (lowp); `sampler3D`,
  the array, shadow and integer samplers have none in either stage and must be qualified (GLSL ES 3.00 §4.5.4). The
  return value of a lookup has the precision of its sampler (§8, introduction), so a `lowp` sampler returns `lowp`
  values. All samplers here are declared `highp`.
- **`texelFetch` and the minification filter.** Fetching a level other than the base level is undefined when the
  minification filter is `NEAREST` or `LINEAR` (§2.12.9.2). Use a mipmap filter.
- **Integer and depth textures are not filterable.** Their only complete mipmapped configuration is
  `NEAREST_MIPMAP_NEAREST` with `NEAREST` magnification (§3.8.13).
- **Offsets are constant expressions** and do not apply to the array layer; cube maps have no offset forms.
- **`textureProj` with a vec4 on a 2D sampler ignores P.z**, and the shadow forms divide the reference by q too.
- **Array layers are rounded, not floored:** layer = clamp(⌊r + ½⌋, 0, layers − 1), and r does not contribute to ρ.
- **Implicit derivatives need uniform control flow.** The probe shader selects the case with a `switch` on a uniform.
- **Driver limit.** A first version used one shader with a `switch` over all ~80 cases; llvmpipe and zink (on
  lavapipe) crashed inside the driver on its first draw (the backtrace shows a deep recursion in libgallium, consistent
  with the depth of the nested control flow a long `switch` becomes). `probe.frag` is therefore compiled six times
  with `#define GROUP n`, one sampler family per program.
- Out-of-range `texelFetch` is undefined without robustness and is not tested.
