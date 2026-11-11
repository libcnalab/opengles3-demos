# es20-mipmaps

**API introduced in:** OpenGL ES 2.0 (`TEXTURE_BASE_LEVEL`/`TEXTURE_MAX_LEVEL`/`TEXTURE_MIN_LOD`/`TEXTURE_MAX_LOD`
and `textureSize` arrived with ES 3.0) · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

A mipmap is an ordered set of images of the same picture, each half the size of the previous one. The GL
picks a level (or two) from the *level of detail* λ of the fragment, and the four mipmap minification
filters differ in two independent decisions: how the level is chosen (`NEAREST_` = the closest level,
`_LINEAR` = blend the two closest) and how a texel is taken inside a level (`NEAREST_MIPMAP_` = nearest texel,
`LINEAR_MIPMAP_` = bilinear). This demo makes both decisions visible and checks them against the equations of
ES 3.0 §3.8.10 — and checks the rules that decide whether a mip chain is usable at all (§3.8.13), and what
`glGenerateMipmap` does and does not promise (§3.8.10.5).

All levels are uploaded by hand, level by level, so every level has a content of its own: seven solid colours
("ladder": red, orange, yellow, green, cyan, blue, magenta for levels 0…6 of a 64×64 texture) and a texel
pattern ("stripes").

## The image

The 256×256 framebuffer is a grid of 8 columns by 15 rows of 32×17-pixel cells (a "strip" each). Column `k`
is drawn at λ = k (or k + ¼, k + ½, k + ¾ as stated):

| row | texture | filter | λ | what you see |
|---|---|---|---|---|
| 0 | ladder | NEAREST_MIPMAP_NEAREST | k | level k's colour: red … magenta; columns 6 and 7 are both level 6 (the last level, q = 6) |
| 1 | ladder | LINEAR_MIPMAP_NEAREST | k | identical to row 0 (a solid level looks the same whichever way it is filtered) |
| 2–3 | ladder | NEAREST_/LINEAR_MIPMAP_LINEAR | k + ½ | the 50/50 blend of levels k and k+1 (columns 6, 7: level 6) |
| 4 / 5 | ladder | NEAREST_MIPMAP_NEAREST | k + ¼ / k + ¾ | level k / level k+1: the nearest level is chosen by rounding λ |
| 6–9 | stripes | the four filters | k, k, k + ½, k + ½ | stripes of texels; row 7 differs from row 6 because it interpolates between texels |
| 10 | checker | NEAREST_MIPMAP_NEAREST | k | a 2×2 checker (blown up to 64×64) whose levels 1–6 were made by `glGenerateMipmap`: two colour halves, then one mixed colour at the 1×1 level |
| 11–12 | 7 pairs | NEAREST_MIPMAP_NEAREST (default for one) | — | `[incomplete (black) | complete twin]`: missing level, wrong-size level, wrong-format level, BASE > MAX, MAX_LEVEL fix, MIN_FILTER fix, no image |
| 13 | ladder | NEAREST_ and _MIPMAP_LINEAR | 1…5.5 | `TEXTURE_BASE_LEVEL` 2, `TEXTURE_MAX_LEVEL` 3 / 4 |
| 14 | ladder | NEAREST_ and _MIPMAP_LINEAR | 0, 5 | `TEXTURE_MIN_LOD` 2 / 1.5, `TEXTURE_MAX_LOD` 3 / 2.5 |

## Key ideas

- **A strip has a λ you can state in advance.** Each cell is a quad over its whole viewport whose `s`
  coordinate grows linearly with `x` and whose `t` coordinate is a constant (`shaders/strip.vert`). Then
  `∂u/∂x = w·Δs/width`, `∂u/∂y = ∂v/∂x = ∂v/∂y = 0`, and the scale factor of equation 3.20 is exactly
  ρ = |∂u/∂x|. The spec lets an implementation approximate ρ for general quads (`max{mu, mv} ≤ f ≤ mu + mv`
  with `mu`, `mv` the largest derivative magnitudes, §3.8.10.1), but with `mv = 0` every allowed
  approximation equals ρ. So λ = log₂ρ is a property of the test, not of the driver. The demo chooses
  `Δs` so that λ = k (one level-k texel per pixel), k + ¼, k + ½, k + ¾.
- **Choosing the level.** `NEAREST_MIPMAP_*` uses equation 3.23: `d = ⌈base + λ + ½⌉ − 1` for λ > ½ (capped at q),
  `d = base` otherwise, i.e. λ is rounded. `*_MIPMAP_LINEAR` uses equations 3.24/3.25 and
  `τ = (1 − frac λ)·τ₁ + frac λ·τ₂` of levels `⌊base + λ⌋` and the next. λ ≤ 0 is *magnification*:
  `TEXTURE_MAG_FILTER` applies, on the base level (§3.8.11).
- **Ties are implementation-defined.** λ = k + ½ is the boundary of equation 3.23, and footnote 4 of the spec
  allows `⌊base + λ + ½⌋` instead of `⌈base + λ + ½⌉ − 1`. radeonsi indeed picks level 1 at λ = ½. The demo
  therefore only asks `*_MIPMAP_NEAREST` for λ at ¼ distance from a boundary (k, k + ¼, k + ¾) and uses
  λ = k + ½ only with the blending filters, where the result is 50/50 either way.
- **Inside a level.** NEAREST takes texel ⌊u⌋; LINEAR blends ⌊u − ½⌋ and the next with weight
  `frac(u − ½)` (§3.8.10.2, Table 3.22; `CLAMP_TO_EDGE` here). In the "stripes" strips with λ = k there
  is one level-k texel per pixel and the sample point is ¼ texel past a texel centre, so NEAREST returns a
  texel and LINEAR a 3 : 1 mixture of two neighbours.
- **BASE_LEVEL, MAX_LEVEL.** `TEXTURE_BASE_LEVEL = b` makes level b the base image: λ is measured in units of
  that image (a strip with λ = 1 relative to level 0 is *magnified* relative to level 2, and shows level 2);
  `TEXTURE_MAX_LEVEL = m` ends the chain at `q = min(⌊log₂ size⌋ + b, m)`, and any larger λ selects q.
- **MIN_LOD, MAX_LOD** clamp λ before the level is chosen (equation 3.18).
- **Completeness (§3.8.13).** A texture is *mipmap complete* when levels `base … q` all exist, have the same
  effective internal format, and have the sizes `max(1, ⌊w / 2ⁱ⌋)`, and `base ≤ max`. A texture whose
  `TEXTURE_MIN_FILTER` needs mipmaps (everything but NEAREST and LINEAR, and the *default* is
  `NEAREST_MIPMAP_LINEAR`) and that is not mipmap complete is *incomplete*, and a lookup returns
  (0, 0, 0, 1) — without any GL error (§3.8.13, §3.9.2). Even a lookup that only needs level 0 returns black.
  Rows 11–12 break a texture in seven different ways and repair each with one call:

  | scenario | broken because | repaired by |
  |---|---|---|
  | 0 | level 3 (1×1) never specified | `glTexImage2D` level 3 |
  | 1 | level 1 is 3×3 instead of 4×4 | `glTexImage2D` level 1, 4×4 |
  | 2 | level 2 is RGB8, the others RGBA8 | `glTexImage2D` level 2, RGBA8 |
  | 3 | `BASE_LEVEL` 2 > `MAX_LEVEL` 1 | `MAX_LEVEL` = 3 |
  | 4 | level 3 missing | `MAX_LEVEL` = 2: level 3 becomes insignificant |
  | 5 | level 3 missing | `MIN_FILTER` = LINEAR: no mipmap is needed any more |
  | 6 | no image at all | a 1×1 level 0 (a one-level chain is complete) |

- **`glGenerateMipmap` (§3.8.10.5)** replaces levels `base + 1 … q` with a "filtered reduction" of the base level,
  "regardless of their previous contents", and leaves every other level (including the base level) alone.
  *No particular filter algorithm is required, though a box filter is recommended.* So the spec does **not**
  say that the 1×1 level of a 2×2 checker is the mean of its two colours. All three Mesa drivers do produce
  the mean (`note:` lines in the log), but the demo does not assert it. What it asserts is what the spec
  guarantees: derived level sizes (`textureSize`), unchanged base level, untouched levels beyond `MAX_LEVEL`, the
  texture being complete afterwards, and the error conditions. "Regardless of their previous contents" is tested
  directly: the same base is generated over two different sets of stale levels (red and green, or the ladder
  colours and red), and the derived levels must come out identical. Whether a uniform base gives uniform levels
  depends on the filter, so the largest deviation from the base colour is only printed (0/255 on all drivers here).
  Completeness after the call is proven through the base level: before the call the texture is incomplete and
  samples (0, 0, 0, 1) at every λ, level 0 included, so level 0 showing its texels means the chain is complete.
- **`glHint(GENERATE_MIPMAP_HINT, …)`** is a hint ("An implementation may ignore them entirely", §5.3); what the
  spec fixes is its state: initial value `DONT_CARE`, and it is a `glGetIntegerv` state (Table 6.27). Since the
  hint may change the filter quality, the independence test above is repeated under all three hint values. (Reading the hint back needed `GetParameter::GenerateMipmapHint`, meta-gl gap G2.)
- **Shader side.** `texture(sampler2D, vec2)` computes λ from the screen-space derivatives of its coordinate,
  and `textureSize(sampler, lod)` returns the size of level `BASE_LEVEL + lod`. The demo uses `textureSize`
  as an independent probe of the level sizes `glGenerateMipmap` produced (the size is encoded in the red and
  green channel). `precision highp sampler2D` is declared because samplers default to `lowp`.

## Validation

64 checks per run. Every cell is compared pixel by pixel (all 32×17 pixels of a cell) with either hard-coded level
colours or a CPU model that implements equations 3.16–3.25 and Table 3.22 for the very strips the shader draws.

- Rows 0, 1, 4, 5: exact level colours (tolerance 0 for NEAREST inside the level; ±1 for LINEAR, whose weights are
  convex, so only the 8-bit conversion can move the result). Rows 2, 3: a blend of levels k and k + 1 whose weight
  lies within ¼ of ½ (sanity bound, see below), ±1.
- Rows 6–9: the CPU sampler's range. Pixels within 0.1 texel of a texel boundary between different texels are skipped for
  NEAREST (a texel boundary is not robust); LINEAR pixels are checked everywhere against the min/max of the texels that
  can contribute.
- Rows 13, 14: `BASE_LEVEL`, `MAX_LEVEL`, `MIN_LOD`, `MAX_LOD`, with exact colours or blends and with the model.
- `glGenerateMipmap`: see above. A base level plus stale levels 1–2 is incomplete (black) before the call; the derived
  levels do not depend on the stale content (under each hint); a 37×21 base gives 18×10, 9×5, 4×2, 2×1, 1×1;
  `MAX_LEVEL = 2` makes the call replace levels 1 and 2 (identical over two different stale contents) and keeps
  levels 3–6 exactly; `RGBA8UI` and `DEPTH_COMPONENT24` bases → `INVALID_OPERATION`.
- Completeness: for each scenario the broken tile must be exactly (0, 0, 0, 1) and the twin exactly the colour of
  the intended level, then the *same* texture object is repaired in place with one call and re-drawn.
- State: the four mipmap filters read back; initial `BASE_LEVEL` 0, `MAX_LEVEL` 1000, `MIN_LOD` −1000, `MAX_LOD` 1000,
  `MIN_FILTER` `NEAREST_MIPMAP_LINEAR`; negative `BASE_LEVEL`/`MAX_LEVEL` → `INVALID_VALUE`; hint initial value and
  read-back after each of FASTEST, NICEST, DONT_CARE.

**What is a hard check and what is measured.** ES 3.0 bounds neither the precision of λ nor that of the filter
weights. The checks are therefore limited to what the equations guarantee:

- *Level choice.* For a strip, condition 2 of §3.8.10.1 pins the approximated ρ to `|∂u/∂x|`, so λ is fixed by the
  test. Every strip keeps λ at least ¼ from the boundaries where the chosen level(s) change, and the level choice is
  checked exactly.
- *NEAREST inside a level* returns one texel and is exact.
- *LINEAR inside a level* (§3.8.10.2): the bilinear weights are non-negative and sum to one for any α in [0, 1), so
  the result lies between the minimum and maximum of the contributing texels (taken over u ± 0.1 texel, the margin
  the NEAREST probes use). That range is checked.
- *Blending two levels.* The blend weight `frac λ` has no specified precision. The check is a sanity bound with the
  same margin as the level choice: the weight lies within ¼ of its ideal ½. A λ error that large would already
  break the k + ¼ / k + ¾ rows. Without this bound a filter that never blends would pass.
- ±1 whenever a result is a weighted sum: the conversion to 8 bits returns either of the two closest integers
  (§2.1.6.2).

The deviation from the ideal equations is printed in each check's detail. The measured blend weight is printed per
row (0.493–0.500 on radeonsi, llvmpipe, zink and ANGLE on lavapipe), and so is how many pixels of rows 6 and 7 differ
visibly (all of them here). None of these values is checked.

**Negative controls.** (1) The LINEAR_MIPMAP_NEAREST stripe cells (row 7, k = 1–3) are redrawn with the solid ladder
texture, whose level colour is exact on any implementation. The stripes range must reject it at more than a quarter
of the pixels (1530 of 1632 at 256×256). (2) The blend cell at λ = 2.5 is redrawn with
`MIN_FILTER` = NEAREST (a non-mipmapped filter always uses level 0): the colour becomes level 0's, and the
blend range must reject all pixels. (3) Changing the filter of row 3 to LINEAR_MIPMAP_NEAREST in the source makes
the two row-3 checks fail (done once during development). (4) Every incomplete texture is compared to
black and to its twin; a texture that was accidentally complete would fail the first, one that stayed incomplete
after the repair would fail the second.

## Findings

- **`glGenerateMipmap` on an `R32F` base level depends on extensions.** ES 3.0 §3.8.10.5 requires the base
  level to be both color-renderable and texture-filterable, otherwise INVALID_OPERATION. In core ES, `R32F`
  is never filterable, and it is color-renderable only in ES 3.2. Extensions amend those tables, though:
  `GL_OES_texture_float_linear` makes it filterable, and `GL_EXT_color_buffer_float` makes it renderable.
  Mesa exposes both on every context here, so the call is legal, and Mesa accepts it. (This was first
  suspected to be a Mesa deviation. It is not.) The demo computes the expectation from the context's
  extensions without using them, and asserts it strictly. `RGBA8UI` (not filterable) and `DEPTH_COMPONENT24`
  (not a color format) must always fail.

## Pitfalls

- The default `TEXTURE_MIN_FILTER` is `NEAREST_MIPMAP_LINEAR`. A texture with only level 0 and the default filter is
  incomplete and samples black — silently. Either set `LINEAR`/`NEAREST`, call `glGenerateMipmap`, or upload
  every level.
- Every level must have the same *effective* internal format and the right size. `glTexImage2D` accepts a wrongly
  sized level without a complaint; only the texture becomes incomplete.
- `glGenerateMipmap` does not promise any particular filter and the hint may be ignored; never rely on the exact
  contents of a generated level.
- λ = k + ½ is a tie for `*_MIPMAP_NEAREST`; implementations legitimately differ there.
- `texture()` derives λ from screen-space derivatives, which are only defined in uniform control flow (GLSL ES 3.00
  §8.8); the fragment shader chooses between `texture()` and `textureSize()` with a uniform for that reason.
- With `TEXTURE_BASE_LEVEL` above 0, `textureSize(s, 0)` is the size of the base level, not of level 0.
