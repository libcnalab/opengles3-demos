# es30-texture-swizzle-lod

**API introduced in:** OpenGL ES 3.0 (`TEXTURE_SWIZZLE_*`, `textureLod`/`textureGrad`/`textureSize`/`texelFetch`/texel
offsets, `MIN_`/`MAX_PROGRAM_TEXEL_OFFSET`) · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 ·
**Run:** `es30-texture-swizzle-lod [--headless --validate] [--width W --height H]` (any surface size)

## Purpose

ES 3.0 lets an application change *what a lookup returns* (texture swizzle) and lets a shader decide *which level* it
reads (explicit level-of-detail functions, fetches, offsets). Both are easy to get subtly wrong, because the rules
interact: the swizzle is applied to a colour that does not exist in the stored texels, and the level numbers that
shaders pass are counted from `TEXTURE_BASE_LEVEL`. This demo draws one small cell per rule and compares every pixel
of every cell with a CPU model.

`es20-mipmaps` already verifies the implicit-LOD side (level selection from screen-space derivatives, the four mipmap
filters, `BASE_LEVEL`/`MAX_LEVEL` and `MIN_LOD`/`MAX_LOD` as they affect `texture()`, `glGenerateMipmap`). This demo
does not repeat any of it; it only takes the same texture parameters and asks what they do to the *explicit* functions.

## The image

The cells are drawn into an offscreen **contact sheet**, an `RGBA8` texture attached to a framebuffer object: 16×16-pixel
cells, 16 to a row, row 0 at the bottom, 256 pixels wide and as high as the rows in use (9 rows, 256×144). Sections start on
a new row; the first cell of a row is at the left.

The window shows that sheet **stretched over the whole surface** by a second pass (`present.frag`): window pixel `(x, y)`
shows sheet texel `(x · 256 / w, y · 144 / h)`, computed in integer arithmetic and read with `texelFetch`, so every window
pixel is one exact sheet texel (no filtering), and the window is opaque. Row 0 is at the bottom of the window and of the PNG
capture. The cells themselves keep their 16-pixel design at every window size on purpose, because what they test depends
on pixel geometry (the implicit-LOD cell on screen-space derivatives over 16 pixels, the swizzle cells on 4 pixels per
texel, the fetch and offset cells on one texel per pixel). Scaling the cells with the window would change those experiments; drawing
them offscreen and stretching the finished picture does not.

| row | cells | content |
|---|---|---|
| 0 | 0–15 | swizzle: R8 (×3 swizzles), RG8 (×3), legacy LUMINANCE_ALPHA, legacy ALPHA (×2), RGBA8 (×4), RGBA8 swizzled (B,G,R,A) through `texelFetch`, `textureLod`, `textureProj` |
| 1 | 16 | the same swizzled RGBA8 texture on unit 3 with a LINEAR sampler object: smooth, still swizzled |
| 2 | 17–29 | `textureLod` on a 64×16 ladder whose 7 levels are solid colours: one solid cell per case |
| 3 | 30–40 | `textureGrad` (7 cells), `textureProjLod` (2), `textureProj` (2): solid level colours |
| 4 | 41–50 | `textureSize(lod)`: the cell's red byte is the width, the green byte the height |
| 5 | 51–61 | `texelFetch`: 3 single-pixel cells on the ladder and 8 full-level images of a 13×13 patterned chain |
| 6–8 | 62–101 | eight lookup functions × five offsets, each a 13×13 or 6×6 pixel image |

## Key ideas

- **Swizzle is applied to the texture base colour (§3.9.2.1, Table 3.24).** After filtering, the GL builds
  `Cb` and `Ab` from the base internal format — R8 gives `(R, 0, 0)` and alpha 1, RG8 `(R, G, 0)` and 1, LUMINANCE_ALPHA
  `(L, L, L)` and `A`, ALPHA `(0, 0, 0)` and `A` — and *then* picks, for each output component, one of
  `RED`, `GREEN`, `BLUE`, `ALPHA` (of that colour) or the constants `ZERO` and `ONE`. So swizzling an R8 texture with
  `(ALPHA, GREEN, BLUE, RED)` yields `(1, 0, 0, r)`: `ALPHA` finds the implicit 1, `GREEN` and `BLUE` the implicit 0s.
  The classic uses are visible in the sheet: `(R, R, R, ONE)` shows a single-channel texture as gray, `(ALPHA, ALPHA,
  ALPHA, ONE)` does the same for an ALPHA texture, and an RG8 texture with `(R, R, R, G)` is exactly what a
  LUMINANCE_ALPHA texture returns. The legacy luminance/alpha formats still exist in ES 3.0 but are not
  colour-renderable and have no sized form; the RG8 + swizzle route replaces them. Cell 5 and cell 6 are the same
  texels with the two methods, and the demo checks that they are the same picture.
- **Swizzle is texture state, not sampler state.** Table 6.9 lists the swizzle in the per-texture state, Table 6.10 (per
  sampler object) lists only filters, wraps, LOD range and comparison. A sampler object therefore cannot carry it
  (meta-gl's `SamplerParameter` has no swizzle enumerator either). Cell 16 binds the swizzled texture together with a
  LINEAR/CLAMP sampler: the sampler changes the filtering (the cell is smooth, compared against a bilinear CPU model),
  the texture's swizzle still applies.
- **Swizzle applies to every lookup.** §2.12.9.2 says a texel fetch is point-sampled "but the remaining steps of texture
  access are still applied", so cells 13–15 run `texelFetch`, `textureLod` and `textureProj` on the same swizzled texture
  and all produce the same colours as `texture()`.
- **Explicit LOD ignores the geometry.** The cell quad maps a 64×16 texture onto 16×16 pixels (4 texels per pixel in s,
  1 in t), so `texture()` selects level 2 by itself (cell 17: ρ lies in [4, 5], λ ∈ [2, 2.32], §3.8.10.1). `textureLod(0.0)`
  on the same quad returns level 0, `textureLod(3.0)` level 3, a `lod` beyond the last level is clamped to it (eq. 3.23),
  and a fractional `lod` follows eq. 3.23 (nearest: rounded, 2.75 → level 3) or 3.24/3.25 (linear: 2.75 → ¼ level 2 +
  ¾ level 3). The explicit λ is still clamped by `TEXTURE_MIN_LOD`/`MAX_LOD` (eq. 3.18) and counted from `BASE_LEVEL`
  and `MAX_LEVEL` — those are the interactions that are new relative to the mipmap demo.
- **`textureGrad` picks a level from derivatives you supply.** With only one non-zero partial derivative every allowed
  approximation of ρ (§3.8.10.1: `max{mu, mv} ≤ f ≤ mu + mv`) equals the exact value, so the level is a property of the
  test: ρ = `g · 64` for an s-gradient and `g · 16` for a t-gradient (s scales with the *width* of the base level, t with
  the *height*, equation 3.19). The cells use a larger x than y footprint and vice versa (ρ is the maximum), and a base
  level of 2 (where the texture is 16×4) to show that the gradient is measured in texels of the base level. Gradients
  that are non-zero in both axes are deliberately avoided: ρ may then be approximated within a factor of two.
- **Projection.** `textureProj(P)` divides the coordinates by the last component (`P.z` for a `vec3`, `P.w` for a
  `vec4`; the third component of a `vec4` is ignored for a 2D sampler). `P = (s q, t q, q)` is therefore the same
  lookup as `(s, t)` for every `q`; the cells use q = 4, 0.5 and 0.25. Without the division the `textureProj` cells
  would show levels 4 and 0 instead of 2.
- **`textureSize(sampler, lod)`** returns the size of level `BASE_LEVEL + lod` (§2.12.9.3). On the 64×16 ladder the
  levels are 64×16, 32×8, 16×4, 8×2, 4×1, 2×1, 1×1: halved, rounded down, never below 1 per dimension. With
  `BASE_LEVEL = 2`, `textureSize(0)` is already 16×4. The size is returned in the colour as bytes `(w, h, 0, 255)`.
- **`texelFetch` is not a filtered lookup (§2.12.9.2).** Its `lod` is added to the base level; MIN/MAX_LOD do *not* clamp
  it ("texel fetches do not support LOD clamping or any texture wrap mode"): with `MIN_LOD = 5` a fetch of `lod 0`
  returns level 0, where `textureLod(0.0)` would return level 5. Fetches outside the level, below the base level or above the
  last level (`q`) are undefined; so is a fetch of a level other than the base level when the minification filter is
  NEAREST or LINEAR. The demo stays inside the defined range. `MAX_LEVEL` changes how many levels there are (`q`), not
  the numbering.
- **Offsets are constant expressions.** `textureOffset(s, uv, ivec2 offset)` needs a compile-time constant, but the legal
  range is implementation-dependent: `MIN_PROGRAM_TEXEL_OFFSET ≤ −8`, `MAX_PROGRAM_TEXEL_OFFSET ≥ 7` (Table 6.31). The demo
  queries both, pastes them into `shaders/offset.frag` as `#define OFF_MIN`/`OFF_MAX` (two lines inserted after
  `#version`) and compiles five lookups per function: `(1, 2)`, `(MIN, MIN)`, `(MAX, MAX)`, `(MIN, MAX)`, `(MAX, MIN)`.
  Mesa reports −32 and 31. The offset enters equation 3.19 (`u = w_t s + δ_u`) *before* the wrap mode is applied and in
  texels of the level actually sampled, so a REPEAT lookup at `(MIN, MIN)` shows the level rotated by 32 texels, and a
  CLAMP_TO_EDGE lookup shows mostly edge texels. On a 13×13 level an offset of ±32 is not a multiple of 13 (nor is ±31
  one of 6 on the 6×6 level 1), so no case degenerates to the identity — the demo checks that.
- **`texelFetchOffset` and the range rule.** `P + offset` must lie inside the level (§2.12.9.2). The demo therefore
  fetches the *mirrored* texel: `P = (size − 1 − x, y) − offset`, so `P + offset` is always in range while `P` itself
  runs far outside it. The image is the level flipped horizontally for every offset, and a driver that ignored the
  offset or applied it with the wrong sign would read other texels.
- **A macro generates the five lookups.** `shaders/offset.frag` defines `DEFINE_LOOKUP(NAME, OFFSET)` with a
  line-continued body (GLSL ES 3.00 §3.2: backslash-newline is a line continuation) and instantiates it five times
  with different constant offsets. Which one runs is chosen by a uniform, and so is the lookup function, which keeps
  every implicit-derivative call in dynamically uniform control flow (GLSL ES 3.00 §8.8).

## Validation

120 checks per run.

- **Window.** Every window pixel is compared with the sheet texel the presentation's integer mapping names (RGB exact,
  alpha 255). Every other check reads the sheet itself (`glReadPixels` from the sheet framebuffer), so the checks below are
  independent of the window size.
- **Cells.** Every pixel of every cell is compared with the CPU model of that cell (about 18 000 pixel comparisons).
  Swizzle cells: `swizzle(base_color(format, texel))` with the exact Table 3.24 mapping. LOD cells: the level colour. Blend
  cells: `(1 − f)·c₁ + f·c₂`. Size cells: `(w, h, 0, 255)`. Fetch and offset cells: the chain texel after
  `wrap(x + δx)` (REPEAT, CLAMP_TO_EDGE) or the mirrored texel. The sampler cell: bilinear weights with CLAMP_TO_EDGE
  (Table 3.22, equation of §3.8.10.2) on swizzled texels, since a per-component selection commutes with linear filtering.
- **Probes avoid ambiguity instead of loosening tolerances.** Swizzle cells map 4 texels onto 16 pixels, so a pixel
  centre is at least 0.125 texel from a boundary. Offset and fetch cells map one texel to exactly one pixel (pixel centre
  = texel centre). LOD cells use solid levels and avoid every λ at a rounding tie (x.5); gradient cells have one
  non-zero derivative (see above).
- **Tolerances.** ±1 for everything that is a stored 8-bit value moved through a shader (a float → 8-bit conversion
  returns one of the two nearest integers, §2.1.6.2); ±3 for the bilinear cell (filter weights have implementation-defined
  precision; real hardware has ≥ 6–8 fractional bits); `1 + |c₂ − c₁|/16` for the two level-blend cells (the blend weight
  `frac λ` has at least 4 fractional bits in every implementation). The measured maximum deviation is 0 in every cell on llvmpipe and zink and on radeonsi
  except 1 in the bilinear cell and in the 3/4-blend cell.
- **State.** Initial swizzle `(RED, GREEN, BLUE, ALPHA)`; the six legal values round-trip through `glTexParameteri` and
  `glTexParameteriv`; an illegal value (`LINEAR`, `REPEAT`) raises `INVALID_ENUM` (ES 3.2 §8.10; ES 3.0 leaves the code
  to the general rule) and leaves the old value; `BASE_LEVEL`/`MAX_LEVEL`/`MIN_LOD`/`MAX_LOD` round-trip; the offset limits
  meet the minimums of Table 6.31.
- **Negative controls.** (1) The cell swizzled `(A, R, G, B)` compared with the model of `(B, G, R, A)` must differ in
  more than 3/4 of its pixels. (2) The `(MIN, MIN)` cell of `textureOffset` compared with the *unshifted* level must differ
  in more than 3/4 of its pixels (after the check that no offset is a multiple of the level size). (3) The cell
  `textureLod(3.0)` must differ from level 0 *and* from level 2 in every pixel, and the implicit `texture()` cell must
  not be level 0 while `textureLod(0.0)` is not level 2 — an explicit lod does not follow the geometry. (4) Green
  of the R8 cell is 0 without the swizzle and equal to red with `(R, R, R, ONE)`.
  (5) Once during development the shader's `textureOffset` was changed to use the negated offset: the eight REPEAT and
  CLAMP_TO_EDGE cells that depend on the sign failed (the two (MIN, MIN) cells did not fail: −MIN = 32 is outside the legal
  range, so that lookup is undefined, and on this driver it happened to return the texels of −32).

## Pitfalls

- The swizzle is applied to `Cb`/`Ab`, so `ALPHA` of an RGB or R8 texture is **1** and `GREEN`/`BLUE` of an R8 texture are
  **0**, not whatever bytes happen to be in memory.
- Swizzling is per texture; a sampler cannot change it, and `glTexImage2D` does not reset it.
- `textureLod(s, uv, 3.0)` with `BASE_LEVEL = 2` reads level **5**. Always count from the base level.
- `texelFetch` of a level above the last level `q`, of a coordinate outside the level, or of a level other than the base
  level when `MIN_FILTER` is NEAREST or LINEAR, is undefined (§2.12.9.2). It is not an error and nothing detects it.
- A texel offset has to be a constant expression and has to be inside `[MIN, MAX]_PROGRAM_TEXEL_OFFSET`; the range is only
  guaranteed to be −8…7. Anything else is undefined for textureOffset, and a compile error for a non-constant offset.
- Implicit-derivative functions (`texture`, `textureProj`, `textureOffset`) must not be called in non-uniform control
  flow. The demo selects its functions with uniforms, which are dynamically uniform.
