# es20-textures-2d

**API introduced in:** OpenGL ES 2.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

Sixteen cells, each sampling one texture configuration, cover the 2D texturing that ES 3.x
inherits from ES 2.0. Every probed pixel is compared with a CPU implementation of the sampling
equations of ES 3.0 §3.8.10, not with a stored image.

```
row 3 (top): RGBA4444   | RGBA5551   | unit 0 | unit 3 | incomplete
row 2:       LUMINANCE  | LUM_ALPHA  | ALPHA           | RGB565
row 1:       CLAMP [-1,2]| SubImage  | RGB align 4     | RGB align 1
row 0:       NEAREST    | LINEAR     | REPEAT [-1,2]   | MIRRORED [-1,2]
```

## Key ideas

- **Objects, units and samplers.** `glGenTextures`/`glBindTexture` bind a texture to the *active
  unit* (`glActiveTexture`). A `sampler2D` uniform holds a **unit index** (`glUniform1i`), never a
  texture name. Bindings are per unit: `GL_TEXTURE_BINDING_2D` reports the binding of the active unit.
- **Sampling parameters are texture state** in ES 2.0. The demo sets them with all four setter
  variants (`glTexParameteri/f/iv/fv`) and reads them back. ES 3.0 sampler objects
  (`es30-sampler-objects`) separate this state from the texture.
- **NEAREST vs LINEAR.** NEAREST takes texel `wrap(⌊u·w⌋)`. LINEAR blends the four texels around
  `u·w − 0.5` with weights `α = frac(u·w − 0.5)` (ES 3.0 eq. 3.23). A 4×4 texture stretched over 64
  pixels shows the difference clearly.
- **Wrap modes** (Table 3.22) only matter outside [0,1]: REPEAT tiles, MIRRORED_REPEAT reflects
  every other tile, CLAMP_TO_EDGE repeats the edge texels.
- **Unpack alignment.** With `GL_UNPACK_ALIGNMENT = 4` (the default), each row of a 3-texel RGB
  image (9 bytes) is read from a 12-byte stride. The padding bytes are skipped. With alignment 1
  the rows are tight. Both uploads produce identical texels. A mismatch here is the classic cause
  of skewed textures.
- **Legacy unsized formats** reach the shader as in Table 3.24: LUMINANCE → (L,L,L,1),
  LUMINANCE_ALPHA → (L,L,L,A), ALPHA → (0,0,0,A). Packed 16-bit types are unsigned-normalized per
  field (c / (2ⁿ−1)).
- **Completeness.** The default `MIN_FILTER` is `NEAREST_MIPMAP_LINEAR`, so a 2×2 texture with only
  level 0 is incomplete and samples as (0,0,0,1) (§3.8.13, §3.9.2). A 1×1 texture would be complete:
  its mip chain *is* level 0. This was a mistake in the first version of this demo, and the
  validation caught it.

## Validation

33 checks:
- Grids of probes in the filter and wrap cells. NEAREST probes skip texel boundaries and must match
  exactly. LINEAR is allowed ±3/255, because filter-weight precision is implementation-defined.
- Exact texels for the sub-image and alignment cells.
- Format conversions (±1 for the packed types).
- Per-unit bindings, parameter queries, `glIsTexture` life cycle, and `glGetIntegerv(GL_UNPACK_ALIGNMENT)`.
  The last one was not expressible before meta-gl fix G1.
- Five defined errors: border ≠ 0, negative width, sub-image out of range, a 2D texture bound
  to `TEXTURE_CUBE_MAP`, and unpack alignment 3.

## Pitfalls

- Sampling a texture that is not complete gives black, with no GL error. Check `MIN_FILTER` and the mip chain.
- Implicit-LOD `texture()` calls inside non-uniform control flow have undefined derivatives. The
  fragment shader samples both textures and selects the result afterwards.
- The capture PNG contains alpha. ALPHA and LUMINANCE_ALPHA cells look light in image viewers that
  composite over white.
