# es30-texture-3d-and-arrays

**API introduced in:** OpenGL ES 3.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 ·
**Run:** `es30-texture-3d-and-arrays [--headless --validate]`

## Purpose

OpenGL ES 2.0 has two-dimensional textures and cube maps. ES 3.0 adds two more texture targets whose images have
a third dimension:

* **`TEXTURE_3D`**, a volume of `width x height x depth` texels, addressed by three normalized coordinates
  `(s, t, r)` and filtered in all three directions. The slices are *not independent*: `LINEAR` filtering
  blends the two nearest slices (trilinear), `TEXTURE_WRAP_R` decides what happens outside `[0, 1]`.
* **`TEXTURE_2D_ARRAY`**, a stack of `layers` independent 2D images. The third coordinate is **not normalized**: it
  is a layer *index*. It is rounded to the nearest layer and clamped, and filtering never mixes layers.

The demo shows each consequence of that sentence and checks it with pixels, not with return values. It also covers
the calls that fill such textures (`glTexImage3D`, `glTexSubImage3D`, `glCopyTexSubImage3D`, with
`UNPACK_IMAGE_HEIGHT` and `UNPACK_SKIP_IMAGES`), the shader functions that read them (`texture`, `texelFetch`,
`textureSize` on `sampler3D` and `sampler2DArray`) and rendering into one slice or layer
(`glFramebufferTextureLayer`).

## What the image shows

A contact sheet of 8 x 8 panels of 32 x 32 pixels (panel 0 is at the lower left, numbers grow to the right and then
upwards; in the capture the last row of panels is at the top). Every panel is one draw call that shows one
slice, layer or lookup. The textures are tiny (4 x 4 texels per slice, each texel its own color) so that every
texel is 8 x 8 pixels and can be told apart by eye.

| Panels | What is shown | Spec |
|---|---|---|
| 0, 1 | 3D `NEAREST`: `r = 0.375` selects slice 1, `r = 0.875` slice 3 (`k = floor(r * depth)`) | §3.8.10.2 |
| 2, 3 | 3D `LINEAR` across slices: `r = 0.25` (depth coordinate 1.0, exactly between slices 0 and 1: 50% each), `r = 0.3` (70% / 30%). Colors differ from both slices | eq. 3.22 |
| 4, 5, 6 | 3D `NEAREST` with `r = 1.375` (outside the texture) and `WRAP_R` = `REPEAT` (slice 1), `MIRRORED_REPEAT` (slice 2), `CLAMP_TO_EDGE` (slice 3) | Table 3.22 |
| 7 | 3D `LINEAR`, `REPEAT`, `r = -0.05`: blends the LAST slice (70%) and the first (30%) | eq. 3.22, Table 3.22 |
| 8, 10 | array `NEAREST`: layer coordinate 2.0 gives layer 2, coordinate 1.7 *rounds* to layer 2 (a `floor` would give 1) | §3.8.10.2 |
| 9, 11, 14, 15 | array `LINEAR` with layer coordinates 1.3, 2.6, 1.4, 1.6: still ONE layer each (1, 3, 1, 2), filtered only inside it | §3.8.10.2 |
| 12, 13 | array layer coordinates -3.2 and 9.6 are clamped to layer 0 and 3, although the texture's `WRAP_R` is its initial `REPEAT` (it applies to 3D textures only) | §3.8.10.2 |
| 16, 17 | `texelFetch` on a `sampler3D` slice and a `sampler2DArray` layer; the textures are set to `LINEAR` and the result is still the exact texels | GLSL ES 3.00 §8.8 |
| 18-21 | `glTexSubImage3D`: a gray volume with a 2 x 2 x 2 box and a whole slice replaced | §3.8.5 |
| 22-24 | `glCopyTexSubImage3D` into a 3D texture: slice 2 got a 4 x 3 rectangle copied from a framebuffer, slices 0 and 1 are untouched gray | §3.8.5 |
| 25-27 | the same into an array: layer 1 changed, layers 0 and 2 untouched | §3.8.5 |
| 28-35 | `UNPACK_IMAGE_HEIGHT` / `UNPACK_SKIP_IMAGES`: panels 28/29 (height 6, skip 1), 30/31 (the same memory with height 0: different texels), 32/33 (as 28/29 through `glTexSubImage3D`), 34/35 (array, skip 2) | §3.7.2, §3.8.3 |
| 36-41 | `textureSize(sampler3D, 0..2)` and `textureSize(sampler2DArray, 0/1/3)`; the size is written into the color (`20 x size`) | GLSL ES 3.00 §8.8, ES 3.0 §2.12.9.3 |
| 42-45 | `texelFetch` from level 0 and from level 1 of mipmapped 3D and array textures | GLSL ES 3.00 §8.8 |
| 46-49 | a 16 x 16 x 4 3D texture: slices 1 and 3 were **rendered** into through `glFramebufferTextureLayer`, slices 0 and 2 were never attached and are still gray | §4.4.2.4 |
| 50-53 | a 16 x 16 x 4 array texture: layers 0 and 2 rendered, 1 and 3 still gray | §4.4.2.4 |
| 54-57 | the largest allowed index: layers `MAX_ARRAY_TEXTURE_LAYERS-1` and 0 of a 1 x 1 x N array, slices `MAX_3D_TEXTURE_SIZE-1` and 0 of a 1 x 1 x N 3D texture, rendered through `glFramebufferTextureLayer` (green and red) | §3.8.3, §4.4.2.4 |
| 58-63 | unused (background) | |

## Central calls

`glTexImage3D`, `glTexSubImage3D`, `glCopyTexSubImage3D` (target `TEXTURE_3D` or `TEXTURE_2D_ARRAY`; there is no
`CopyTexImage3D`, because the framebuffer is two-dimensional), `glPixelStorei(UNPACK_IMAGE_HEIGHT / UNPACK_SKIP_IMAGES)`,
`glFramebufferTextureLayer(target, attachment, texture, level, layer)` (no `textarget`: the texture already says
what it is), `glTexParameteri(TEXTURE_WRAP_R)`, and the limits `MAX_3D_TEXTURE_SIZE`, `MAX_ARRAY_TEXTURE_LAYERS`.

## Key ideas

- **3D filters across slices, arrays do not.** The same 4 x 4 x 4 numbers behave differently depending on the target.
  With `LINEAR` a 3D lookup reads eight texels (equation 3.22); an array lookup reads four, all from the layer
  `floor(r + 0.5)` (panels 9 and 14: the layer coordinates 1.3 and 1.4 both give layer 1 with no trace of layer 2).
- **The depth coordinate of a 3D texture is normalized, the layer coordinate of an array is an index.** The demo's
  shader passes `r = (k + 0.5) / depth` for a 3D slice k, but `r = k` for an array layer k.
- **Layer rounding.** `NEAREST` uses round-to-nearest-even (`RNE`), `LINEAR` may use `floor(r + 0.5)`; the spec's
  footnote calls them "nearly equivalent". They differ only at exact halves, so the demo never probes at `.5`
  (1.3, 1.4, 1.6, 1.7, 2.6, 9.6, -3.2 are all far from a half).
- **Wrapping applies to texels, clamping to layers.** `TEXTURE_WRAP_R` exists only for 3D textures; the layer of an
  array is always clamped to `[0, layers-1]` even though the array's `TEXTURE_WRAP_R` (initially `REPEAT`) is ignored.
- **`texelFetch` is exact.** Integer coordinates, a level, no filter and no wrapping (panels 16, 17, 42-45). Its
  companion `textureSize(sampler, lod)` returns `(w, h, d)` of the level, where `d` of an array is the *layer count*,
  which does not shrink with the level (panels 39-41).
- **`UNPACK_IMAGE_HEIGHT` / `UNPACK_SKIP_IMAGES` select the sub-volume of client memory.** The pointer is advanced by
  `SKIP_IMAGES x (rows per image x row stride)` and each 2D image occupies `IMAGE_HEIGHT` rows (the texture's `height` when
  it is zero). `glTexImage2D` ignores `UNPACK_SKIP_IMAGES`.
- **`glFramebufferTextureLayer` attaches exactly one 2D image.** Rendering touches only that layer (panels 46-53), the
  framebuffer's `FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER` reports it, and a layer that does not exist makes the
  attachment `FRAMEBUFFER_INCOMPLETE_ATTACHMENT` (section 4.4.4.1).
- **Default filters need mipmaps.** Every texture is created with `TEXTURE_MIN_FILTER = NEAREST_MIPMAP_LINEAR`; a texture
  with only level 0 would be incomplete and sample as black. The demo sets `NEAREST` or `LINEAR` for both filters
  before each panel.

## Shader concepts

`sampler3D` and `sampler2DArray` have **no default precision** in GLSL ES 3.00 (section 4.5.4 predeclares only
`sampler2D` and `samplerCube`), so `sample.frag` declares `precision highp sampler3D;` and
`precision highp sampler2DArray;`. The two sampler uniforms are bound to different units (0 and 1): two samplers
of different types on one unit make the draw call `INVALID_OPERATION`. A uniform `u_mode` switches between `texture`,
`texelFetch` and `textureSize` (a uniform condition is dynamically uniform, so the implicit-derivative `texture`
calls stay legal). `pattern.frag` renders `base + gx * x + gy * y` from `gl_FragCoord`, which is what is rendered into the layers.

## Validation (`--validate`)

117 checks in `validate()` (the runtime adds its own: context, version, `glGetError` after `init()`, after the last frame and after `shutdown()`). None of them only checks that a call returned.

- **Panels (58 checks).** Every pixel of every panel (576 to 1024 probes) is compared with a CPU implementation of
  equation 3.22, Table 3.22 and the array rules of section 3.8.10.2 (`sample_volume`, `sample_array`). `NEAREST`
  probes within 0.1 texel of a texel boundary are skipped (where an implementation may round either way); the rest must be exact.
  Texel data is 8-bit and the lookups need no arithmetic, so **tolerance 0** for `NEAREST`, `texelFetch`, `textureSize`,
  the uploads and the copies. **`LINEAR`: +-3/255.** ES 3.0 does not specify the precision of the filter weights. Hardware keeps
  at least 6 (usually 8) fractional bits; the trilinear error for texel steps of 60 per axis is at most 3 x 60/64 = 2.8, so 3 plus the 0.5 for
  rounding the 8-bit result covers every conforming implementation (the three Mesa drivers deviate by 0.25 at most). **Rendered
  slices: +-1/255**, the 8-bit conversion of a computed color (section 2.1.2).
- **Negative controls (11 checks).** The same panels compared with the *wrong* model must fail: panels 2 and 3 do not match
  either neighbouring slice alone (so "no filtering across slices" is detectable); the three wrap modes are pairwise
  different; `LINEAR` + `REPEAT` is not the `CLAMP_TO_EDGE` result; the array at 1.3 does not match what a 3D texture
  would return (the cross-layer blend); layer 1.7 does not match layer 1 (a `floor`); 9.6 is not layer 0 (no wrapping);
  `IMAGE_HEIGHT` 0 and 6 give different images; the rendered slices are not gray while the unattached ones
  are *exactly* gray. A manual mutation (replacing `RNE` by `floor` in the CPU model) makes panel 10 fail, as intended.
- **Read back through the other path.** `glReadPixels` from a framebuffer with each slice and layer attached returns the same
  texels (8 checks), and the copy results of `glCopyTexSubImage3D` are compared exactly with the framebuffer that was copied (read back
  before the copy, so the shader's rounding does not matter).
- **State.** `MAX_3D_TEXTURE_SIZE` and `MAX_ARRAY_TEXTURE_LAYERS` >= 256 (Table 6.28); `TEXTURE_BINDING_3D` and
  `TEXTURE_BINDING_2D_ARRAY`; initial and changed `TEXTURE_WRAP_R`; initial, set and rejected `UNPACK_IMAGE_HEIGHT` and
  `UNPACK_SKIP_IMAGES` (read through `glGetIntegerv`, which needed meta-gl gap G1); `FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE`, `OBJECT_NAME`,
  `TEXTURE_LEVEL` and `TEXTURE_LAYER`.
- **Defined errors (19 `glGetError` checks, plus 3 framebuffer-completeness checks).** `glTexImage3D` with a depth, width or layer count over the maximum, a negative depth, a nonzero border, and
  target `TEXTURE_2D`; `glTexSubImage3D` / `glCopyTexSubImage3D` outside the image; `glFramebufferTextureLayer` with a 2D texture, a layer
  >= the maximum, a negative layer, an out-of-range level, and with the default framebuffer bound; negative `glPixelStorei` values;
  attachments of a layer that does not exist are `INCOMPLETE_ATTACHMENT`.
- **Known driver deviation.** On Mesa **zink** `glCopyTexSubImage3D` from a framebuffer that was painted with scissored `glClear` calls and not flushed returns stale
  texels (a `glFlush` before the copy fixes it). ES 3.0 section 2.1 says that commands are processed in the order they are received and that
  pixel reads (which includes the framebuffer reads of `CopyTexSubImage*`) see all previous commands. The check keeps the strict expectation and is reported as XFAIL on zink only;
  the main copy test is not affected because it paints its source by drawing and reads it back.

## Pitfalls

- A 3D texture and an array texture look identical in memory and in `glTexImage3D`; the *target* decides the filtering.
- Layer coordinates are floats: `texture(sampler2DArray, vec3(uv, 2.0))` selects layer 2, `texture(sampler3D, vec3(uv, 2.0))` is far outside the volume.
- Mipmap filters make a single-level 3D or array texture incomplete (black). Set `TEXTURE_MIN_FILTER` or upload all levels.
- `glFramebufferTextureLayer` with a layer beyond the texture raises no error if it is below the implementation limit; the framebuffer is
  just incomplete (`glCheckFramebufferStatus`).
- Sampling a texture while one of its layers is attached to the *draw* framebuffer is a feedback loop with undefined results; the
  demo samples only with the default framebuffer bound.
- `UNPACK_IMAGE_HEIGHT` and `UNPACK_SKIP_IMAGES` are pixel-store state that persists: reset them after uploading, or the next upload reads
  from the wrong place. `glTexImage2D` ignores `SKIP_IMAGES`.
- `glCopyTexSubImage3D` copies a single 2D rectangle into one slice or layer (`zoffset`), never a volume.
