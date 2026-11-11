# es30-depth-textures

**API introduced in:** OpenGL ES 3.0 (sized depth and depth-stencil textures, `TEXTURE_COMPARE_MODE`,
shadow samplers) · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

`es30-shadow-mapping` uses one depth format (`DEPTH_COMPONENT24`) for one job (a PCF shadow map). This demo goes
through depth textures *systematically*: all five formats that ES 3.0 defines (Table 3.14), how data gets into them
(uploads of Table 3.2, rendering, `gl_FragDepth`, clears), what exactly they store (fixed-point quantization versus
float), how they are read (plain depth with `TEXTURE_COMPARE_MODE = NONE`, or comparisons through all three shadow
sampler types and all eight compare functions), when they are *not* readable (texture completeness), and the stencil
half of `DEPTH24_STENCIL8` / `DEPTH32F_STENCIL8`, which ES 3.0 cannot sample but can use in the stencil test.

| Format | Upload type (Table 3.2) | Depth storage | Stencil |
|---|---|---|---|
| `DEPTH_COMPONENT16` | `UNSIGNED_SHORT` (also `UNSIGNED_INT`) | >= 16-bit unsigned normalized | - |
| `DEPTH_COMPONENT24` | `UNSIGNED_INT` | >= 24-bit unsigned normalized | - |
| `DEPTH_COMPONENT32F` | `FLOAT` | 32-bit float | - |
| `DEPTH24_STENCIL8` | `UNSIGNED_INT_24_8` (depth in bits 31..8) | >= 24-bit unsigned normalized | 8-bit uint |
| `DEPTH32F_STENCIL8` | `FLOAT_32_UNSIGNED_INT_24_8_REV` (float word + stencil word) | 32-bit float | 8-bit uint |

## The image

**Top: one panel per format** (16, 24, 32F in the first row, then 24_8 and 32F_8 when the window is wide enough
for 3 + 2; the arrangement adapts to the window and the panels are cropped, never stretched). Each panel is the
*depth texture* of the same 3D scene, a ground plane with six boxes, two of them far away, as seen from one camera.
Nothing in it is a color render: a shader reads the stored depth `d`, turns it back into an eye-space position, and
takes screen-space derivatives of that position to get a surface normal. The color encodes distance (amber near, violet and
teal far), the shading comes from those reconstructed normals, and thin lines mark every whole unit of eye distance
(the horizontal stripes on the ground). Where depth is coarse the normals become stair-stepped: on
`DEPTH_COMPONENT16` the far purple and teal boxes are covered in speckle, while 24-bit and 32F stay smooth. The two
depth-stencil panels are additionally tinted per box. The tint comes from the **stencil** component of the same
texture, which received the object id during the scene pass, applied with the stencil test (`EQUAL id`) in a framebuffer
that has the depth-stencil texture as its `DEPTH_STENCIL_ATTACHMENT`.

**Bottom: the compare-function grid.** Columns are `NEVER, LESS, EQUAL, LEQUAL, GREATER, NOTEQUAL, GEQUAL,
ALWAYS` (GL enum order). Rows are `sampler2DShadow` (the scene in `DEPTH_COMPONENT32F`), `sampler2DArrayShadow`
(layer 1 of a two-layer array, rendered from a second camera) and `samplerCubeShadow` (a cube map rendered from a
point inside the scene, shown as an equirectangular panorama). In the upper part of a cell, `Dref` is a "slicing
surface" whose eye distance grows from left to right, and a pixel is lit where the function passes against the
scene's depth. `LEQUAL` lights everything behind the slice (and the sky, whose depth is 1.0), and `GREATER` lights exactly the
complementary pixels. The **bottom band** of every cell uses `Dref = Dt`, the raw depth of the same texel read
through a second sampler object with compare mode `NONE`, so the band is fully lit for `LEQUAL`, `GEQUAL`, `EQUAL` and
`ALWAYS` and dark for the other four, as Table 3.23 says for `Dref == Dt`.

## How to run

```
./build/debug/bin/es30-depth-textures                       # window
./build/debug/bin/es30-depth-textures --headless --validate --capture out.png
./build/debug/bin/es30-depth-textures --headless --validate --width 1280 --height 400
```

The window must be at least 128 x 128 (smaller surfaces report `SKIP_UNSUPPORTED`). The scene is static, so there are no keys.

## Central calls

- `glTexImage2D(TEXTURE_2D, 0, DEPTH_COMPONENT16, 8, 2, 0, DEPTH_COMPONENT, UNSIGNED_SHORT, data)`, and likewise for the
  other formats with `UNSIGNED_INT`, `FLOAT`, `DEPTH_STENCIL / UNSIGNED_INT_24_8`,
  `DEPTH_STENCIL / FLOAT_32_UNSIGNED_INT_24_8_REV`; `glTexImage3D` for a depth `TEXTURE_2D_ARRAY`, six
  `glTexImage2D` calls for a depth cube map.
- `glTexStorage2D / glTexStorage3D` with the sized depth formats; `glFramebufferTexture2D(..., DEPTH_ATTACHMENT or
  DEPTH_STENCIL_ATTACHMENT, ...)`, `glFramebufferTexture2D` with the six `TEXTURE_CUBE_MAP_*` faces,
  `glFramebufferTextureLayer` for the array layers.
- `glClearBufferfv(DEPTH, 0, &d)` and `glClearBufferfi(d, s)` (meta-gl supplies the fixed `DEPTH_STENCIL, 0`).
- `glTexParameteri / glSamplerParameteri(TEXTURE_COMPARE_MODE, COMPARE_REF_TO_TEXTURE)` and
  `TEXTURE_COMPARE_FUNC` with all eight functions; the same texture on two units with different sampler objects.
- `glGenerateMipmap` on depth textures (an error), `glGetFramebufferAttachmentParameteriv` (`DEPTH_SIZE`,
  `STENCIL_SIZE`, `COMPONENT_TYPE`), `glStencilFunc / glStencilOp` against a depth-stencil texture attachment.
- GLSL: `texture(sampler2DShadow, vec3(uv, Dref))`, `texture(sampler2DArrayShadow, vec4(uv, layer, Dref))`,
  `texture(samplerCubeShadow, vec4(dir, Dref))`, `texture(sampler2D, uv)` on a depth texture, `gl_FragDepth`.

## Spec sections

- ES 3.0 §2.1.6.1 / §2.1.6.2: normalized fixed point. A b-bit value c means c / (2^b - 1) (eq. 2.1); a float f
  is stored as *one of the two nearest* b-bit integers to f * (2^b - 1), round-to-nearest preferred (eq. 2.3).
- ES 3.0 §2.13.1 (window z in fixed point), §3.9.2.3 (`gl_FragDepth` is clamped to [0, 1] and, for fixed-point
  buffers, converted like a window z; for float buffers it is only clamped), §4.2.3 (clear values: same conversion).
- ES 3.0 §3.7.2: Table 3.2 (valid combinations; note that `DEPTH_COMPONENT16` also accepts `UNSIGNED_INT`), Tables 3.8 /
  3.9 (packed depth-stencil layouts), §3.7.2.2 (component-count mismatch is `INVALID_OPERATION`), §3.7.2.3 (only the
  depth is converted to float; the stencil index is not).
- ES 3.0 §3.8.3: "For depth component groups, the depth value is clamped to [0, 1]"; depth and depth-stencil formats
  only for `TEXTURE_2D`, `TEXTURE_2D_ARRAY` and `TEXTURE_CUBE_MAP` (`INVALID_OPERATION` otherwise); Table 3.14
  (minimum resolutions); invalid combinations give `INVALID_OPERATION`. §3.8.5 applies the same rules to `TexSubImage*`.
- ES 3.0 §3.8.10.5: `GenerateMipmap` requires a color-renderable and texture-filterable format, so it is
  `INVALID_OPERATION` for every depth format (Appendix E.7: "Require that GenerateMipmap throw INVALID_OPERATION for depth
  textures (Bug 9481)").
- ES 3.0 §3.8.12: the stencil component of a depth-stencil texture is ignored by texturing. Sampling it needs
  `DEPTH_STENCIL_TEXTURE_MODE` (ES 3.1, see `es31-stencil-texturing`).
- ES 3.0 §3.8.13, last bullet: a texture whose format is a sized depth or depth-stencil format is **incomplete**
  when `TEXTURE_COMPARE_MODE` is `NONE` and the magnification filter is not `NEAREST` or the minification filter is
  neither `NEAREST` nor `NEAREST_MIPMAP_NEAREST`. With `COMPARE_REF_TO_TEXTURE`, `LINEAR` is allowed.
- ES 3.0 §3.8.15.1: `r = Dt` when the compare mode is `NONE`; Table 3.23 otherwise. "If the texture's internal format
  indicates a fixed-point depth texture, then Dt and Dref are clamped to the range [0, 1]; otherwise no clamping is
  performed." With non-`NEAREST` filters "r may be computed by comparing more than one depth texture value ... The
  details of this are implementation-dependent, but r should be a value in the range [0, 1] which is proportional to
  the number of comparison passes or failures."
- ES 3.0 §3.9.2.1: depth lookups produce the base color `Cb = (Rt, 0, 0, 1)`; "(0, 0, 0, 1) will be returned for a
  non-shadow sampler and 0 for a shadow sampler" when the texture is incomplete. Table 3.21 (cube face selection).
- ES 3.0 §4.1.5: with the depth test disabled the depth buffer is not updated. Depth writes therefore need
  `DEPTH_TEST` with `ALWAYS`.
- ES 3.0 §4.4.4.1: a depth-only format at `STENCIL_ATTACHMENT` or `COLOR_ATTACHMENTi` is attachment-incomplete;
  §6.1.13: `COMPONENT_TYPE` cannot be queried for `DEPTH_STENCIL_ATTACHMENT`, and a `DEPTH_STENCIL_ATTACHMENT` query
  with different depth and stencil objects is `INVALID_OPERATION`.
- GLSL ES 3.00 §8.8 (shadow forms: `Dref` is the last coordinate, the array layer the one before), §4.5.4 (no default
  precision for the shadow and array sampler types, and `sampler2D` / `samplerCube` default to `lowp` in fragment
  shaders, too little for depth).

## Shader concepts

- `probe.frag` performs one lookup per fragment of a 64x1 `RGBA32UI` target and stores the result with
  `floatBitsToUint`, so the CPU sees the exact float the shader got. `R32F` would be the obvious target, but it is not
  color-renderable in core ES 3.0 (§3.8.3.1 lists it as texture-only) and `RGBA32UI` is. Six samplers sit on six units
  because two sampler types on one unit is a draw-time error.
- `frag_depth.frag` writes `gl_FragDepth = u_depth[int(gl_FragCoord.x)]` into a 16x1 depth strip, which bypasses
  rasterizer interpolation, so the stored value depends only on the conversion rules.
- `depth_vis.frag` reconstructs eye-space positions from the depth texture (`texelFetch`, inverse projection) and shades
  with `cross(dFdx(p), dFdy(p))`.
- `compare_grid.frag` binds the same `DEPTH_COMPONENT32F` texture twice (raw sampler object and comparing sampler
  object) and builds the `Dref == Dt` band from the raw lookup.
- All samplers are declared `highp`.

## How it is validated

195 checks, all on observable results; everything except the window checks is offscreen and independent of the window size.

1. **Attachments and state.** Every format is a complete depth-only (or depth-stencil) attachment; `DEPTH_SIZE` meets
   Table 3.14 (32 for the float formats), `COMPONENT_TYPE` is `UNSIGNED_NORMALIZED` / `FLOAT`, `STENCIL_SIZE >= 8` and the
   texture is named at both `DEPTH` and `STENCIL` for `DEPTH_STENCIL_ATTACHMENT`; `COMPONENT_TYPE` of
   `DEPTH_STENCIL_ATTACHMENT` raises `INVALID_OPERATION`. Initial `COMPARE_MODE` / `COMPARE_FUNC` are `NONE` / `LEQUAL`.
2. **Uploads.** Six 8x2 textures, one per Table 3.2 combination, 16 chosen values each, read back through
   `sampler2D` with a `NEAREST` / `NONE` sampler. Float formats must be bit-exact (after the [0, 1] clamp: -0.25 and 1.75
   become 0 and 1, and 2.0 / -0.5 likewise in `DEPTH32F_STENCIL8`). Fixed-point formats must lie in
   `[floor(r), ceil(r)] / (2^b - 1) +- 2^-23` with `r = f * (2^b - 1)`, where `f = c / (2^n - 1)` is the uploaded value and `b`
   the **queried** `DEPTH_SIZE` (an implementation may allocate more bits than Table 3.14 requires). The interval is
   eq. 2.3's "one of the two nearest"; the 2^-23 is two float32 ulps in [0.5, 1) for the float conversions of eq. 2.1
   (the value is converted to float on upload and returned to the shader as a float). The other three components must be
   exactly `(0, 0, 1)`. `TexSubImage2D` accepts `UNSIGNED_INT` into `DEPTH_COMPONENT16` (the value is read back) and
   rejects `UNSIGNED_SHORT` into `DEPTH_COMPONENT24` (the texel is unchanged).
3. **Format errors.** Nine invalid `TexImage2D` combinations (wrong type for the format, `DEPTH_STENCIL` data for a
   depth-only format, depth-only data for a depth-stencil format, a two-component packed type with `DEPTH_COMPONENT`)
   give `INVALID_OPERATION`; a valid unusual one (`DEPTH_COMPONENT16 / UNSIGNED_INT`) gives none. A depth format on
   `TEXTURE_3D` is `INVALID_OPERATION`, the same call on `TEXTURE_2D_ARRAY` is fine.
4. **Quantization through rendering.** For every format a 16x1 strip is cleared to 0.8125 (`ClearBufferfv` or
   `ClearBufferfi`), and 15 values are written with `gl_FragDepth`: 0, 1, 0.5, 1/3, 0.1, 0.9, five values 0.3 / 0.7 of a 16-bit
   step off the 16-bit grid, one 0.4 off the 24-bit grid, 0.987654321, 1.5 and -0.25. Float formats store `clamp(f)`
   bit-exactly; fixed-point formats use the interval above. **Negative control:** when `DEPTH_COMPONENT16` really has 16
   bits, a 24-bit expectation fails at all five off-grid values, which shows the check sees the quantization. The printed strips show
   0.500007629 (16-bit) against 0.5 (32F).
5. **Render-to-depth against a CPU model.** The scene is planar, so a double-precision ray caster gives the exact
   window depth at every texel center. About 60 texels per target, each with a 3x3 neighbourhood on the same object,
   are compared for all five formats, both array layers (two cameras, `FramebufferTextureLayer`) and all six cube faces
   (face cameras with the up vectors of Table 3.21, sampled with the direction through the texel center, so that the
   rendered texel is also the one the sampler selects). Sky texels must be exactly 1.0. The spec gives no bound for
   depth interpolation, so the tolerance is a sanity bound of 1e-4 plus one step of the stored precision; the measured
   maximum error is printed (below 2e-7 for 24-bit and 32F, 7.6e-6 for 16-bit, i.e. half a 16-bit step).
6. **Comparisons, 2D.** For each uploaded texture and each function, `Dref` is placed below, at and above each
   texel's measured value (+-0.02, or +-0.5 for texels holding exactly 0 or 1, so that `Dref` leaves [0, 1]). The result
   must be exactly 1.0 or 0.0 as Table 3.23 predicts. `Dref == Dt` is used only where it is well defined, for every
   texel of a float format and for 0 and 1 of a fixed-point format: a fixed-point `Dt` like k / 65535 has no exact float.
   The lookups with `Dref` outside [0, 1] form separate checks per format: **clamped** for fixed point (`Dref = 1.5`
   against `Dt = 1` passes `LEQUAL`), **not clamped** for float (`1.5 <= 1.0` fails). Compare state comes from the
   texture parameters, and each function reads back with `glGetTexParameteriv`.
7. **Comparisons, array and cube.** A 4x2x2 `DEPTH_COMPONENT32F` array (layer 1 = 1 - layer 0, so a wrong layer
   gives wrong answers) is compared through **sampler objects** while the texture's own compare mode stays `NONE`. A
   cube with a different value per face is compared through **texture state** at 12 directions (6 axes, 6 off-axis).
   Raw `sampler2DArray` / `samplerCube` lookups are checked first.
8. **LINEAR comparisons.** 4x4 textures (columns 0-1 = 0.25, 2-3 = 0.75, in 32F and 16-bit) with `LINEAR` filters and
   `LEQUAL`. Only what the spec pins down is asserted: `Dref = 0.1` passes against *every* texel, so whatever
   texels an implementation combines, the result is 1.0; `Dref = 0.9` gives 0.0; every result is in [0, 1]. The
   values across the edge for `Dref = 0.5` are printed (radeonsi, llvmpipe and zink all give close to the bilinear weights 0.10, 0.30, 0.50, 0.70,
   0.90). With `NEAREST` the same lookups are exactly 0 or 1 from the selected texel (either neighbour at the u = 0.5 boundary).
9. **Completeness.** Immutable 2x2 `DEPTH_COMPONENT32F` and `DEPTH_COMPONENT24` textures holding 0.75, compare mode
   `NONE`, texture state only: `NEAREST/NEAREST` and `NEAREST_MIPMAP_NEAREST` (one immutable level) return
   `(0.75, 0, 0, 1)`; mag `LINEAR`, min `LINEAR`, `NEAREST_MIPMAP_LINEAR` and `LINEAR_MIPMAP_NEAREST` must return
   `(0, 0, 0, 1)`. With `COMPARE_REF_TO_TEXTURE`, `LINEAR` is complete (`ALWAYS` = 1, `NEVER` = 0). A mutable texture with
   only level 0 and `NEAREST_MIPMAP_NEAREST` is mipmap-incomplete: its shadow lookup must be 0 **even with `ALWAYS`**
   (control: 1 with `NEAREST`). `GenerateMipmap` is `INVALID_OPERATION` for all five formats and for that texture, which
   stays incomplete afterwards; on `RGBA8` it raises no error (control).
10. **Stencil.** The stencil component is read back with the stencil test itself. The texture is the
    `DEPTH_STENCIL_ATTACHMENT` of a scratch framebuffer with an `RGBA8` color buffer, and pass `b` draws with
    `StencilFunc(EQUAL, 1 << b, 1 << b)` and `ColorMask` selecting channel `b mod 4`, so two rounds recover all 8 bits. Checked:
    the uploaded stencil values of both packed types; the stencil of `ClearBufferfi(0.8125, 0xA5)`; a draw with depth test
    `LESS` at 0.55 against the strip's stored depths and `StencilOp(KEEP, INCR, REPLACE 0x3C)`, which must give 0x3C exactly where
    the stored depth is greater than 0.55 and 0xA6 elsewhere (depth and stencil of one texture working together); and the scene's object ids at
    the CPU probe texels. Attachment completeness: `DEPTH_COMPONENT24` at `STENCIL_ATTACHMENT` or `COLOR_ATTACHMENT0`
    is `FRAMEBUFFER_INCOMPLETE_ATTACHMENT`, `DEPTH24_STENCIL8` alone at `STENCIL_ATTACHMENT` is complete; a
    `DEPTH_STENCIL_ATTACHMENT` query after re-attaching only `DEPTH_ATTACHMENT` is `INVALID_OPERATION`.
11. **Window.** In every panel, a texel that is sky in a 5x5 neighbourhood (inside the visible, cropped area) shows
    exactly the sky color (+-1), and a ground texel does not. In the grid, every row's band pixel is lit exactly for `LEQUAL`,
    `GEQUAL`, `EQUAL`, `ALWAYS`. All cells of a row have the same integer size, so the same local pixel samples the same
    texel with the same `Dref` in every column. At 24 such pixels per row the relations of Table 3.23 must hold:
    `NEVER` dark, `ALWAYS` lit, `LEQUAL = !GREATER`, `GEQUAL = !LESS`, `EQUAL = !NOTEQUAL`, `LEQUAL = LESS | EQUAL`,
    `GEQUAL = GREATER | EQUAL`. `LEQUAL` must contain both lit and dark pixels (non-vacuity).

Verified at 128x128, 256x256, 640x480, 800x800, 333x777, 1280x400 and 150x600 on radeonsi, llvmpipe and zink.

### Known driver deviations (XFAIL)

- **All Mesa drivers** ignore the §3.8.13 depth-texture rule: a sized depth texture with compare mode `NONE` and a
  `LINEAR` (or `NEAREST_MIPMAP_LINEAR`, `LINEAR_MIPMAP_NEAREST`) filter is sampled (filtered) instead of returning
  `(0, 0, 0, 1)`. No exposed extension amends the rule: `OES_depth_texture` covers only the unsized `DEPTH_COMPONENT`
  format and says nothing about filters. The checks stay strict and are reported as XFAIL on "Mesa".
- **lavapipe** (the Vulkan driver under zink and under ANGLE in this matrix) returns 1.0 (the `ALWAYS` result) for a
  shadow lookup on a mipmap-incomplete depth texture instead of the 0 required by §3.9.2.1. radeonsi and llvmpipe
  return 0. The demo accepts the deviation only when `GL_RENDERER` is lavapipe-backed, that is when it contains both
  "Vulkan" and "llvmpipe" (`zink Vulkan ... (llvmpipe ...)`, `ANGLE (Mesa, Vulkan ... (llvmpipe ...)`), and then
  reports it as XFAIL on "zink" or "ANGLE". On every other renderer, including zink or ANGLE on a GPU Vulkan driver
  such as RADV, the check stays strict.
- **ANGLE** (Vulkan backend) does not clamp `Dref` to [0, 1] for `DEPTH_COMPONENT24` and `DEPTH24_STENCIL8`
  (`DEPTH_COMPONENT16` is clamped): `LESS` with `Dref = -0.5` against `Dt = 0` returns 1. §3.8.15.1 (ES 3.2 §8.20.1):
  *"If the texture's internal format indicates a fixed-point depth texture, then Dt and Dref are clamped to the range
  [0, 1]"*. Reported as XFAIL on "ANGLE" for the two 24-bit formats only.

## Pitfalls

- A depth texture with the **default** filters (`NEAREST_MIPMAP_LINEAR` / `LINEAR`) and compare mode `NONE` is
  incomplete in ES 3.0. Use `NEAREST`, or comparison. Many drivers filter it anyway, so code that relies on that is not portable.
- Default sampler precision in fragment shaders is `lowp` for `sampler2D` / `samplerCube` and *undefined* for the shadow
  and array types. Declare `highp` or depth may lose bits, and a missing declaration is a compile error.
- Using a shadow sampler while the compare mode is `NONE` (or a plain sampler while it is `COMPARE_REF_TO_TEXTURE`) gives
  undefined results. Keep two sampler objects when you need both views of one texture, as this demo does.
- `Dref` is clamped for fixed-point formats only. With `DEPTH_COMPONENT32F` a reference of 1.0001 *fails* `LEQUAL` against
  the cleared value 1.0, while with 24-bit it passes.
- `DEPTH_COMPONENT32F` with the standard projection buys almost nothing over 24-bit: near 1.0, float32 spacing (2^-24)
  equals the 24-bit step. Here the 24-bit and 32F strips agree closely, and on radeonsi bit for bit.
- Depth writes need `DEPTH_TEST` enabled (use `ALWAYS` to write unconditionally), and `glClearBufferfi` / depth clears
  respect `glDepthMask` / `glStencilMask`.
- `GenerateMipmap` never works on depth textures; there is no mipmapped shadow map without manual levels.
- The stencil half of a depth-stencil texture is invisible to `texture()` in ES 3.0. Read it with the stencil test, or
  use ES 3.1 `DEPTH_STENCIL_TEXTURE_MODE`.
