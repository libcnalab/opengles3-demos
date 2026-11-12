# es32-cube-map-array

**API introduced in:** OpenGL ES 3.2 · **Minimum context:** ES 3.2 · **Shaders:** GLSL ES 3.20 ·
**Run:** `es32-cube-map-array [--headless --validate] [--window]` (window keys: `a`/`d` orbit, `w`/`s` pitch, `q`/`e` zoom)

Spec numbering follows ES 3.2 (§8.5 Texture Image Specification, §8.13 Cube Map Texture Selection with Table 8.20, §8.14.3 Mipmapping, §8.17 Texture Completeness, §8.20 Texture Comparison Modes,
§8.23 Texture Image Loads and Stores with Table 8.24 "Layer numbers for cube map texture faces", §9.2.8 FramebufferTextureLayer) and GLSL ES 3.20 (§4.1.7, §4.7.3, §8.9).

## Purpose

A **cube map array** is an array of complete cube maps in a single texture object: `N` cubes, each with six square faces and a full mip chain. It exists for the same reason as a 2D array texture, one bind and one
sampler instead of `N`, but for directional lookups: an environment-map atlas (one reflection probe per object or per room), a set of point-light shadow cubes, a skybox per weather state. A shader indexes it with
one four-component coordinate, `vec4(direction, layer)`: the direction `(s, t, r)` selects face and texel exactly as for a `samplerCube` (Table 8.20), the fourth component `q` selects the cube.

The storage is *a 2D array texture with `6 N` layers*, called **layer-faces** in the specification. Layer-face `z` is cube `z / 6`, face `z mod 6`, with the faces in the order of the face targets (`+X -X +Y -Y +Z -Z`, Table 8.24).
That numbering is what `glTexStorage3D` (`depth = 6 * layers`), `glTexSubImage3D` (`zoffset`), `glFramebufferTextureLayer` (`layer`), the image coordinate (`ivec3(x, y, z)`) and `glBindImageTexture` (`layer`)
all use, while `textureSize` and `imageSize` report the number of *cubes*. This demo checks all of those against each other and against an independent CPU model.

## What the image shows

A 256 × 256 frame (frame 19 of the animation, time = frame/60 s, camera orbiting slowly):

- **Top 181 rows: five spheres, one draw call** (`glDrawElementsInstanced`, 5 instances, a perspective camera with a depth buffer). The fragment shader reflects the view ray on the sphere and looks up
  `textureLod(u_env, vec4(reflect(view, normal), float(layer)), 0.0)` with `layer = gl_InstanceID`: each sphere reflects **its own cube** of the array. The green channel of every texel encodes the layer, so the spheres
  go from dark to bright green; the red channel encodes the cube face and the blue channel the texel, which gives the blocky 4 × 4-per-face patchwork. At 256 × 256 the spheres stand in two rows,
  three above two (sphere 0 at the top left, reading order).
- **Bottom 75 rows: 30 tiles in a 10 × 3 grid**, tile `n` = layer-face `z = n`, in reading order, i.e. layer 0's six faces, then layer 1's, and so on (the red channel therefore cycles 24, 60, 96, 132, 168, 204
  within every group of six). Each tile is the 4 × 4 image of its layer-face, upright and unmirrored, looked up with the plain `texture()` function through a direction on the plane of the face.

On other surfaces the layout scales: the tile rows are `min(w/10, 3h/10/3)` pixels high, the ten tile columns partition the whole width (`[c·w/10, (c+1)·w/10)`,
so the tiles may be wider than high), and the spheres take the rest. The camera's vertical field of view is 45°, except that a portrait scene viewport keeps a 45°
*horizontal* field of view, and neither field of view exceeds 90° (very wide or very tall viewports), beyond which the perspective would stretch the outer spheres into ovals.

The spheres fill the scene viewport whatever its shape. `layout_of()` considers four arrangements of the five spheres (centres 1.95 apart, radius 0.8): one row of five, two rows (3 + 2),
three rows (2 + 2 + 1) and one column of five. For each it computes the camera distance at which the arrangement's bounding box fits the field of view with a 20 % margin (for the
perspective and the slow orbit), and it uses the arrangement with the smallest distance, i.e. the largest spheres on screen: two rows on a square window, one row on a wide strip
(1000 × 200), three rows or a column on a tall one. The rows are stacked along the up vector of the initial camera (pitched 16° down), so the arrangement faces the camera and a
column does not crowd together at the bottom of the screen. The `q` / `e` keys scale the distance. The sphere check derives its probes from the same camera and centres, so it
follows automatically.

**Minimum size.** `init()` reports SKIP in two cases. The first is a tile row under 8 pixels or a scene under 64 rows.
The second is a surface too small for the sphere check. The sphere check probes every second pixel in x and y, so
it gets about one ray per 4 pixels of a sphere's projected area. It needs at least 40 hits on every sphere, at least
a quarter of them comparable, and more than 20 compared probes on each of at least four faces. Whether a surface
meets that depends on the projected sphere sizes at the frame `validate()` shows (frame 19 by default, since the
camera orbits) and on the aspect ratio. So `init()` runs the same CPU ray model (`sphere_model()`) for that frame
and skips when the model cannot meet these counts. The check itself can then fail only when the picture
disagrees with the model. Because the minimum is computed, it is not one rectangle and not monotonic in every
direction. Since the spheres fill the scene viewport, the sphere check no longer decides the minimum in practice: on llvmpipe a sweep of
widths 80–140 against heights 85–140 (step 3) gives PASS everywhere except where the tile/scene guard skips (for example 80 × 80 and 90 × 90,
whose scene is 63 rows high), and 80 × 89, 95 × 200 and 1000 × 101 pass. The sphere-model guard stays, so a surface where it cannot
meet the counts still reports SKIP rather than a check that cannot fail.

## Central calls

`glTexStorage3D(TEXTURE_CUBE_MAP_ARRAY, levels, RGBA8, 4, 4, 6 * layers)`, **`glTexSubImage3D(..., zoffset, w, h, depth, ...)`** (one layer-face, one cube or a whole level per call),
`glTexImage3D` (mutable variant), `glGenerateMipmap`, `glFramebufferTextureLayer`, `glBindImageTexture`, `glGetTexLevelParameteriv`, `glGetIntegerv(TEXTURE_BINDING_CUBE_MAP_ARRAY)`.
In GLSL: `samplerCubeArray`, `isamplerCubeArray`, `usamplerCubeArray`, `samplerCubeArrayShadow`, `imageCubeArray`, `texture`, `textureLod`, `textureSize`, `imageLoad`, `imageSize`.

## Key ideas

- **`depth` counts layer-faces.** `glTexStorage3D(target, levels, format, w, h, depth)` for a cube map array needs `w == h` and `depth` a multiple of six (the command is defined by the `TexImage3D` pseudo-code of §8.18, and
  `TexImage3D` raises `INVALID_VALUE` otherwise, §8.5). The depth does *not* halve with the level (unlike a 3D texture), so a 4 × 4 × 30 texture has levels 4 × 4 × 30, 2 × 2 × 30 and 1 × 1 × 30:
  `glGetTexLevelParameteriv(TEXTURE_DEPTH)` returns 30 for every level.
- **`TexSubImage3D` addresses layer-faces.** "For cube map array textures, `zoffset` is the first layer-face to update, and `depth` is the number of layer-faces to update" (§8.6). The demo uploads level 0 one
  layer-face per call, level 1 one cube (6 layer-faces) per call and level 2 in a single call, then reads every one back.
- **The coordinate.** `q` is rounded to the nearest layer and clamped: `layer = max(0, min(d - 1, floor(q + 0.5)))` (GLSL ES 3.20 §8.9.2) with `d` the number of cubes. So `q = 1.2` selects cube 1, `q = 1.8` cube 2,
  `q = -3.7` cube 0 and `q = 100` the last cube. Pass `float(layer)`; do not rely on truncation.
- **Level of detail.** `textureLod(sampler, vec4, lod)` selects `NEAREST_MIPMAP_NEAREST` level `nearest(lod)` (equation 8.11): `lod <= 0` is level 0, `lod = 0.7` is level 1, anything above `levels - 1 + 1/2` the last level.
  In non-fragment shaders `texture()` has no derivatives and uses the base level (§11.1.3.5).
- **`texelFetch` does not exist for cube maps.** "Texel fetches cannot access cube map textures" (§11.1.3.2); GLSL ES 3.20 §8.9.2 lists no `gsamplerCube(Array)` form, so `texelFetch(samplerCubeArray, ivec3, int)` is a
  compile error. Use `imageLoad(imageCubeArray, ivec3(x, y, z))` (z = layer-face, Table 8.25) when you need raw texels.
- **Sizes.** `textureSize(sampler, lod)` and `imageSize(image)` return `(width, height, number of cubes)`, not the number of layer-faces.
- **Shadow cube arrays.** `float texture(samplerCubeArrayShadow, vec4 P, float compare)`: the fourth component of `P` is the layer, the depth reference is a **separate** argument (there is no room for it in `P`). The texture needs
  `TEXTURE_COMPARE_MODE = COMPARE_REF_TO_TEXTURE`; the comparison function is Table 8.22 with `Dref` on the left (`LEQUAL` is `Dref <= Dt`). There is no `textureLod` for it.
- **Completeness.** A cube map array is *cube array complete* if every slice is cube complete (square, same format); `glTexImage3D` guarantees that, so the usual rules remain: the default `MIN_FILTER`
  (`NEAREST_MIPMAP_LINEAR`) needs a full mip chain, otherwise lookups return `(0, 0, 0, 1)` (§11.1.3.5); `glGenerateMipmap` filters every layer independently.
- **Integer and depth formats** need `NEAREST` filtering; `glGenerateMipmap` fails with `INVALID_OPERATION` for formats that are not both color-renderable and texture-filterable (`RGBA8UI`).

## Shader concepts

- Opaque types other than `sampler2D` and `samplerCube` have **no default precision** (§4.7.3): `uniform highp samplerCubeArray u_env;` is required, `uniform samplerCubeArray u_env;` does not compile.
- `lookup.comp` is a bulk lookup harness: probes `(direction, q, lod, Dref)` come in a shader storage buffer, the raw result words go out in another, so the CPU compares the result of every lookup bit for bit
  (the sampler type, the lookup expression and the conversion to words are injected as macros).
- `tile.frag` contains `face_direction()`, Table 8.20 solved for `r` with `|ma| = 1`, so a tile pixel `(s, t)` samples the direction that lies exactly on the plane of the face; it is a different implementation from the CPU model
  `model_lookup()` (which goes from `r` to face and texel).
- `image_cube.comp` binds the same level twice: layered (`imageCubeArray`, `ivec3(x, y, z)`) and non-layered (`image2D`, one layer-face).

## Validation (`--validate`)

All references are computed on the CPU. 79 checks:

- **Test data (1).** All 630 texels (5 cubes × 6 faces × (16 + 4 + 1) texels of the three levels) have pairwise different colours: `R = 24 + 36 face`, `G = 24 + 40 layer`, `B = 16 + 14 texel + 8 level`.
  A lookup that returns the wrong face, layer, texel or level therefore returns a *different* colour, and the colour says which of the four was wrong.
- **State (6).** `TEXTURE_BINDING_CUBE_MAP_ARRAY` after `glBindTexture` and after binding 0; `TEXTURE_IMMUTABLE_FORMAT` / `TEXTURE_IMMUTABLE_LEVELS`; width, height, depth (30) and internal format of all three levels.
- **Layer-face numbering without sampling (1).** 90 framebuffer attachments (3 levels × 30 layer-faces, `glFramebufferTextureLayer(layer = z)`) are read back with `glReadPixels` and equal the texels that `glTexSubImage3D` stored at `zoffset = z`.
- **Images (7).** `imageLoad(imageCubeArray, ivec3(x, y, z))` returns the right texel for all 30 layer-faces at each of the 3 levels (3 checks) and `imageSize` is `(size, size, 5)` (3 checks);
  `glBindImageTexture(layered = FALSE, layer = n)` for `n = 0..29` binds exactly layer-face `n` (1 check, 480 texels).
- **Float lookups (10).** `textureLod` at the centre of every texel of every level, layer and face (630 lookups, each equal to the colour of that texel; the worst difference to `c/255` is 3e-8); six negative controls (mirrored
  t, mirrored s, transposed, face and layer swapped, layer + 1, level 0 instead of the lookup level): each wrong model disagrees with the GPU on a large share of the lookups (so the check would notice such a bug);
  layer coordinates -3.7 ... 100; fractional levels of detail -2 ... 9; 2400 random lookups (random direction, `q` from -1.5 to 6.5, `lod` from -0.5 to 3.5) of which about 1900 are compared. A lookup is skipped when it is within
  0.05 texel of a texel boundary, within 2 % of a face edge (where the major axis can tie) or within 0.05 of a rounding tie of `q` or `lod` (the specification allows two formulas there).
- **Integer lookups (2).** `usamplerCubeArray` on RGBA8UI and `isamplerCubeArray` on RGBA16I (negative components), 72 texel centres + 300 random directions each, exact words.
- **Shadow lookups (10).** `samplerCubeArrayShadow` on DEPTH_COMPONENT32F with each of the 8 functions of Table 8.22: 72 texels × `Dref = Dt - 1/256, Dt, Dt + 1/256` (exactly representable, so `EQUAL` and the boundary
  cases are decided exactly); a negative control that the 8 functions give different results; with `COMPARE_MODE = NONE` a plain `samplerCubeArray` returns `(Dt, 0, 0, 1)` with the exact depth.
- **textureSize (4).** All levels of the float array (4,4,5), (2,2,5), (1,1,5) and the three other sampler types.
- **Completeness and mipmaps (8).** `glTexImage3D` with depth 12; not mipmap complete -> `(0, 0, 0, 1)`; `MIN_FILTER NEAREST` makes it complete; `glGenerateMipmap` creates levels 1 and 2 with 12 layer-faces each,
  and every generated texel keeps the R (face) and G (layer) code of its own layer-face (so layers and faces were filtered independently); `glGenerateMipmap` on `RGBA8UI` gives `INVALID_OPERATION`.
- **Errors (16).** `glTexStorage3D`: depth 7, 4 × 8, depth 0 (`INVALID_VALUE`), 4 levels of a 4 × 4 texture, texture 0 (`INVALID_OPERATION`); `glTexImage3D` and `glTexStorage3D` on an immutable texture (`INVALID_OPERATION`);
  `glTexImage3D` depth 7, 4 × 8, border 1; `glTexSubImage3D` with `z + d > 30`, `zoffset = 30`, `x + w > width`, `zoffset = -1`, `level = -1` (`INVALID_VALUE`); the failed calls left layer-face 29 unchanged.
- **Compile rules (9).** A control shader with the four sampler types compiles; eight shaders do not: `texelFetch` on `samplerCubeArray` and on `samplerCube`, no precision, a 3-component coordinate, `texture(shadow, vec4)` without the
  compare argument, `textureLod` on the shadow sampler, `textureOffset` on a cube array, `samplerCubeArray` in `#version 310 es`.
- **Picture (4).** *Tiles:* 14520 pixels at 256 × 256 (more than half of the tile area at any size; texel boundaries excluded) of the 30 tiles equal the CPU texel of the right layer-face; negative controls: the models with t mirrored, s mirrored, transposed, face and layer swapped, or layer + 1 reproduce fewer than half of the tile pixels (measured: none, none, 25 % because the texels on the diagonal are symmetric, 17 %, none).
  *Spheres:* a ray-sphere intersection from the camera gives the point and normal, `reflect()` the direction, `model_lookup()` the colour; about 2400 pixels are compared at 256 × 256 (about 800 before the spheres were made to fill the scene) (every second pixel of the scene except grazing angles and
  anything that changes when the normal is tilted by ±0.012 rad, which covers the difference between the tessellated and the analytic sphere), all five spheres and all six faces appear (every sphere is
  hit by at least 40 probe rays and at least a quarter of them survive the filters, a bound that follows the sphere's size on screen), and a negative control (sphere `k`
  reflecting layer `k + 1`) must reproduce fewer than a quarter of the pixels (measured: none). Tolerance 0 (the colours are exact bytes; the sampled value `c/255` converts back to `c`).

## Spec ambiguities and how they were resolved

- **Rounding of the layer coordinate.** ES 3.2 §8.14.2 gives `l = clamp(RNE(r), 0, d_t - 1)` for array textures and a footnote allows `floor(r + 1/2)`; GLSL ES 3.20 §8.9.2 says `floor(layer + 0.5)`. They differ only at exact
  ties (`2.5`). The demo never probes within 0.05 of a tie.
- **"d is the depth of the texture array"** (GLSL ES 3.20 §8.9.2) is read as the number of cubes for a cube map array, consistent with `textureSize` and `imageSize`. `q = 100` therefore selects the last cube; all drivers agree.
- **`nearest(lambda)`** (eq. 8.11) has a "preferred" `ceil(l + 1/2) - 1` and an "alternative" `floor(l + 1/2)` form that differ only at ties; ties are avoided as above.
- **`TexStorage3D` and `depth % 6`.** ES 3.2 removed the explicit error from the §8.18 list as redundant (the function is defined through `TexImage3D`); the demo checks `INVALID_VALUE` on that basis.
- **`imageSize` of an `imageCubeArray`** is specified as the number of cubes (GLSL ES 3.20 §8.12); the demo follows that text and all drivers agree with it.
- **Section numbers.** The face-selection table is Table 8.20 and the layer-number table Table 8.24 in ES 3.2 (Table 3.21 in ES 3.0). The FramebufferTextureLayer text refers to "layer-face numbers in section 8.5.3";
  the definition is in the TexImage3D text of §8.5.

## Pitfalls

- `depth` is `6 * layers`, in every command that takes a depth, a `zoffset` or a layer number. `textureSize` and `imageSize` divide by six again; mixing the two is the classic off-by-six bug.
- The fourth coordinate is a *float*: write `float(layer)`, and remember that it rounds, so `layer + 0.5` selects the *next* cube. Out-of-range values clamp silently.
- `samplerCubeArrayShadow` takes the reference as a separate argument; a `vec4` coordinate is direction + layer. There is no `textureLod`, no offset form and no `texelFetch`.
- The face order within a layer is `+X -X +Y -Y +Z -Z`, not the order of the faces in a skybox image strip or in another API.
- Integer-format cube map arrays need `NEAREST` filters (otherwise the texture is incomplete and every lookup returns zeros), and cannot be mipmapped by `glGenerateMipmap`.
- `glTexStorage3D` makes the texture immutable: re-specifying with `glTexImage3D` is an error. Use `glTexSubImage3D`.
- `glFramebufferTexture2D` cannot attach a cube map array (it takes a 2D or face target); attach a single layer-face with `glFramebufferTextureLayer` (layer = `6 * cube + face`).
