# es30-normal-mapping

**Technique available since:** OpenGL ES 2.0 (it needs only textures and shaders) · **Minimum context here:** ES 3.0
(sized `R8`/`RGBA8` formats, `glFramebufferTexture2D` on any mip level) · **Shaders:** GLSL ES 3.00

## Purpose

Normal mapping fakes geometric detail. The mesh stays flat (a wall is two quads), but the lighting equation is
evaluated per fragment with a normal that is *read from a texture* instead of being interpolated from the vertices.
The bricks of this demo have bevelled edges and sunken mortar joints that exist only in the normal map, yet the light
that orbits in front of the wall rakes across them as if they were real. A second map, the **specular map**, says where
the surface is glossy: the bricks shine, the mortar does not.

The scene is a real 3D composition: perspective camera, depth buffer, back-face culling, a wall (two panels), a
rotating brick crate, a brick sphere and an animated point light (drawn as a small unlit sphere). Everything is
procedurally generated; there are no asset files. The same three textures are used on all three objects:

| unit | texture | internal format | content |
|---|---|---|---|
| 0 | albedo | `RGBA8` | brick colours per brick (random tone), darker near the edges, grey mortar |
| 1 | normal | `RGBA8` | tangent-space normal `n * 0.5 + 0.5`, derived from the brick height field |
| 2 | specular | `R8` | mortar ≈ 0, bricks 0.2-0.45, one brick in four "glazed" at 0.8-1.0 |

Each texture is 128×128 with a complete mip chain (8 levels, 128 … 1) and is sampled with trilinear filtering
(`LINEAR_MIPMAP_LINEAR` / `LINEAR`).

## Running and controls

```
./build/debug/bin/es30-normal-mapping --window                 # interactive, animated
./build/debug/bin/es30-normal-mapping --headless --validate --capture out.png
```

| key | effect |
|---|---|
| `N` | normal map on / off (off = the interpolated geometric normal: the bricks go flat) |
| `S` | specular map on / off (off = every texel is equally glossy: mortar shines too) |
| `H` | tangent handedness on / off (off = the bitangent ignores the sign `w`: the **right half of the wall** is lit wrongly, the bevels look inside out) |
| `F` | filter: trilinear / bilinear (no mip levels) / nearest — the difference shows at the grazing angles of the wall |
| `V` | view: lit / albedo / normal map / specular map (the raw texels, mapped onto the geometry) |
| `P` | pause the animation |
| `A`, `D` | orbit the camera |

Animation time is `frame / 60` s. The headless/validation run renders frame 0.

## Key ideas

**Tangent space.** A normal-map texel stores a normal relative to the surface: `(0, 0, 1)` (the familiar light-blue)
means "unperturbed", `x` leans along the texture's `u` axis, `y` along `v`. To light with it, the fragment shader needs
the basis `(T, B, N)` in which those coordinates are expressed: `T = ∂P/∂u` (the tangent), `B = ∂P/∂v` (the bitangent)
and `N` (the vertex normal). The perturbed normal is `normalize(T·n.x + B·n.y + N·n.z)`, a matrix product with
`mat3(T, B, N)` (GLSL ES 3.00 §5.4.2 constructors, §5.10 matrix operations).

**Generating the map from a height field** (`make_textures()`). The brick pattern is a height function (4 courses per tile,
alternate courses shifted by half a brick, 4 texels of mortar, a smoothstep bevel from the mortar to the brick face, plus
tileable value-noise grain). The normal at a texel is `normalize(-S·∂h/∂u, -S·∂h/∂v, 1)` with central differences and `S = 2`
(slopes up to about 45°), written as `rgb = n * 0.5 + 0.5`. The albedo and the specular map come from the same brick
classification (mortar / brick / which brick), so the three maps agree edge for edge.

**CPU tangent generation** (`compute_tangents()`). For every triangle the position edges `e1, e2` and UV differences give
`∂P/∂u = (e1·dv2 − e2·dv1)/det` and `∂P/∂v = (e2·du1 − e1·du2)/det`, `det = du1·dv2 − du2·dv1`. The per-vertex sum of
`∂P/∂u` is made perpendicular to the vertex normal (Gram-Schmidt) and normalised: that is `T`. The **handedness**
`w = sign(dot(cross(N, T), ∂P/∂v))` is stored in the fourth component of the tangent attribute, so a vertex carries
`T` and `w` (4 floats) and the bitangent is rebuilt as `B = cross(N, T) · w`. `w = +1` when `B = N × T` already points along `+v`;
`w = −1` on a mirrored texture.

**Why the handedness matters: the mirrored panel.** The wall is two panels meeting at `x = 0`. The right panel runs its
`u` axis backwards (`u = 4` at the seam on both sides, `u` decreases to the right), a mirrored texture. The
brick pattern is symmetric about the seam, so the picture is continuous, but the right panel's UV basis is left-handed
relative to its normal: `T = −x`, `B = +y`, `N = +z` and `N × T = −y`. Reconstructing `B` as `N × T` without `w` points `B` the
wrong way, the `y` component of every sampled normal is applied upside down, and bevels that should face the light face away
from it (press `H`). This is what a mirrored UV island, e.g. the other half of a symmetric model, does to a naive shader.
The sphere's UVs are not mirrored (`w = +1`), the crate's faces neither.

**Vertex shader** (`shaders/lit.vert`). `T` goes through the model matrix, `N` through the normal matrix (inverse transpose
of the model matrix's upper 3×3, `demo::math::normal_matrix`); `T` is re-orthogonalised against `N`; `B = cross(N, T) * w`.
`T`, `B`, `N` and the world position leave as varyings.

**Fragment shader** (`shaders/lit.frag`) lights in world space with Blinn-Phong and a point light:

```
n   = normalize(T·n_ts.x + B·n_ts.y + N·n_ts.z)          // n_ts = texel.xyz * 2 - 1
l   = normalize(light - P),  v = normalize(eye - P),  h = normalize(l + v)
att = 1 / (1 + 0.12 d + 0.04 d²)
facing  = smoothstep(0, 0.2, dot(N_geometric, l))         // a bumped normal must not light the back of a surface
diffuse = max(dot(n, l), 0) · facing
spec    = pow(max(dot(n, h), 0), 48) · facing · specular_map
colour  = albedo · (ambient + L · diffuse · att) + L · spec · att
```

The specular map is a *mask* on the highlight, not a colour: with it, the brick faces glint and the mortar stays matte;
with `S` (off) the constant 1.0 makes the mortar as glossy as the glazed bricks. The albedo is not multiplied into the
highlight (a non-metal's highlight is not tinted by the base colour).

**Mipmaps and trilinear filtering.** `glTexImage2D` is called once per level (0 … 7) with data built on the CPU: each
level is the 2×2 box average of the previous one. This is deliberate. `glGenerateMipmap` (§3.8.10.5) leaves the filter
unspecified, but the validation needs to know every byte of every level (see below). Averaging the *encoded* normal
bytes averages the normals (the encoding is affine), and the result is a *shorter* vector pointing along the mean
direction. The shader normalises after the TBN transform, so what survives is the mean slope: distant bricks lose their
bevels gradually instead of sparkling, which is the point of mipmapping a normal map. Only the direction matters here;
techniques that also use the shortened length (Toksvig, LEAN) are not part of this demo. With `LINEAR_MIPMAP_LINEAR`
the GL blends two levels: `τ = (1 − frac λ)·τ₁ + frac λ·τ₂` (equations 3.24/3.25), each level bilinearly filtered.
At the default camera one pixel covers about 2 texels of the wall, so it is shown at λ ≈ 1.1, between levels 1 and 2.

**Texture units.** The three textures live on units 0, 1, 2 for the whole run; `glUniform1i(u_albedo, 0)` etc. assign
*unit indices* to the sampler uniforms, not texture names. A sampler whose texture is incomplete reads `(0, 0, 0, 1)`
(§3.9.2), so a broken chain would turn the picture black rather than raise an error. The device limits are
`MAX_TEXTURE_IMAGE_UNITS` ≥ 16 (Table 6.32) and `MAX_COMBINED_TEXTURE_IMAGE_UNITS` ≥ 32 (Table 6.33); this demo needs 3.

## Central calls

`glTexImage2D` (per level, `RGBA8` and `R8`), `glPixelStorei(UNPACK_ALIGNMENT, 1)`, `glActiveTexture`, `glBindTexture`,
`glTexParameteri` (`TEXTURE_MIN_FILTER` / `TEXTURE_MAG_FILTER`), `glUniform1i` for the samplers, `glUniformMatrix3fv`
(normal matrix), `glUniformMatrix4fv`, `glVertexAttribPointer` (position, normal, uv and a 4-component tangent in one
interleaved buffer), `glDrawElements`.

## Specification

- ES 3.0 §3.8.1 Texture Objects, §3.8.3 Texture Image Specification, §3.7.1 Pixel Storage Modes
- ES 3.0 §3.8.10 Texture Minification (§3.8.10.1 scale factor and level of detail, §3.8.10.2 coordinate wrapping and texel
  selection, §3.8.10.4 Mipmapping), §3.8.11 Magnification, §3.8.13 Texture Completeness
- ES 3.0 §3.9.2 Shader Execution (texture access, incomplete textures, `MAX_TEXTURE_IMAGE_UNITS`), §2.12.9.4
- ES 3.0 §4.1.9 Dithering, §2.1.6.2 float to normalized fixed-point conversion
- ES 3.0 §4.4.2.4 Attaching Texture Images to a Framebuffer (any level), §6.1.13 Framebuffer Object Queries
- GLSL ES 3.00 §4.1.7 Opaque Types (samplers), §4.5.1 Range and Precision, §5.4.2 constructors, §8.5 Geometric Functions,
  §8.8 Texture Lookup Functions

## Expected image

A dark red-brown brick wall filling the view, lit by an orbiting warm light whose bulb is visible as a small cream
disc. Around the highlight every brick shows a lit bevel on the side facing the light and a darker one on the opposite
side, the glazed bricks flash, and the mortar stays grey and matte. In front of the wall hang a tilted brick crate (left)
and a brick sphere (right); the faces of the crate that point away from the light are dark (ambient only). In the
`--headless` capture (frame 0, 256×256) the light is at the upper right, and the crate is turned by 0.55 rad.

## Validation (`--validate`)

All references are computed on the CPU in double precision from the same equations. 33 checks (the exact count is printed
by the run); the interesting ones:

- **Probes with known surface data.** The camera is a plain pinhole (`eye`, `target`, `fov`); the orbit camera's 38 degree
  field of view spans the shorter side of the surface (horizontal in portrait), so the scene stays in view at any aspect
  ratio, while the trilinear view below keeps its exact pixels-per-unit scale. For a pixel centre the
  CPU intersects the ray `eye + t·dir` with the plane of a face (the faces are stored as origin + two edge vectors +
  normal, i.e. `P(s, t)`, UV = `(s·tile_u, t·tile_v)`). That gives the world position, the UV, the texel and the
  tangent frame `T = normalize(eu)`, `B = N × T · w`, `N` **analytically**, independent of the tangents that
  `compute_tangents()` produced. A pixel is a probe only if it is at least 0.12 world units from the panel border
  and the seam and **at least 0.2 texel from a texel border**. These passes use `NEAREST` for minification and
  magnification, so level 0 and texel `floor(u·128)` are selected exactly (§3.8.10.2); the probe margin makes the choice
  insensitive to sub-texel errors in the interpolated UV. 2454 wall probes (1257 on the normal panel, 1197 on the mirrored
  one) are compared per pass.
- **Four wall renders** (normal map + specular map on; normal map off; specular map off; handedness ignored) each match
  their CPU model at every probe. Measured maximum error: 0.62/255 on radeonsi, 0.50/255 on llvmpipe and zink.
- **The normal map changes the shading**: at 449 probes the CPU predicts a difference of ≥ 12/255 between the normal-mapped
  and the flat-normal render; the GPU shows it at all 449. **The specular map masks highlights**: at 71 mortar probes where
  the unmasked render has a highlight of ≥ 20/255 the map removes it on the GPU. **Handedness**: no effect on the normal
  panel (1257 of 1257 probes identical), and a visible one on the mirrored panel (144 of 144 sensitive probes).
- **Negative controls** (checks that must be able to fail): a CPU model that ignores the handedness sign mismatches the
  correct GPU render at 665 probes, all on the mirrored panel and none on the other; a model with the green channel
  inverted (DirectX-style normal map) mismatches at 1402 of 2454; the unmasked-specular model mismatches at 610. During
  development the shader itself was mutated (`n_ts.y` negated): 9 checks failed, including the green-inversion control that
  then matched.
- **The rotating crate** is validated the same way at its model matrix (angle `0.55 + 0.7·t`), 369 probes on its visible
  faces. This exercises the model matrix, the normal matrix and the transformation of the tangent frame. A CPU model that
  does not rotate the tangent frame mismatches at 337 of 369 probes.
- **Trilinear / mipmaps.** A face-on wall at exactly 64 pixels per world unit puts one pixel over two level-0 texels, so
  λ = 1. The albedo-only view makes the output equal to the filtered texel. 3904 pixels must equal a blend
  `(1 − f)·level 1 + f·level 2` of CPU bilinear lookups for one common `f ∈ [0, 1]`, fitted by least squares. The
  specification only bounds λ for this square mapping to `[1, 2]` (the approximation of ρ in §3.8.10.1 may range from
  `max{mu, mv}` to `mu + mv`), which is why `f` is fitted instead of assumed. Measured: `f` = 0.003 (radeonsi),
  0.005 (llvmpipe, zink); max residual 0.59 resp. 0.93 out of 255. Models using only level 0, 2 or 3 are off by
  33, 29, 36 out of 255: the check can fail. The camera is shifted a fraction of a pixel so the samples are not at texel
  centres of level 1 (there, a bilinear lookup of level 0 would coincide with level 1, and the control would be blind).
- **Uploads.** All 24 levels (3 textures × 8) are attached to a framebuffer object one by one, checked for completeness
  and read back: every byte equals the CPU chain, so sizes, formats (`R8` reads back as `(r, 0, 0, 1)`) and data
  are right; `FRAMEBUFFER_ATTACHMENT_*_SIZE` shows 8/8/8/8 for the normal map and 8/0/0/0 for the specular map. This
  also proves `UNPACK_ALIGNMENT` 1 was needed: the 2×2 and 1×1 `R8` levels have rows of 2 and 1 bytes.
- **State**: `TEXTURE_BINDING_2D` on units 0, 1, 2 are the three textures; `glGetUniformiv` returns 0, 1, 2 for the samplers;
  the device limits meet the ES 3.0 minima; `TEXTURE_MIN_FILTER` is `LINEAR_MIPMAP_LINEAR`; the CPU tangents of the wall
  and the crate equal `∂P/∂u` to 1e-6 rad with `w = (+1, −1)` for the two wall panels and `+1` on the crate; the sphere's
  tangents are unit length, perpendicular to the normal and right-handed away from the poles.

**Tolerances.** Lit colours: 1.25/255. ES 3.0 §4.1.9 requires the stored 8-bit value to be one of the two representable
values next to the incoming colour, so the quantisation error is below 1 (whatever the rounding or dithering); the shader's
float error is bounded by GLSL ES 3.00 §4.5.1 (correctly rounded `+ − ×`, 2 ULP `inversesqrt`, `3 + 2|x|` ULP `exp2`, < 2⁻²¹
absolute `log2`; `pow(n·h, 48)` is the largest term at about 2·10⁻⁵ relative = 0.005/255), and the CPU reference adds only
double rounding, so a quarter of an LSB of headroom is generous. Trilinear: 3/255. The spec gives no precision for filter
weights or the blend factor, so this is only a sanity bound; the measured residual is printed and is below 1/255.

**What is not compared pixel by pixel.** The sphere and the animated frame with trilinear filtering are rendered by the
same shader, but only checked by their inputs (tangent frames, the crate and wall pixels, the LOD pass). The validation
renders its passes into the default framebuffer and finally re-renders the normal frame so that `--capture` shows the scene.
`glGenerateMipmap` is not used and therefore not claimed.

## Pitfalls

- **Tangent handedness.** `B = cross(N, T)` without the stored sign is wrong on mirrored UVs. Storing `w` in `tangent.w`
  costs nothing and fixes it.
- **Normal map conventions.** This demo uses OpenGL's: `+Y` of the texture is the direction of increasing `v`. A normal map
  made for DirectX (`+Y` down) needs its green channel inverted; the negative control above is exactly that mistake.
- **Transforming the frame.** `T` and `B` are *directions on the surface*: they use the model matrix, not the normal matrix.
  After a non-uniform scale the transformed `T` is no longer perpendicular to the transformed `N`, hence the Gram-Schmidt step.
- **Interpolated vectors are not unit length.** `T`, `B`, `N` interpolate linearly across a triangle and must be normalised
  after use (the shader normalises `n`; the lighting vectors are normalised per fragment as well).
- **Mipmapped normal maps.** The averaged vectors are shorter than 1; either normalise after sampling (done here) or the
  lighting will be too dark. Don't generate the chain with a filter that treats the encoded bytes as colours in sRGB.
- **`R8` uploads need `UNPACK_ALIGNMENT` 1** unless every row is a multiple of 4 bytes: the 2×2 and 1×1 mip levels are not.
- **Sampler precision.** In a fragment shader the default `precision` of `sampler2D` is `lowp` (GLSL ES 3.00 §4.5.4); the
  shader declares `precision highp sampler2D;` so the texture lookups are not reduced to 8-10 bits on mobile GPUs.
- **Implicit derivatives.** `texture()` takes its level of detail from derivatives, which are only defined in uniform
  control flow. All three lookups happen before the first `if`.
