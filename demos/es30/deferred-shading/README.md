# es30-deferred-shading

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 (`#version 300 es`) · **Run:** `es30-deferred-shading [--window] [--headless --validate]`

## Purpose

A real 3D scene lit by 48 coloured point lights that move over time, rendered with *deferred shading*: the geometry is
rasterized once into a **G-buffer**, and lighting is computed afterwards, per pixel, from what the G-buffer holds. The cost of
the lighting no longer depends on how many objects overlap a pixel, and a light only costs something where it actually shines.

Everything is generated in `main.cpp`: a tiled floor, ten pillars, eight rotated boxes, six small cubes, seven spheres, a torus and
one small emissive "bulb" sphere per light (81 draws per pass). The camera orbits slowly; the animation clock is `frame / 60 s`,
so the headless run is deterministic (it validates frame 29, `t = 29/60 s`).

## The passes

| Pass | Target | What happens |
|---|---|---|
| 1 Geometry (`gbuffer.vert/frag`) | G-buffer FBO | One draw per object. One fragment shader writes **three colour outputs** (multiple render targets) and the depth buffer writes the fourth plane. |
| 2a Lighting, full-screen (`light_fullscreen.frag`) | light buffer (RGB10_A2) | One triangle from `gl_VertexID`; every pixel reads the G-buffer, rebuilds its world position and loops over all lights of a uniform block. |
| 2b Lighting, light volumes (`volume.vert/frag`) | light buffer | A base pass (ambient + emissive) and then **one low-resolution sphere per light** (`glDrawElementsInstanced`, `gl_InstanceID` selects the light). Front faces are culled, there is no depth test, and each covered pixel adds that light with `glBlendFunc(ONE, ONE)`. |
| 3 Composite (`composite.frag`) | default framebuffer | `1 - exp(-L * exposure)` and gamma 1/2.2, or one G-buffer plane as a debug view. |
| (Forward reference, `forward.frag`) | RGB10_A2 + depth renderbuffer | The same scene and lights with the same equations evaluated per fragment, no G-buffer. Only the validation and the `F` key use it. |

The default is the light-volume path; `V` switches to the full-screen pass. Both produce the same image.

### G-buffer layout (and why)

| Attachment | Format | Content |
|---|---|---|
| `COLOR_ATTACHMENT0` | `RGBA8` | albedo.rgb, specular strength in alpha |
| `COLOR_ATTACHMENT1` | `RGB10_A2` | world-space normal as `n * 0.5 + 0.5`, 10 bits per component, alpha = 1 |
| `COLOR_ATTACHMENT2` | `RG8` | `shininess / 255` in red, `emissive / 3` in green |
| `DEPTH_ATTACHMENT` | `DEPTH_COMPONENT24` **texture** | window-space depth, sampled by the lighting pass |

The formats are limited by what a *core* ES 3.0 context lets you render to. ES 3.0 Table 3.13 marks `RGBA8`, `RGB10_A2`, `RG8`
(and the sized integer formats, `SRGB8_ALPHA8`, `RGB565`, ...) as colour-renderable, but `RGBA16F` and `RGBA32F` as texture-only; they
become renderable only with `EXT_color_buffer_float` / `EXT_color_buffer_half_float` or in ES 3.2. This demo never relies on an
extension, so the normal goes into the one format that has more than 8 bits per channel and is renderable: `RGB10_A2`.
No position is stored; it is rebuilt from the depth texture. The planes may have different formats as long as they have the same size.
The light buffer is also `RGB10_A2` and stores *radiance / 2*, so values up to 2 survive and a `1 - exp(-x)` tone map can compress them.

### Position from depth

The pixel centre gives `ndc.xy = gl_FragCoord.xy / size * 2 - 1` and the depth texture gives `ndc.z = 2 d - 1`. The perspective
projection with near `n` and far `f` maps the eye distance `z` along the view axis to `ndc.z = ((f + n) - 2 f n / z) / (f - n)`, which is
inverted to `z = 2 f n / ((f + n) - ndc.z (f - n))`. The shaders then use the camera basis instead of a matrix inverse:
`p = eye + z * (forward + right * tan(fovy/2) * aspect * ndc.x + up * tan(fovy/2) * ndc.y)`.

### Lighting model

Per light: `x = 1 - d^2 / r^2` (zero at the radius), attenuation `x^2`, Lambert diffuse `max(n.l, 0)` and a Blinn-Phong highlight
`specular * max(n.h, 0)^shininess`, multiplied by the light colour; plus a hemispherical ambient term and `albedo * emissive`.
Because the attenuation reaches **exactly zero at the light's radius**, a light volume of that radius loses nothing. The volume is a
16 x 8 UV sphere scaled by `1 / inradius` so that the *polyhedron* circumscribes the sphere (the demo checks that it is convex and that
every face plane is at least at the radius). Front faces are culled so that the back faces cover every pixel the volume can
touch even when the camera is inside it; without the culling each light would be added twice.

## Central calls

| Call | Role here |
|---|---|
| `glTexStorage2D(RGBA8 / RGB10_A2 / RG8 / DEPTH_COMPONENT24)` | immutable storage of the planes |
| `glFramebufferTexture2D`, `glCheckFramebufferStatus` | attach three colour textures and a depth *texture*; completeness with mixed formats |
| `glDrawBuffers({C0, C1, C2, NONE})` | route fragment outputs `layout(location = 0..2)` to the attachments (ES 3.0 4.2.1) |
| `glClear`, `glClearBufferfv(COLOR, 1, ...)` | clear all planes; overwrite only draw buffer 1 (negative control) |
| `texelFetch` on `sampler2D` | read G-buffer planes by integer pixel coordinate, no filtering |
| `glGetUniformBlockIndex`, `glUniformBlockBinding`, `glBindBufferBase`, `glBufferSubData` | the light block (two `vec4[64]`, std140) shared by four programs and rewritten every frame |
| `glDrawElementsInstanced` + `gl_InstanceID` | one sphere per light |
| `glEnable(BLEND)`, `glBlendFunc(ONE, ONE)`, `glCullFace(FRONT)` | additive accumulation of the light volumes |
| `glReadBuffer`, `glReadPixels` | read the G-buffer planes back for validation |
| `glGetFramebufferAttachmentParameteriv`, `glGetActiveUniformsiv`, `glGetFragDataLocation` | state queries of the configuration |

## Shader concepts

- **Multiple outputs:** `layout(location = N) out vec4 ...` in the fragment shader, GLSL ES 3.00 4.3.8.2.
- **Sampler precision:** in GLSL ES 3.00 a `sampler2D` is `lowp` unless declared otherwise, and the precision of a texture lookup
  follows it. The depth and normal samplers are therefore declared `highp`.
- **`flat` integer varying:** `volume.vert` passes `gl_InstanceID` to the fragment shader as `flat out int`; integer varyings must be `flat`.
- **Uniform block with `std140`:** the block has no instance name; `layout(binding = ...)` for blocks does not exist in 3.00, so the binding point
  is set with `glUniformBlockBinding`.
- **Attribute-less drawing:** the full-screen triangle is built from `gl_VertexID` (`(gl_VertexID << 1) & 2`, `gl_VertexID & 2`).
- **Bit operations for validation:** `composite.frag` mode 7 packs the 24-bit depth into RGB8 with shifts and masks, because `glReadPixels` in ES 3.0
  reads colour buffers only.

## Expected image

A dark courtyard seen from above: a grey-blue checkerboard slab, a ring of tall pillars, boxes and glossy spheres, a golden torus around a silver sphere,
and 48 small glowing bulbs in all colours drifting over the floor and washing it with coloured pools of light (magenta and orange on the right,
cyan and green on the left). Spheres show soft highlights, the floor shows specular streaks under nearby lights.
Keys (windowed mode): `1` final, `2` albedo, `3` normals, `4` material (specular, shininess, emissive), `5` depth, `6` world position rebuilt from
depth (a 1-unit grid), `7` mosaic of albedo / normals / depth / final; `V` light volumes or full-screen pass; `F` forward-rendered reference;
`SPACE` pause; `A`/`D` orbit, `W`/`S` zoom; `+`/`-` light count 8..64.

## Validation (`--validate`, headless frame 29, 79 checks)

Everything is compared with an **independent CPU model** in double precision: a ray caster (ray-plane, ray-oriented-box, ray-sphere,
ray-cylinder, sphere-traced torus) built from the same object list, and a CPU evaluation of the lighting equations.

1. **Configuration (state queries):** the G-buffer FBO is complete; `COMPONENT_TYPE` is `UNSIGNED_NORMALIZED` for all three planes; component sizes are 8/8/8/8, 10/10/10/2
   and 8/8/0/0 (a missing component reads 0, ES 3.0 6.1.13); the depth attachment is a texture with 24 bits; `DRAW_BUFFER0..3 = C0, C1, C2, NONE`;
   `glGetFragDataLocation` returns 0, 1, 2; the std140 block has size 2048, offsets 0 and 1024 and array stride 16, as the C++ struct assumes.
2. **Meshes (CPU):** all 6732 non-degenerate triangles are counter-clockwise from outside; the light-volume polyhedron is convex and circumscribes the sphere.
3. **Seven probe pixels** (floor on a light and a dark tile, a box top, a box side, a sphere, a pillar, a light bulb) and a background pixel are chosen on the CPU: *stable*
   pixels whose neighbours 3 pixels away hit the same object, face and checker cell, away from silhouettes and creases, not seen edge-on. At each probe:
   - G-buffer albedo + specular (RGBA8) and shininess + emissive (RG8) equal the material within **1 step** (ES 3.0 2.1.6.2 allows either of the two closest values);
   - the normal plane (RGB10_A2, read back as RGBA/UNSIGNED_BYTE) equals the analytic normal within **0.01** (one 10-bit step 2/1023 plus one 8-bit readback step 2/255; note that
     0.5 is not representable, so the floor normal reads back as 1/255 off) and **0.03** on spheres and cylinders (plus the tessellation of the 48-segment meshes);
   - the depth texture (read through the packing mode) converts to an eye distance within **2 mm** of the analytic distance on planes and within `0.006 r + 2 mm` on curved
     surfaces (facet sag); the measured planar deviation is 3e-5 to 1.5e-4 world units. The spec gives no precision for rasterized depth, so this is a stated sanity tolerance;
   - the world position rebuilt from depth (shown as `fract(p + 0.25)`) is within **2 / 3 steps** of 1/255 world unit of the analytic hit (measured 0.2 to 0.6 steps);
   - the final colour equals the CPU lighting result within the per-pixel budget below.
   The background pixel has an all-zero albedo plane, depth exactly 1.0 (`0xFFFFFF`) and the sky gradient colour.
4. **Whole image against the forward reference:** all 65536 pixels of the deferred image (full-screen pass; light volumes; and light volumes against the full-screen pass) must be
   within a **per-pixel, per-channel budget**. Both images end in the same composite pass, so they can only differ by quantization before it. ES 3.0 2.1.6.2 defines the
   float to fixed-point conversion as "one of the two closest values", i.e. up to **one step** of the format: the RGB10_A2 light buffer (step `2/1023` of radiance; the forward and
   the full-screen pass convert once, the volume path converts the running sum after the base pass and after each of the `k` lights touching the pixel, because blending is done in
   floating point and converted back, 4.1.7), the G-buffer inputs (the CPU re-evaluates the lighting with each stored input shifted by one step, all eight sign combinations for the normal,
   and sums the radiance changes), and the 8-bit rounding of the two final images (1 step; 1.5 against the CPU model). A radiance error `e` becomes a display error `f(L + e) - f(L)`
   with `f(L) = (1 - exp(-1.15 L))^(1/2.2)`, which is steepest in the dark, so the budget follows each pixel's radiance (about 2 steps in bright areas, a few tens in the darkest).
   The measured worst case is 0.6 to 0.7 of the budget (radeonsi: at most 6 steps, mean 0.24; llvmpipe/zink: at most 12 steps, mean 0.77, because they truncate when converting back,
   which the spec allows).
5. **Negative controls** (each must fail in the intended way):
   - (a) The CPU picks the (pixel, light) pair where removing one light changes the display colour most (about 110 of 255 steps) and that light's colour is zeroed in the uniform
     block: that pixel then equals the CPU model *without* the light (within the budget) and misses the forward image by more than twice the budget;
   - (b) `glEnable(BLEND)` left off in the volume pass: at least 5 % of the pixels must leave the budget (measured: about 60 %);
   - (c) `glClearBufferfv(COLOR, 1, ...)` replaces the normal plane by "up": the plane reads `(128, 255, 128)` at every probe, the albedo plane is untouched, and at least 5 % of the
     pixels leave the budget (measured: about 16 %);
   - (d) `glCullFace(FRONT)` left off in the volume pass: every light is added twice and at least 5 % of the pixels leave the budget (measured: about 40 %).
6. **Animation:** light 0 moves 0.7 units in one second; one second later the deferred image still matches the forward image everywhere, and with the camera held the image
   changes on more than 10 % of the pixels (measured: 46 to 49 %).
7. At the end the default framebuffer is redrawn so the runtime captures the normal frame; the runtime also checks `glGetError` after `init`, after the last frame and after `shutdown`.

## Pitfalls

- **Format choice:** `RGBA16F` is texture-only in core ES 3.0 (Table 3.13). Check the table instead of assuming the desktop GL rule. Normals in `RGB10_A2` cannot represent 0.5
  exactly, so a flat normal `(0, 1, 0)` is stored a hair off (about 0.001); it matters in lighting at grazing angles (the validation budget includes it).
- **Accumulation precision:** each additive light pass rounds to 10 bits again. A driver that truncates (llvmpipe, zink) loses about half a step per overlapping light, visible as slightly darker,
  banded dark regions compared with the full-screen pass. Production engines use `RGBA16F` accumulation, which needs an extension or ES 3.2; the full-screen pass (`V`) is exact to one rounding.
- **Depth texture:** a depth texture with `TEXTURE_COMPARE_MODE = NONE` is only complete with NEAREST filters (3.8.13). The default minification filter wants mipmaps, so the texture would
  be incomplete and `texelFetch` returns zeros; set `NEAREST` on every G-buffer texture. Declare the depth sampler `highp`.
- **Never sample what you are rendering to.** The G-buffer FBO, the light-buffer FBO and the default framebuffer are separate, so no texture is attached and sampled at once (a
  feedback loop has undefined results, 4.4.3).
- **Depth 1.0 and packing:** `uint(d * 16777215.0 + 0.5)` rounds to `2^24` in float for `d = 1`, wrapping to zero; clamp it.
- **Light volumes:** cull *front* faces and skip the depth test so the camera may sit inside a volume; scale the sphere so that the polyhedron circumscribes it, otherwise the light is clipped
  to a faceted ball; make the attenuation reach zero at the radius or the volume shows as a visible shape.
- **Dither:** `GL_DITHER` is enabled by default and may perturb stored 8-bit values; the demo disables it so the planes hold exactly the converted shader outputs.
- **`--seed`:** the scene is generated by a built-in deterministic generator, so the seed has no effect.

## Spec references

ES 3.0 §4.2.1 Selecting Buffers for Writing · §3.9.2.3 Shader Outputs · §3.8.3.1 Required Texture Formats, Tables 3.13 / 3.14 · §3.8.13 Texture Completeness · §4.4.2 / §4.4.4 Framebuffer
Objects, Completeness · §6.1.13 Framebuffer Object Queries · §4.3.2 Reading Pixels · §2.1.6.2 Float to Normalized Fixed-Point · §4.2.3 Clearing the Buffers · §4.1.5 Depth Test · §4.1.7 Blending · §4.1.9
Dithering · §3.6.1 Culling · §2.9.3 Drawing Commands · §2.12.6 Uniform Variables, §2.12.6.4 std140.
GLSL ES 3.00 §4.3.8.2 Output Layout Qualifiers · §4.3.8.3 Uniform Block Layout Qualifiers · §4.3.9 Interpolation · §4.5.4 Default Precision Qualifiers · §7.1 / §7.2 Special Variables · §8.8 Texture Lookup Functions.
