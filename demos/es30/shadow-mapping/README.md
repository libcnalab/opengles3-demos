# es30-shadow-mapping

**API introduced in:** OpenGL ES 3.0 (depth textures with comparison, `sampler2DShadow`) · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

A 3D scene with real shadows, made with a **shadow map**, a common approach to real-time shadows. The scene
(a checkered ground, a low platform, three boxes, four pillars with a lintel, three spheres, one of them
bobbing and one box spinning) is lit by an animated directional light. Shadows come from two passes:

1. **Pass 1 - depth from the light.** The scene is drawn from the light's point of view into a
   `DEPTH_COMPONENT24` *texture* that is the **only** attachment of a framebuffer object. There is no color
   attachment (`glDrawBuffers(NONE)`, `glReadBuffer(NONE)`) and the fragment shader has no output; the only product is the
   depth buffer.
2. **Pass 2 - the camera.** Every fragment is transformed into the light's clip space and the shadow map is asked
   "is anything nearer to the light than I am?" with a `sampler2DShadow`. The texture holds
   `TEXTURE_COMPARE_MODE = COMPARE_REF_TO_TEXTURE` and `TEXTURE_COMPARE_FUNC = LEQUAL`, so the lookup returns
   `Dref <= stored depth` as 0.0 or 1.0 (or a weighted mix of such results when the filter is `LINEAR`: hardware
   percentage-closer filtering).

Around those two passes the demo shows the three things that make shadow maps work in practice: **PCF** (hard,
hardware-filtered and 3x3-tap soft edges), **depth bias** (`glPolygonOffset` in pass 1 plus a small receiver bias, to
remove *shadow acne*), and a **shadow-map inset** that reads the same depth texture as plain depth values through a
sampler object whose comparison mode is `NONE`.

Everything is checked against an independent CPU model of the scene (analytic ray/box, ray/sphere and ray/plane
intersections in double precision), not against golden images.

## The image

The camera orbits the scene slowly while the light circles it at a constant 34 degrees elevation. The validated
frame (and the capture) is frame 30, t = 0.5 s, of the animation: the sphere's shadow lies on the platform and a second, longer
one runs over the ground, the floating blue sphere casts a detached shadow, the pillars and lintel throw a striped
shadow, the spinning purple box shadows the top of the orange box. Surfaces turned away from the light show only the
hemisphere ambient colour. The distance fades into a blue haze that matches the clear colour; the checker board
fades to its average colour beyond 25 units so that the far ground does not alias.

The yellow-framed square in the top-left corner is the shadow map itself (the middle 60 % of it): a gray ramp over the
whole depth range plus a repeating sawtooth that draws iso-depth bands. Every object is a distortion of the
otherwise regular stripes - the shadow map is just "the scene as depth seen from the light".

## Keys (windowed mode)

| Key | Effect |
|---|---|
| `P` | cycle the PCF mode: off (NEAREST, one comparison) / hardware (LINEAR) / hardware + 3x3 taps |
| `B` | depth bias on / off (both `glPolygonOffset` in pass 1 and the receiver bias in the shader) |
| `Space` | pause the animation |
| `A` / `D` | orbit the camera |
| `W` / `S` | move the camera closer / farther |

Switch the bias off with PCF set to "off" to see shadow acne: every lit surface is covered by a moire of dark
stripes. With PCF on, the same acne turns into a finer, fainter hatching.

```
./build/debug/bin/es30-shadow-mapping --headless --validate --capture shadow.png
./build/debug/bin/es30-shadow-mapping --window            # interactive
./build/debug/bin/es30-shadow-mapping --window --width 800 --height 600
```

## Central calls

- **Shadow map:** `glTexStorage2D(TEXTURE_2D, 1, DEPTH_COMPONENT24, 1024, 1024)`,
  `glTexParameteri` (`TEXTURE_MIN/MAG_FILTER`, `TEXTURE_COMPARE_MODE`, `TEXTURE_COMPARE_FUNC`),
  `glGenFramebuffers`, `glFramebufferTexture2D(DEPTH_ATTACHMENT, ...)`, `glDrawBuffers(1, {NONE})`,
  `glReadBuffer(NONE)`, `glCheckFramebufferStatus`.
- **Pass 1:** `glClearDepthf`, `glClear(DEPTH_BUFFER_BIT)`, `glEnable(POLYGON_OFFSET_FILL)`,
  `glPolygonOffset(factor, units)`, `glDrawElements`.
- **Pass 2:** `glUniformMatrix4fv` (light view-projection), `glUniform1i` (sampler unit), `glActiveTexture`,
  `glBindTexture`, `glDrawElements`, `glDrawArrays` (attribute-less inset and read-back triangles).
- **Raw depth through a sampler object:** `glGenSamplers`, `glSamplerParameteri(COMPARE_MODE, NONE)`,
  `glBindSampler(1, sampler)` - the same texture is bound to units 0 (comparing) and 1 (raw depth).
- **Read-back:** `glFramebufferTexture2D` onto an `RGBA8` texture, `glUniform2iv`, `glReadPixels`.
- **Queries:** `glGetFramebufferAttachmentParameteriv` (`OBJECT_TYPE`, `OBJECT_NAME`, `DEPTH_SIZE`), `glGetIntegerv`
  (`DRAW_BUFFER0`, `READ_BUFFER`, `SAMPLER_BINDING`), `glGetTexParameteriv`, `glGetSamplerParameteriv`, `glGetFloatv`
  (`POLYGON_OFFSET_FACTOR/UNITS`), `glIsEnabled`.

## Spec sections

- ES 3.0 §3.8.15 *Texture Comparison Modes* and §3.8.15.1: `Dref <= Dt` for `LEQUAL` (table 3.23); fixed-point `Dt`
  and `Dref` are clamped to [0,1]; with a non-`NEAREST` filter the result "may be computed by comparing more than one
  depth texture value ... The details of this are implementation-dependent".
- ES 3.0 §3.8.2 *Sampler Objects*: a bound sampler's state supersedes the texture's; §3.8.13 *Texture Completeness*: a
  depth texture with comparison `NONE` is incomplete unless both filters are `NEAREST` (or `NEAREST_MIPMAP_NEAREST`).
- ES 3.0 §3.6.2 *Depth Offset*: `o = m * factor + r * units` (equations 3.9-3.11); `m` may be approximated as
  `max(|dz/dx|, |dz/dy|)`.
- ES 3.0 §3.8.3 *Texture Image Specification* (depth formats) and §3.8.4 *Immutable-Format Texture Images*.
- ES 3.0 §4.2.1 *Selecting Buffers for Writing* (`DrawBuffers`), §4.3.1 *Selecting Buffers for Reading*, §4.3.2 *Reading Pixels*
  (no depth read-back; `INVALID_OPERATION` when `READ_BUFFER` is `NONE`).
- ES 3.0 §4.4.2 *Attaching Images to Framebuffer Objects*, §4.4.3 *Feedback Loops Between Textures and the Framebuffer*,
  §4.4.4 *Framebuffer Completeness* (one attached image is enough), §6.1.13 *Framebuffer Object Queries*.
- GLSL ES 3.00 §8.8 *Texture Lookup Functions* (shadow forms: the last component of `P` is `Dref`; a shadow lookup on a
  texture with comparison off, or a plain lookup on one with comparison on, is undefined), §4.5.4 *Default Precision
  Qualifiers* (`sampler2DShadow` has none), §7.1 `gl_VertexID`, §7.2 `gl_FragCoord`.

## Shader concepts

- **Light-space matrix.** A directional light has parallel rays, so its "camera" is orthographic: `ortho(-11, 11,
  -11, 11, 10, 40) * look_at(25 * L, origin, up)`. `scene.vert` multiplies the world position by it; the interpolated clip
  position `v_light` has `w = 1`. `scene.frag` maps NDC to texture space (`* 0.5 + 0.5`); `.xy` addresses the map,
  `.z` is the reference depth. (A spot light would use `perspective` and divide by `w`.)
- **`sampler2DShadow`.** `texture(u_shadow, vec3(s, t, Dref))` returns the comparison result. The depth value written by the
  rasterizer is the window-space z in [0,1], which for this orthographic projection is an affine function of the
  distance along the light direction - the same quantity as `sc.z`.
- **Comparison first, filtering second.** Depth comparison and filtering are separate steps. With `LINEAR` filters on
  the `sampler2DShadow` texture, the GL does not average depths and compare the average with `Dref`; it may compare
  several depth texels with `Dref` and combine the 0/1 *comparison results*, giving a value in [0, 1] "proportional to
  the number of comparison passes or failures" (ES 3.0 §3.8.15.1). The weights are implementation-dependent. (Averaging
  the depths across a depth edge would give a depth that belongs to no surface; averaging the results gives a soft edge.)
- **Receiver bias and out-of-map points.** `Dref - bias` is compared; points outside the map's [0,1]^2 or beyond the
  far plane are treated as lit (the map holds nothing there and the clamp-to-edge texels would shadow them wrongly).
- **PCF.** One tap on a `LINEAR` texture is hardware PCF; the 3x3 loop averages nine such taps, one texel apart.
- **Pass 1 shader.** A vertex shader that only transforms and an *empty* fragment shader. With `DRAW_BUFFER0 = NONE` an
  output would be meaningless.
- **Attribute-less triangle.** `fullscreen.vert` builds one oversized triangle from `gl_VertexID`; it draws the inset and the
  read-back row.
- **Read-back shader.** `depth_dump.frag` uses `texelFetch` on a raw-depth sampler and writes the 24-bit depth into the RGB
  channels of an `RGBA8` target (`uint(d * 16777215 + 0.5)` split into bytes) - ES cannot `glReadPixels` depth.
- **Lighting.** Blinn-Phong with a hemisphere ambient term, a checker board in the ground material, exponential fog and
  output gamma. The direct term and the specular term are multiplied by the shadow factor; the ambient term is not, which is what
  makes "shadowed = ambient only" an exact statement.

## How it is validated

The CPU model (`main.cpp`, double precision) knows the same shapes, camera and light as the shaders but shares no
rendering code with them. For a grid of pixels it casts the camera ray, finds the nearest of ground / box / sphere, casts a
ray from the hit point towards the light to decide whether it is occluded, and evaluates the lighting equation for the lit
and for the shadowed case. The GPU pixel must equal the matching colour.

**Pixels that are compared.** Only pixels well inside one surface (the 3x3 neighbourhood at +-2 pixels hits the same
primitive, the same box face and the same checker cell), with `N.L >= 0.25` (below that, lit and shadowed colours are
too close to separate, and slope-dependent bias dominates) or facing away from the light (ambient only), at most 24 world units
from the camera (beyond that the checker fades and fog flattens colours). **Penumbra exclusion:** a pixel is held out when
the occlusion differs anywhere on two rings (radius 0.25 and 0.125 world units, 8 points each) around it **or** when the shadow
map itself holds more than one surface within 4 texels of its map position. 0.25 world units are 11.6 shadow texels
(texel = 22 / 1024 = 0.0215 units); the PCF footprint reaches 3 texels (0.065 units) and the bias moves a shadow edge by at
most `o * cos(elevation)` = 0.085 units, so the margin has a comfortable reserve. The second condition came from a real
case: two shadows (lintel and cube) met exactly at a pixel, leaving a sliver of light only one texel wide that the ring test
could not see.
About 1800 pixels survive at 256 x 256 (810 lit / 51 shadowed ground, 59 lit / 180 shadowed object surfaces, 661 turned
away from the light), 166 ground pixels in the excluded zone are used for the penumbra experiment.

**Tolerances.** Compared colours may differ by 3 levels (of 255): the GPU and the CPU each round to 8 bits (1 level), and
2 levels remain for the precision of `pow`/`exp`/`normalize` in the shader and for the tessellated spheres (96 x 48
facets, whose interpolated vertex normals differ slightly from the analytic ones). The measured worst
difference is 1 level on every driver of the matrix. The sky shows the clear colour within 1 level. Depth values are
compared with 5e-6 (planar surfaces; the measured error is about 1e-7, a D24 step is 6e-8) and 1e-4 (spheres: an inscribed
triangle lies up to `rho^2 / (2 r)` = 9.6e-4 units inside the sphere, 3.2e-5 in window depth before the slant of the ray;
measured 4.2e-5). The specification gives no depth accuracy, so these are sanity bounds, and the measured values are printed.

**The checks**

| Group | What is compared |
|---|---|
| Pipeline state | Depth-only framebuffer complete; `DEPTH_ATTACHMENT` is the shadow texture with >= 24 depth bits; `COLOR_ATTACHMENT0` empty; `DRAW_BUFFER0` and `READ_BUFFER` are `NONE`; texture compare mode/func and the sampler object's state read back; `SAMPLER_BINDING` of units 0 and 1 |
| Defined errors | `glReadPixels` from the colour-less FBO -> `INVALID_OPERATION` (§4.3.2.1); `glDrawBuffers(COLOR_ATTACHMENT1)` at index 0 -> `INVALID_OPERATION` (§4.2.1); `glReadBuffer(BACK)` with an FBO bound -> `INVALID_OPERATION` (§4.3.1); the state survives all three |
| Shadow-map contents | Depth texels at 16 known positions (open ground, the top of every box, the centre of every sphere) are read back through the dump pass and equal the analytic window-space depth; occluders hold smaller depths than the ground behind them |
| `glPolygonOffset` | The same texels with the bias on are larger by `factor * m` (+ `units * r`): measured / (factor * m) is 1.000 on all three drivers, the accepted range follows §3.6.2 (m may be `max(|gx|,|gy|)`, so down to 0.64) |
| Inset | 160 inset pixels, chosen away from texel ties, depth edges and sawtooth edges, equal the display function of the analytic depth (+-2 levels); proves the sampler-object override (raw depth on unit 1 while unit 0 compares) |
| Lit / shadowed pixels | All ~1800 compared pixels for each of the three PCF modes (NEAREST, LINEAR, LINEAR + 3x3) with the bias on |
| Clear colour | 580 sky pixels equal `glClearColor` |
| Penumbra | PCF off: of the 166 edge-zone ground pixels none is anything but fully lit or fully shadowed (a NEAREST comparison can only return 0 or 1); 3x3 taps: 18 pixels lie strictly between the two colours; hardware PCF alone: all pixels lit, shadowed or between (the weights are implementation-dependent, so no count is required) |
| Comparison functions | `ALWAYS`: nothing is shadowed; `NEVER`: every pixel facing the light is shadowed; `GREATER`: lit and shadowed swap places (table 3.23) |
| Shadow acne | Bias off, PCF off: of the lit ground pixels away from texel ties, the CPU predicts for each one (reference depth minus the stored depth at the texel centre, from the plane equation) whether it turns dark - **all 708 agree**, and 53 % are dark (analytic expectation: about half) |
| Negative controls | see below |

**Negative controls (checks that must be able to fail).**
- The acne experiment is itself the control for the lit-colour check: with the bias off the same strict comparison fails on 53 % of the lit
  ground pixels (required: at least 25 %).
- A CPU model with the light direction rotated by 0.25 rad, and one with every occluder shifted by (0.6, 0, 0.4),
  is compared with the correct image; each must disagree on at least 1 % of the pixels (84 and 143 of 1761).
- The lit/shadowed colour separation of every compared pixel is checked to be >= 3x the tolerance (smallest: 44 levels), so a
  wrong shadow state cannot hide inside the tolerance.
- The polygon-offset range is strictly above zero: a driver that ignored `glPolygonOffset` would fail it.

## Bias, in numbers

For this scene (light elevation 34 degrees) the ground plane's depth changes by about 1.06e-3 (window z) per shadow texel,
so one "texel of slope" is worth ~0.03 world units along the light. `glPolygonOffset(3, 4)` therefore pushes the stored ground depth
away by `3 * m` = 3.2e-3 (about 0.1 units); a constant 2e-4 is added in the shader. Measured in a development run: `factor = 1`
was enough for NEAREST and for hardware PCF, but not for the 3x3 mode (472 of 2203 compared pixels still showed acne);
`factor = 2` was clean; the demo uses 3. Larger factors only move shadows away from their casters ("peter panning"): at
`factor = 3` the shadow edge moves by less than 0.09 units.

## Pitfalls

- **`sampler2DShadow` has no default precision** in GLSL ES 3.00 (§4.5.4) - the shader does not compile without `precision highp
  sampler2DShadow;`. And `sampler2D` defaults to `lowp`: a plain depth lookup without `highp` returns garbage.
- **The default minification filter is `NEAREST_MIPMAP_LINEAR`.** A single-level depth texture with it is incomplete and
  every lookup returns 0 (all shadow). Set `LINEAR` or `NEAREST`. And the other way round: with comparison `NONE`, a depth
  texture needs `NEAREST` (§3.8.13) - that is why the raw-depth sampler object is `NEAREST`.
- **A plain `sampler2D` on a texture whose comparison is on is undefined** (GLSL §8.8). Use a sampler object (§3.8.2) or change
  the texture parameter.
- **Depth cannot be read back** with `glReadPixels` (§4.3.2). The inset and the dump pass exist because the shadow map is
  otherwise invisible; the pixels of the inset are real depth values.
- **A depth-only framebuffer needs `glDrawBuffers(NONE)` and `glReadBuffer(NONE)`** (the defaults point at
  `COLOR_ATTACHMENT0`, which does not exist), and the fragment shader must not be required to write anything.
- **Feedback loops** (§4.4.3): a texture attached to the draw framebuffer must not be sampled by the current program. In pass 1
  the shadow texture is still bound to units 0 and 1, but `shadow.frag` samples nothing, so there is no loop.
- **Reference depth outside [0,1]** is clamped for fixed-point depth textures (§3.8.15.1); with `CLAMP_TO_EDGE` the edge texels
  would smear shadows to infinity, so the shader treats everything outside the light's frustum as lit.
- **Hardware PCF is implementation-dependent** (§3.8.15.1). The demo only relies on the weights being a mix of the comparison
  results (and on a `NEAREST` lookup being exactly 0 or 1).
- **`m` in the polygon offset is defined two ways** (§3.6.2, `sqrt(gx^2 + gy^2)` or `max(|gx|, |gy|)`). The measured drivers use the
  former; the check accepts both.
- **Bias is a trade-off:** too little gives acne, too much detaches shadows from their casters. The receiver bias should
  also scale with the filter radius - the 3x3 mode needs the most.
