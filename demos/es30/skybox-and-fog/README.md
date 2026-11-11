# es30-skybox-and-fog

**API introduced in:** OpenGL ES 2.0 (cube maps), ES 3.0 (seamless cube map filtering, `glTexStorage2D`)
· **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

One cube map, used three ways, is enough to make a small scene look like it stands outdoors:

* as a **skybox**: a cube around the camera that only rotates with it, drawn *last* with the
  `gl_Position = clip.xyww` depth trick so that it can never cover anything;
* as a **mirror environment**: a chrome sphere and a gold box show `texture(sky, reflect(I, N))`;
* as a **refracted environment**: a glass sphere shows `texture(sky, refract(I, N, 1/1.5))`, blended with
  the reflection by the **Fresnel-Schlick** factor, so its rim turns into a mirror.

A checkered ground plane 800 units wide recedes to the horizon and dissolves in **exponential-squared
fog**. OpenGL ES has no fixed-function fog (no `glFog`, no `GL_FOG`): the fragment shader computes the
fog factor from the eye distance and mixes towards the fog colour, which is exactly the sky colour at the
horizon, so the far ground melts into the haze band of the sky.

The cube map is not loaded from files: `make_sky_image()` paints the six 512 x 512 faces on the CPU
(sky gradient, a sun disc with a warm glow, two ranges of distant mountains, a hazy ground below the
horizon). Painting it means walking through ES 3.0 Table 3.21 backwards: for every texel of every face the
program computes the direction that the GPU will later map to that texel, and evaluates the sky along it.

## The image

The camera looks across the ground towards the mountains, slightly downwards. From left to right: the
**chrome sphere** (the blue zenith on top, the hazy ground below, the horizon as a bright band), the
**gold box** (tinted reflections; its faces are flat mirrors), the **glass sphere** (the sky upside down,
because refraction through a sphere inverts it, and a bright reflective rim). The sun hangs top left with
a glow, the mountains are two silhouettes, the paler one farther away. The checker tiles get paler and
lose contrast with distance until they are pure fog colour.

What the demo does **not** do: the objects reflect and refract only the environment map, never each other
or the ground (a *static* environment map). The glass sphere is a single refracting interface (the ray
is not refracted a second time when it leaves the sphere), the usual real-time approximation.

### Controls (`Demo::on_key`, also printed by `--help`)

| Key | Action |
|---|---|
| `a` `d` | orbit the camera left / right (it also orbits slowly by itself, 0.08 rad/s) |
| `w` `s` | dolly in / out |
| `q` `e` | raise / lower the camera |
| `f` `g` | fog density times / divided by 1.25 (default 0.035 per unit) |
| `o` | fog on / off |
| `c` | show the face-coded validation cube map instead of the sky (+X red, -X cyan, +Y green, -Y magenta, +Z blue, -Z yellow) |
| `n` | NEAREST instead of trilinear sky filtering (the validation configuration) |
| `k` | depth trick off: the sky cube is drawn at its true depth and hides the whole scene |
| `i` | glass with the inverse ratio eta = 1.5 (wrong): total internal reflection near the rim |
| space | pause the animation |
| `r` | reset everything |
| ESC | quit |

Run it: `./build/debug/bin/es30-skybox-and-fog --window --width 960 --height 540`.
`--headless --validate --capture out.png` is what CTest runs; time is `frame / 60` s and frame 0 is
validated.

## Cube maps in ES 3.0

A cube map is one texture object with six square images, one per face target
`TEXTURE_CUBE_MAP_POSITIVE_X ... NEGATIVE_Z`. The demo allocates all levels of all faces at once with
`glTexStorage2D(GL_TEXTURE_CUBE_MAP, 10, GL_RGBA8, 512, 512)` and fills level 0 of each face with
`glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, ...)`; `glGenerateMipmap` builds the rest.

A `samplerCube` is looked up with a **direction**, not a coordinate. The largest component in magnitude
(the *major axis*) selects the face, and the other two, divided by it, give the position on the face
(ES 3.0 §3.8.9, Table 3.21):

| major axis | target | sc | tc | ma |
|---|---|---|---|---|
| +rx | POSITIVE_X | -rz | -ry | rx |
| -rx | NEGATIVE_X | rz | -ry | rx |
| +ry | POSITIVE_Y | rx | rz | ry |
| -ry | NEGATIVE_Y | rx | -rz | ry |
| +rz | POSITIVE_Z | rx | -ry | rz |
| -rz | NEGATIVE_Z | -rx | -ry | rz |

`s = (sc / |ma| + 1) / 2`, `t = (tc / |ma| + 1) / 2`. The direction does not have to be normalized. The
signs are the classic trap: on the side faces `t` grows *downwards* (`tc = -ry`), so an image painted
"top row first" must be uploaded with that row at `t = 0`; on `+Y` the `t` axis points to `+z`. The first
row of a `glTexSubImage2D` upload is `t = 0`. If two components have the same magnitude the
implementation may choose either face with any deterministic rule.

**Seamless filtering (ES 3.0 §3.8.9.1).** "The rules for texel selection ... are modified for cube maps
so that texture wrap modes are ignored. Instead, if NEAREST filtering is done within a miplevel, always
apply wrap mode CLAMP_TO_EDGE. If LINEAR filtering is done within a miplevel, always apply border
clamping. Then, if a texture sample location would lie in the texture border in either u or v, instead
select the corresponding texel from the appropriate neighboring face." Appendix F.2 lists this as a
behaviour change from ES 2.0, where one face was selected and filtered alone (and the usual fix was
`CLAMP_TO_EDGE` plus visible seams). In ES 3.0 there is nothing to enable: the sky shows no seams along
the cube edges with LINEAR filtering, and setting `TEXTURE_WRAP_S/T` on a cube map has no effect. (The
corner texels where three faces meet are the one place where the implementation has some freedom.)

## The skybox depth trick

`sky.vert` draws the unit cube with `mat3(u_view)`, the view matrix without its translation: the cube
turns with the camera but never moves relative to it, so the sky is infinitely far away. The object-space
corner position is the look-up direction.

```glsl
vec4 clip = u_proj * vec4(mat3(u_view) * a_position, 1.0);
gl_Position = clip.xyww;            // z_c = w_c
```

With `z_c = w_c` the perspective division gives `z_d = z_c / w_c = 1` (ES 3.0 §2.13) and the viewport
transformation `z_w = (f - n)/2 * z_d + (n + f)/2` gives `z_w = 1`, the far plane, for every sky fragment
(§2.13.1). Such a vertex is still inside the clip volume (`-w_c <= z_c <= w_c`, §2.18). The sky is then
drawn **after** the opaque scene with `glDepthFunc(GL_LEQUAL)`: where the depth buffer still holds the
cleared 1.0, `1.0 <= 1.0` passes and the sky appears; where anything was drawn, its depth is smaller and
the sky fails. Drawing the sky last also means its fragment shader runs only on the visible sky pixels.

Why `LEQUAL` and not `LESS`: with `LESS` the sky would have to be strictly nearer than the cleared 1.0. In
exact arithmetic it never is, so the sky would vanish, but `z_c / w_c` may be computed as
`z_c * (1 / w_c)` and land one ULP below 1, so with `LESS` some sky pixels might survive and others not.
The demo does not test `LESS` for that reason: the outcome is not specified.

## Environment mapping and Fresnel

In `scene.frag`, `I = normalize(world - eye)` points from the eye to the surface and `N` is the unit
normal. GLSL ES 3.00 §8.5:

```
reflect(I, N)      = I - 2 dot(N, I) N
refract(I, N, eta) : k = 1 - eta^2 (1 - dot(N, I)^2)
                     k < 0 -> vec3(0)   (total internal reflection)
                     else  -> eta I - (eta dot(N, I) + sqrt(k)) N
```

`eta` is the ratio n1 / n2 of the refractive indices, here air into glass: 1 / 1.5. Passing 1.5 instead
is a common mistake; it makes `k` negative wherever `sin(theta) > 1/1.5`, i.e. on a wide ring, and the
shader then has a zero direction. `scene.frag` replaces a zero `refract()` result by the reflection
(physically: total internal reflection), so that the wrong ratio (key `i`) still produces a defined
picture. A zero direction must never reach `texture()`: Table 3.21 would divide by `|ma| = 0`.

The Fresnel-Schlick approximation gives the fraction of light that is reflected:
`F = F0 + (1 - F0) (1 - cos theta)^5` with `cos theta = dot(-I, N)` and
`F0 = ((1 - 1.5) / (1 + 1.5))^2 = 0.04`. The glass colour is
`mix(tint * sky(refract), sky(reflect), F)`: 4 % reflection when looking straight at the sphere, rising
towards 100 % at grazing angles. The fifth power is written as products, `(m*m)*(m*m)*m`, instead of
`pow(m, 5.0)`: products need no transcendental function and keep the few-ULP accuracy of `*`, whereas
the precision of `pow` is only "inherited from `exp2(y * log2(x))`" (GLSL ES 3.00 §4.5.1), and `m` is
exactly 0 at the centre of the sphere.

The ground is lit by the sky ambient and the sun (Lambert), with the sun direction taken from the painted
sun disc, so shading and sky agree.

## Fog

```
f      = exp(-(density * d)^2)        d = |world - eye|
colour = mix(fog_colour, surface_colour, f)
```

This is the `GL_EXP2` formula of desktop OpenGL's old fixed-function fog, computed per fragment with the
true eye distance instead of the eye-space depth. The fog colour is the horizon colour of the sky, so the
ground meets the sky without a seam. The sky itself is not fogged: it *is* the fog's destination.

## Central calls

`glGenTextures`, `glBindTexture(GL_TEXTURE_CUBE_MAP)`, `glTexStorage2D(GL_TEXTURE_CUBE_MAP, ...)`,
`glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, ...)`, `glGenerateMipmap`, `glTexParameteri`
(`MIN/MAG_FILTER`, `WRAP_S/T`), `glGetTexParameteriv`, `glActiveTexture`, `glUniform1i` (sampler unit),
`glDepthFunc(GL_LESS / GL_LEQUAL)`, `glClearDepthf`, `glUniformMatrix4fv`, `glUniform1f/3f`,
`glDrawElements`, `glDrawArrays` (attribute-less probe triangle), `glReadPixels`.

## Spec sections

* ES 3.0 §3.8.9 Cube Map Texture Selection and Table 3.21; §3.8.9.1 Seamless Cube Map Filtering;
  Appendix F.2 (seamless filtering is new in ES 3.0).
* ES 3.0 §3.8.10.2 Coordinate Wrapping and Texel Selection (NEAREST texel; LINEAR weights
  `alpha = frac(u - 1/2)`).
* ES 3.0 §3.8.4 Immutable-Format Texture Images, §3.8.5 Alternate Texture Image Specification Commands.
* ES 3.0 §2.13 Coordinate Transformations, §2.13.1 Controlling the Viewport, §2.18 Primitive Clipping.
* ES 3.0 §4.1.5 Depth Buffer Test, §4.2.3 Clearing the Buffers.
* ES 3.0 §2.1.6.2 Conversion from Floating-Point to Normalized Fixed-Point.
* ES 3.0 Table 6.28 (`MAX_CUBE_MAP_TEXTURE_SIZE` at least 2048).
* GLSL ES 3.00 §8.5 Geometric Functions (`reflect`, `refract`, `normalize`, `dot`, `length`), §8.8 Texture
  Lookup Functions (`texture` with `samplerCube`), §8.2 (`exp`), §8.3 (`mix`, `clamp`, `floor`, `mod`),
  §4.5.1 Range and Precision.

## Shader concepts

* `sky.vert` / `sky.frag`: the depth trick, `mat3(mat4)` to drop the translation, an unnormalized
  direction as the cube coordinate, a `bool` uniform (`u_depth_trick`, set with `glUniform1i`).
* `scene.vert` / `scene.frag`: world-space position and normal varyings (the normal is renormalized per
  fragment); a material switch on a uniform `int`, so the `texture()` calls stay in dynamically uniform
  control flow and their implicit derivatives (mipmap selection) are defined; `reflect`, `refract`, the
  zero-vector test for total internal reflection, Fresnel-Schlick, `exp` fog.
* `probe.vert` / `probe.frag`: a full-viewport triangle generated from `gl_VertexID` with no vertex buffer,
  and a cube lookup along a direction given as a uniform.

## Validation

`validate()` records 61 checks at any surface size (checked at 256x256, 800x800, 640x480, 333x777 and
1280x400 on radeonsi, llvmpipe and zink, at ES 3.2 and with the ES 3.0 override).

**Validation configuration.** The sky's filters are switched to `NEAREST` (read back with
`glGetTexParameteriv`), and the second, face-coded cube map is NEAREST anyway. Every lookup then returns
one uploaded RGBA8 texel, so the CPU can predict the pixel exactly instead of modelling an
implementation-dependent filter. The windowed configuration (trilinear, mipmapped) is what the capture
shows; it is not compared pixel by pixel.

**The CPU model** (`class Reference`, double precision) shares only the scene constants and the CPU copies
of the uploaded texels with the renderer. It has its own camera (no matrix projects or generates rays),
exact ray intersections with the analytic spheres, the rotated box (slab test in object space) and the
ground plane, Table 3.21 as `select_face()`, NEAREST texel selection with clamp-to-edge, and the formulas
of the sections above written out again (`glsl_reflect`, `glsl_refract`, Fresnel, fog). A ray that hits
nothing is the sky.

**Accept either neighbour.** A pixel whose direction lies near a texel boundary or a cube edge may
legitimately select either texel (float rounding of the interpolated direction; the tie rule of §3.8.9).
The model therefore evaluates a 5 x 5 grid of directions (sky) or normals (objects) around the exact one
and accepts the per-channel interval of the results, plus the tolerance. The half-widths: 5e-4 rad for the
sky and the box (rounding only); 3e-3 rad for the spheres, whose GPU normal is `normalize(P)` of the point
`P` where the ray meets a flat triangle of the 128 x 64 mesh, at most the sagitta 6e-4 r inside the
analytic sphere; for `N.V >= 0.3` the angle between `P` and the analytic hit is at most
6e-4 * tan(72.5 deg) = 1.9e-3 rad. Grazing sphere pixels (`N.V < 0.3`) are not compared. So that
boundary probes cannot carry a check, at least half of the pixels of every sweep must have an interval
at most 2 steps wide (most have width 0); the detail line prints both counts.

**Tolerance: 2 steps.** NEAREST returns exact texels. What remains is highp arithmetic (GLSL ES 3.00
§4.5.1; a few ULP in `mix`, `exp`, `normalize`, far below one 8-bit step) and the conversion of the output
to 8 bits, which may give either of the two nearest values (ES 3.0 §2.1.6.2): 1 step, plus 1 for products
of two rounded quantities (tint times texel, fog blend). The largest deviation measured on all drivers is 1.

**Stable pixels.** A pixel is compared only if the CPU rays at +-2 px hit the same object (and the same
ground tile or box face), so no probe sits on a silhouette or a colour edge where the rasterization
tie-break rules decide.

What is checked:

1. **Texture state.** `TEXTURE_BINDING_CUBE_MAP` is the sky while the scene is drawn; the sky's
   `MIN_FILTER` / `MAG_FILTER` read back `NEAREST`; `MAX_CUBE_MAP_TEXTURE_SIZE` admits 512.
2. **Face orientation.** With the camera at the origin looking along each of the six axes, only the sky
   is drawn with the face-coded cube map: 5 x 5 texels per face, the face hue scaled by the column and
   lifted by the row, so every texel is unique. 81 probes per face match the CPU model of Table 3.21, and
   (model-free) the screen centre shows the centre texel of the image that was uploaded to that face's
   target. Mirroring `s` or `t` in the model (the sign mistakes of the table) is rejected by every one of
   the 324 discriminating probes.
3. **Negative control: swapped faces.** A temporary cube map gets the +X and -X images uploaded to each
   other's targets. Looking along +X every probe on the +X face fails the correct model, and every probe
   matches the model of the swapped upload.
4. **Seamless filtering.** A third cube map has one uniform colour per face, `LINEAR` filtering and
   `WRAP_S/T = REPEAT`. `probe.frag` samples it along directions exactly on three cube edges, e.g.
   `(1, 0, 1)`: whichever face the tie rule picks, `s` is exactly 0 or 1, the 2 x 2 footprint straddles
   the edge with weights 1/2 and 1/2, and the texels beyond the edge come from the neighbouring face. The
   result must be the average of the two face colours (+-2). A non-seamless implementation would return
   one face's colour, 70-80 steps away; REPEAT, if it were honoured, would wrap to the same face and give
   the same wrong answer. Inside a face the same lookup returns the plain face colour (control).
5. **The scene** in the validation configuration: per-object sweeps of every `max(3, min(W,H)/64)`-th
   stable pixel for the sky, the ground, the chrome sphere, the glass sphere and the gold box, all
   matching. Counterfactual models are rejected wherever they differ from the right one by at least 12
   steps: sampling along `I` instead of `reflect(I, N)` (chrome, 148 of 148 pixels at 256 x 256), and
   `refract` with the inverse ratio 1.5 (glass, 148 of 148).
6. **The scene in the face-coded environment** (key `c`): the same five sweeps. Reflected and refracted
   rays now land on faces of very different hue, which makes every formula far more discriminating than
   the soft sky: the gold box sampled along `I` is rejected at all 84 discriminating pixels. The Fresnel
   term is judged on a **close-up of the glass sphere** in the same environment (the default camera
   direction, 3.7 units from the sphere, so its radius is about 60 % of half the shorter side at any aspect
   ratio): the close-up first matches the right model in its own sweep, then a full-resolution scan of
   the ring where `F > 0.1` rejects the glass without the Fresnel term (refraction only) at every
   discriminating pixel (1126 at 256 x 256, 665 at 1000 x 200). In the full scene that ring is only a pixel
   or two wide, and on a 200-pixel side it holds just two stable pixels.
7. **The depth trick.** On each object the pixel shows the object, not the sky along that ray, although
   the sky is drawn after it. *Negative control:* with `u_depth_trick = false` (`gl_Position = clip`) the
   sky cube is drawn at its true depth, half a unit from the eye, passes `LEQUAL` against everything and
   covers the whole frame: every stable pixel equals the sky along its ray, and the chrome sphere probe
   fails the scene model. The near plane (0.25) must lie inside that cube, or a very wide field of
   view (1000 x 200) would clip the cube's walls near the frame corners and show the scene there; it is
   therefore reduced to `0.9 * 0.5 / |(tan_x, tan_y, 1)|` of the frustum corner when that is smaller.
8. **Fog at a known distance.** For the centre of a ground tile at eye distance `d` (6.56 at 256 x 256) the
   density is set to `sqrt(ln 2) / d`, so `exp(-(density d)^2) = 1/2`: the pixel must be the 50/50 mix of
   the lit ground and the fog colour (measured surface weight 0.50). With the fog off the same pixel is the
   unfogged lit colour, 76 steps from the fogged expectation (negative control). A dense fog (0.08)
   sweep covers all distances.
9. **Negative control: inverse eta rendered.** The glass drawn with eta = 1.5 matches the eta = 1.5 model
   (including the total internal reflection ring), and the correct model fails at every discriminating
   pixel.
10. **Keyboard.** `d`, `q`, `f`, `g`, `o`, `c`, `k`, `i`, space and `r` change the parameters as documented;
    a sweep of all objects with the orbited camera and the denser fog matches.

## Pitfalls

* **Face orientation.** Table 3.21 is not "look at the face from outside": on the four side faces `t`
  grows downwards, on `-Z` `s` grows towards `-x`. Painting or loading images in the wrong orientation
  gives a sky with seams that line up nowhere; the face-coded cube map (key `c`) makes it visible.
* **`mat3(u_view)`.** Forgetting to drop the translation makes the sky a small box that the camera can
  walk out of.
* **`LESS` instead of `LEQUAL`** for the sky: the sky disappears or flickers (see above). Forgetting
  `.xyww` (key `k`): the sky cube covers the scene.
* **`refract` eta is n1/n2**, not n2/n1. The inverted ratio (key `i`) gives total internal reflection
  on a ring where the light should enter the glass.
* **Zero direction into `texture()`.** `refract()` returns `vec3(0)` on total internal reflection; never
  sample a cube map with it.
* **Seams with ES 2.0 habits.** In ES 3.0 cube maps are always seamless and ignore the wrap modes; there
  is no `GL_TEXTURE_CUBE_MAP_SEAMLESS` enable in ES (that is desktop GL).
* **Fog in ES** is shader code. Fixed-function fog (`glFog*`, `GL_FOG`) does not exist in any ES version
  with shaders.
* **Undersized sky textures.** At 50 degrees of vertical field of view an 800 pixel tall window shows
  about 16 pixels per degree; a 256 x 256 face (0.35 degrees per texel at its centre) shows texel steps
  along the mountain ridges even with LINEAR filtering. The sky is therefore 512 x 512 and its edges are
  painted with sub-texel coverage (anti-aliased), which bilinear magnification turns into smooth lines.
