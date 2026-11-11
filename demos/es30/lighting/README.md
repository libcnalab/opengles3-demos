# es30-lighting

**API introduced in:** OpenGL ES 2.0 (shader-based lighting) · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

OpenGL ES has no lighting API at all. There is no `glLight`, no `glMaterial`, no normal
transformation: every light you see on screen is a few lines of GLSL that the application feeds with
vectors and matrices. This demo builds a small 3D scene (a checkered ground, spheres and boxes made of
procedural meshes) and lights it with the textbook model, entirely in shaders:

* a constant **ambient** term,
* a **directional** light (the sun: parallel rays, no attenuation),
* a **point** light with distance attenuation `1 / (kc + kl d + kq d^2)`,
* a **spot** light: a point light inside a cone, with a smooth falloff between an inner and an outer angle,
* **diffuse** (Lambert) and **specular** terms, with the **Phong** or the **Blinn-Phong** specular model,
* lighting per **fragment** (Phong shading) and, for comparison, per **vertex** (Gouraud shading),
* the **normal matrix**: why the normal must not be transformed with the model matrix, shown on a
  non-uniformly scaled object that is rendered once correctly and once with the naive matrix.

The point is not only to draw something that looks lit but to *prove* the equations: validation
ray-casts through chosen pixels on the CPU, evaluates the same formulas in double precision and compares
the 8-bit result.

## The image

Camera and scene (y up, ground at y = 0, the camera looks at the middle from the front and above; its 40 degree
field of view spans the shorter side of the surface, vertical on a landscape surface and horizontal on a portrait one,
so both columns of objects stay in view at any aspect ratio):

```
back row    coarse sphere, lit per VERTEX (Gouraud)     coarse sphere, lit per FRAGMENT
middle row  "crystal" with the CORRECT normal matrix    "crystal" with the NAIVE normal matrix
front row   sphere, PHONG specular                      sphere, BLINN-PHONG specular
```

Every pair differs in exactly one respect, so you read the demo by comparing left with right:

| Pair | What to look at |
|---|---|
| front spheres | identical material (exponent 40). The Blinn-Phong highlight is visibly **larger and softer** than the Phong one: `N.H` falls off half as fast as `R.V` as the viewer leaves the mirror direction (the angle between `R` and `V` is twice the angle between `N` and `H`). |
| crystals | each is a cube that is first **rotated** (35 degrees around y at t = 0, then spinning) and then **stretched** by `S = (1.8, 1.2, 0.85)` along the *world* axes, i.e. `model = T * S * R`. The faces are no longer perpendicular to the transformed `R n`. The left crystal gets `transpose(inverse(mat3(model)))`, the right one `mat3(model)`. Look at the faces that point left (-x): with the correct matrix the face is still slightly lit by the sun, with the naive matrix its normal is 38 degrees off, it faces away from the sun and is visibly darker. Under the animation the error changes with the rotation angle. |
| coarse spheres | one 12 x 8 mesh. Per vertex, the highlight and the edge of the spot cone fall between vertices and are smeared or lost (blotchy, faceted gold); per fragment they are smooth. |

The warm, small sphere is the **point light** (an emissive marker), the pale cone near the top is the
**spot light**. The spot's pool on the floor shows the **smooth cone edge** (cool white, inner cone 11
degrees, outer 22 degrees); the warm pool is the point light's **attenuation**. In windowed mode the
point light moves along the aisle, the spot cone sweeps the floor, the crystals rotate and the camera
sways, so the normal-matrix error changes continuously.

### Controls (`Demo::on_key`, also printed by `--help`)

| Key | Action |
|---|---|
| `1` `2` `3` | toggle the directional / point / spot light |
| `b` | Phong <-> Blinn-Phong for the ground, the crystals and the coarse spheres (the two front spheres always show both) |
| `n` | give the right crystal the correct normal matrix as well (both crystals then look alike) |
| `g` | per-fragment lighting on the left coarse sphere (both coarse spheres then look alike) |
| `a` `d` | orbit the camera left / right |
| `w` `s` | dolly in / out |
| `q` `e` | raise / lower the camera |
| space | pause the animation |
| `r` | reset everything |
| ESC | quit |

Run it: `./build/debug/bin/es30-lighting --window` (SDL3 window, 256 x 256 unless `--width/--height` are
given; `xvfb-run -a ... --window --frames 120` works headlessly). `--headless --validate --capture out.png` is
what CTest runs; time is `frame / 60` s, so frame 0 is the state that is validated and captured.

## The lighting model

All vectors are in **world space**; `N` unit normal, `V` unit vector surface -> camera, `L` unit vector
surface -> light. For one light with radiance `E` (light colour times attenuation times cone factor):

```
reflected = E * ( albedo * max(N.L, 0)  +  specular * pow(max(S, 0), shininess) )      only if N.L > 0

    Phong:        S = R.V,   R = reflect(-L, N) = 2 (N.L) N - L
    Blinn-Phong:  S = N.H,   H = normalize(L + V)

colour = emissive + ambient * albedo + sum(reflected over the three lights)
pixel  = pow(clamp(colour, 0, 1), 1 / 2.2)                                         (gamma encoding)
```

* **Directional:** `L = normalize(-direction)`, `E = colour`.
* **Point:** `L = normalize(P - position)`, `d = |position - P|`, `E = colour / (kc + kl d + kq d^2)`.
* **Spot:** like the point light, times `cone = smoothstep(cos_outer, cos_inner, dot(-L, axis))`. The axis
  is the unit direction the cone points; `cos_inner > cos_outer` because the inner angle is the smaller
  one (`smoothstep` is undefined for `edge0 >= edge1`). Inside the inner cone `cone = 1`, outside the outer
  cone `cone = 0` exactly, between them `3t^2 - 2t^3` of the linear ramp `t`.
* **Specular only where `N.L > 0`:** a surface turned away from the light must not get a highlight just
  because `R.V` is positive.
* **Phong vs Blinn-Phong with the same exponent are different materials.** The angle in `R.V` is twice the
  angle in `N.H`, so Phong needs about 4 times the exponent to match the highlight size of Blinn-Phong.
* **Gamma:** lighting is a *linear* computation; the framebuffer here is not sRGB, so the final
  `pow(c, 1/2.2)` makes mid-tones look right. (`es30-srgb` shows the sRGB framebuffer alternative.)

### The normal matrix

A normal is not an arrow that moves with the surface; it is defined by being perpendicular to it. If a
surface tangent `t` is transformed to `M t`, the transformed normal `n'` must still satisfy `n'.(M t) = 0`.
`n' = M^-T n` (the inverse transpose) does: `(M^-T n).(M t) = n^T M^-1 M t = n.t = 0`. For a rotation
`M^-T = M`, which is why the mistake goes unnoticed until the first non-uniform scale. Here
`model = T * S * R` with `S = diag(1.8, 1.2, 0.85)`, so the correct matrix is `(S R)^-T = S^-1 R` while the
naive one is `S R`. For the cube's -x face at 35 degrees: `R n = (-0.82, 0, 0.57)`; correct normal
`(-0.56, 0, 0.83)`, naive normal `(-0.95, 0, 0.31)`, 37.7 degrees apart, enough to flip the face from lit to
unlit under the sun. `demo::math::normal_matrix(model)` computes the cofactor form of the inverse
transpose of the upper-left 3x3; it is uploaded with `glUniformMatrix3fv` into a `mat3` uniform.

### Per-vertex (Gouraud) vs per-fragment

`gouraud.vert` evaluates the *same* equation once per vertex with the vertex normal and passes the colour
on; `gouraud.frag` only gamma-encodes it. The rasterizer interpolates the colour linearly across the
triangle (perspective-corrected, ES 3.0 equation 3.7). A highlight or a cone edge that falls inside a
triangle but not at its vertices cannot be represented. `lighting.frag` interpolates position and normal
and runs the equation per fragment, so the result does not depend on the tessellation.

## Central calls

`glGenVertexArrays`, `glBindVertexArray`, `glGenBuffers`, `glBindBuffer`, `glBufferData`,
`glEnableVertexAttribArray`, `glVertexAttribPointer`, `glDrawElements` (UNSIGNED_SHORT indices; the
`ELEMENT_ARRAY_BUFFER` binding is vertex-array state), `glUseProgram`, `glGetUniformLocation` (struct members
are addressed as `u_material.albedo`, `u_spot.cos_inner`, ...), `glUniformMatrix4fv` / `glUniformMatrix3fv`
(`transpose = GL_FALSE`, column-major, the layout of `demo::math::Mat4`), `glUniform1i/1f/3f`,
`glEnable(GL_DEPTH_TEST)`, `glDepthFunc`, `glEnable(GL_CULL_FACE)`, `glCullFace`, `glFrontFace`, `glClear`,
`glClearColor`, `glViewport`, `glGetUniformfv`.

## Spec sections

* ES 3.0 §2.12.6.1 Loading Uniform Variables In The Default Uniform Block: `UniformMatrix{34}fv`,
  column-major with `transpose = FALSE`; the size and count rules that raise `INVALID_OPERATION`;
  location -1 is silently ignored.
* ES 3.0 §6.1.12 Shader and Program Queries: `GetUniformfv` returns matrices in column-major order.
* ES 3.0 §3.6.1 Basic Polygon Rasterization: front/back determination from the window-space area
  (equation 3.6), `FrontFace`, `CullFace`; perspective-correct interpolation of varyings (equation 3.7).
* ES 3.0 §4.1.5 Depth Buffer Test.
* ES 3.0 §2.1.6.2 Conversion from Floating-Point to Normalized Fixed-Point: the 8-bit result is one of the
  two closest values.
* GLSL ES 3.00 §4.5.1 Range and Precision (highp is IEEE single with the ULP table; `pow` inherits from
  `exp2`/`log2`), §4.5.4 Default Precision Qualifiers (a fragment shader needs a float precision),
  §4.3 (storage qualifiers, uniform structs), §8.3 Common Functions (`clamp`, `max`, `mix`, `mod`,
  `floor`, `smoothstep`), §8.5 Geometric Functions (`normalize`, `dot`, `length`, `reflect`).

## Shader concepts

* `lighting.vert` outputs the world position and `u_normal_matrix * a_normal` (not normalized: an
  interpolated vector is shorter than 1 inside a triangle, so the fragment shader normalizes again).
  `gl_Position = u_proj * u_view * u_model * position` keeps the three classic matrices separate.
* `lighting.frag` declares `precision highp float` (the fragment language has no default float
  precision; highp is mandatory in GLSL ES 3.00 fragment shaders) and defines `struct Material`,
  `struct DirLight`, `struct PointLight`, `struct SpotLight` uniforms. The checker pattern is
  `mod(floor(x / s) + floor(z / s), 2.0)` on the world position. A lamp is a material with zero albedo
  and zero specular and a non-zero `emissive`.
* `gouraud.vert` repeats `reflect_light()` and moves the lighting into the vertex stage. GLSL ES 3.00 has no
  `#include`; the duplication is deliberate and keeps both programs self-contained.
* Both programs share one uniform interface; uniforms the compiler removes get location -1 and the
  upload code can ignore that.

## Validation

`validate()` runs 156 checks. It renders the scene several times with chosen lights switched off, reads the
frame back and compares it with a CPU model.

**The CPU model** (double precision, `class Reference` in `main.cpp`) shares only the scene *constants*
with the renderer. It has its own vector type, its own camera (no matrix is used to project or to cast
rays), its own inverse-transpose formula (`S^-1 R`), the lighting equation written again from the
section above, and exact intersection tests:

* ground: ray / plane `y = 0`, normal `(0, 1, 0)`, the checker tile from the hit point;
* smooth spheres: ray / sphere, normal = `(hit - centre) / r`;
* crystals: the ray is mapped into object space (`R^T S^-1 (p - c)`), slab test against the unit cube, face
  normal `S^-1 R n` (correct) or `S R n` (naive);
* coarse spheres: exact ray / triangle (Moller-Trumbore) against the same 192 triangles, then either the
  vertex colours interpolated with the barycentric weights (Gouraud) or lighting of the interpolated
  position and normal (per fragment). Perspective-correct interpolation of a varying equals the barycentric
  interpolation in 3D (ES 3.0 equation 3.7), so this model is exact.

**Sweeps.** Every 5th pixel whose CPU ray hits a modelled surface is compared (about 1000 ground, 140 per
sphere, 90 per crystal, 50 per coarse sphere), under five lighting configurations (ambient only,
directional only, point only, spot only, everything), plus Phong-everywhere (key `b`), the animation at
t = 2.28 s, per-fragment on both coarse spheres (key `g`), a moved camera and a reversed draw order. A
pixel is skipped when it is not *stable*: its eight neighbours at +-2 px must hit the same object (and the
same checker tile or crystal face), so no probe sits on a silhouette, a tile edge or a box edge where the
rasterization tie-break rules would matter. Grazing sphere pixels (`N.V < 0.3`) are skipped because the
polygonal sphere deviates most from the analytic one there.

**Named probes.** The pixel is found by projecting a chosen world point on the CPU; the expectation is
computed at the exact hit of the ray through the pixel centre, so projection rounding does not matter.

* *Ambient only:* the two checker tiles show `ambient * albedo A` and `ambient * albedo B`, exactly.
* *Spot light:* seven ground points from the axis to far outside the cone: on the axis, half way to the inner
  edge, just inside it (cone factor exactly 1); in the middle of the falloff (0.669) and near the outer edge
  (0.182); just outside and far outside the outer cone (factor exactly 0, and the pixel equals the
  ambient-only colour of its tile: the spot adds nothing, not even a faint tail).
* *Point light:* three ground points 1.3, 1.9 and 3.1 units from the lamp (attenuation 0.80, 0.69, 0.50), and
  the highlight centre of each front sphere, where `N` bisects the directions to the lamp and the camera.
* *Phong vs Blinn-Phong:* on each front sphere the point of the highlight centre (angle 0 from the half
  vector) and the points 0.14 and 0.28 rad away. At 0.14 rad the Phong sphere shows (185, 126, 127); the
  Blinn-Phong model would give (212, 167, 174) there, 47 steps brighter, and the Blinn-Phong sphere shows
  (208, 166, 174). Each pixel must match its own model and miss the other by at least 12 steps.
* *Normal matrix:* for every visible face of both crystals the pixel must match its own normal model and,
  where the two models differ by at least 12 steps (the -x face: (57, 103, 110) correct vs (43, 80, 87)
  naive), miss the other. With key `n` the right crystal must move to the correct values.
* *Gouraud:* on the per-vertex sphere the CPU finds the point where per-vertex and per-fragment lighting
  of the same triangle differ most (51 steps: (194, 182, 141) vs (227, 219, 192)); the GPU shows the
  per-vertex value, and after key `g` the per-fragment one.
* *Model / view / projection:* the CPU-projected centre of the point lamp and the CPU-projected axis midpoint
  of the (rotated) spot-lamp cone show their emissive colours; after the keys `d`, `w`, `q` have moved the
  camera the lamp is at the newly projected pixel, 45 pixels away from where it was.
* *Matrices:* `glGetUniformfv` of `u_model`, `u_view`, `u_proj` and `u_normal_matrix` equals the CPU matrices
  bit for bit in both programs; `demo::math::normal_matrix` equals `S^-1 R` to 1e-7 and differs from the
  naive 3x3 by 1.0.
* *Depth test and culling:* with the ground drawn last the sweep still matches (the depth test keeps the
  nearer objects); with the depth test disabled the sphere pixel shows the ground. The upward ground is drawn
  with `CullFace(BACK)` / `FrontFace(CCW)` and with `CullFace(FRONT)` / `FrontFace(CW)`, and culled (clear
  colour) with the other two combinations. All 10 404 mesh triangles are counter-clockwise seen from outside.

**Tolerances (8-bit channels).** The conversion of the float colour to 8 bits returns one of the two closest
values (ES 3.0 §2.1.6.2), which is already +-1. GLSL ES 3.00 §4.5.1 allows a few ULP in `pow`, `/`, `sqrt`;
for these formulas that is far below one 8-bit step. The measured largest difference over all sweeps on all
four drivers (gpu radeonsi, llvmpipe, zink; es-max and es-min) is **1**. Flat and exact surfaces (ground,
crystals, coarse spheres) use a tolerance of 2 (rounding plus 1 for interpolation of position and normal);
the 96 x 48 polygonal spheres use 3: the interpolated normal at the point where the ray meets a flat triangle
differs from the analytic one by less than 0.003 rad, and the steepest highlight flanks of these materials
change by 300 to 600 levels per radian, i.e. 1 to 2 more levels. Spot-cone and clear-colour probes outside the
cone use 1 (nothing but the ambient term and the rounding).

**Negative controls** (a check must be able to fail):

* the attenuation constants are changed to (1, 0, 0) on the GPU: the far ground probe moves by 31 steps away
  from the attenuated CPU value and matches a CPU model with the same constants;
* the correct normal matrix is given to the right crystal on the GPU (key `n`): its pixels move to the
  correct-normal values and away from the naive ones;
* the CPU model with a hard-edged cone differs by 35 steps in the falloff, a CPU model with the other
  specular model differs by 27-47 steps at the highlight flanks, a CPU model with the naive normals
  differs by 23 steps on the -x face; the GPU pixels agree with the right model, not the wrong one;
* depth test disabled, face culling switched, as described above.

The shaders were also mutated by hand (sign of `reflect`, Phong using `N.H`, `mat3(u_model)` instead of the
normal matrix, no renormalization, `step` instead of `smoothstep`, a 0.5 % shift of the cone angle, no
attenuation, no gamma, no ambient, a halved exponent, a wrong eye position, the checker removed, the
Gouraud cone ignored): all 15 mutants make the demo FAIL.

**Defined errors.** `glUniformMatrix4fv` on the `mat3` normal-matrix uniform and `glUniformMatrix3fv` with
`count = 2` on a non-array uniform both raise `INVALID_OPERATION` (ES 3.0 §2.12.6.1) and leave the value
unchanged; location -1 is silently ignored.

## Pitfalls

* **Normal matrix.** `mat3(model)` is only correct for rotations, translations and *uniform* scales.
  Use `transpose(inverse(mat3(model)))` and upload it as `mat3` (a `mat4` call on a `mat3` uniform is
  `INVALID_OPERATION`). Normalize after interpolation.
* **Directions.** The directional light uniform is the direction the light *travels*; the equation needs
  the opposite, surface -> light. A sign error gives a scene that is lit from the wrong side and looks
  almost plausible. The same applies to the spot axis (`dot(-L, axis)`).
* **Specular needs `N.L > 0`,** and `pow(x, y)` is undefined for `x < 0`: clamp first.
* **Spot cone:** `smoothstep` needs `cos_outer < cos_inner`; compare cosines, not angles, to avoid `acos`.
* **Blinn-Phong is not Phong with a different name:** the same exponent gives a different highlight size.
* **Gamma:** adding lights in gamma space is wrong; light in linear space, encode last.
* **Interpolated normals are not unit length.** Normalize in the fragment shader.
* **`highp` in fragment shaders:** the fragment language has no default float precision; without a
  `precision` statement the shader does not compile.
* **VAO state:** bind the vertex array before `glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ...)`; unbinding
  the element buffer while the VAO is bound removes it from the VAO.
* **Culling and negative scale:** a negative scale flips the winding; this demo uses none.
