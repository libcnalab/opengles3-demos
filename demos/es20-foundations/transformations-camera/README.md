# es20-transformations-camera

**API introduced in:** OpenGL ES 2.0 (matrix upload with `transpose = GL_TRUE`: ES 3.0) · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

This is the entry point of the 3D path. Every vertex of every earlier demo was handed to the GL in
normalized device coordinates; here the vertices live in the coordinate system of their *object*, and
three matrices carry them to the screen:

```
object --M--> world --V--> eye --P--> clip --(/ w)--> NDC --viewport--> window
```

The demo draws one small scene (a checkerboard ground, a tumbling cube with a different color on each
face, a small cube that orbits it, four identical posts, a pyramid and an axis gizmo) with a depth buffer,
and it draws it **twice with the same camera**: through a perspective projection and through an
orthographic one. Only the projection matrix differs, and the difference is the whole lesson: parallel
lines stay parallel and equal objects stay equal in size under `ortho`, and they converge and shrink with
distance under `perspective`. The *near plane* is deliberately placed in the middle of the scene so that
the clip volume is visible: everything closer than it is cut away.

## What the image shows

256 x 256 default framebuffer, a one-pixel separator gutter, two panes of 254 x 126 pixels.
(A window that is at least twice as wide as it is tall puts the panes side by side instead.)

| Where | What |
|---|---|
| **top pane** (blue-black sky), icon *wedge* | perspective projection, vertical field of view 40 degrees |
| **bottom pane** (purple-black sky), icon *box* | orthographic projection; the box is as tall as the frustum is at the orbit target, so the target has the same size in both panes |
| tumbling cube (center) | one color per face: +X red, -X cyan, +Y green, -Y magenta, +Z blue, -Z yellow. Three faces are visible; the other three are hidden by the depth test |
| small cube | the "moon": `M_moon = T(center) * R_y(1.7 t) * T(2.3, 0.3, 0) * R(2.5 t, (1,1,0)) * S(0.28)`, a model matrix built from five factors |
| three tall orange posts, receding to the upper left | **identical** posts at three distances. Perspective: near > middle > far. Orthographic: all the same pixel height |
| the nearest orange post, cut open | the **near plane** (distance 7.6 from the eye) passes through its middle: its upper half is clipped away and you look into the hollow inside, whose faces are darker |
| ground ends in a straight line above the bottom edge | the same near plane crossing the ground: everything closer than 7.6 is gone, the sky color shows through |
| checkerboard ground | the row through the origin along X is tinted red, the column along Z blue, the origin tile is white: these are the world axes |
| pyramid | static decoration at the right; five more draw calls with the same program |
| dark square, bottom right | **axis gizmo**: the X (red), Y (green), Z (blue) axes seen through the *rotation part of the view matrix only*, with its own viewport and an orthographic projection. It shows how the camera is oriented |

Everything is lit by one fixed directional light so that the faces differ in brightness; this is
a single `max(dot(n, l), 0)` term in the vertex shader, not the subject of the demo (the lighting demo
follows).

## Interactive use

`./build/debug/bin/es20-transformations-camera --window` (add `--width 960 --height 540` for a bigger window)

| Key | Action |
|---|---|
| `W` / `S` | dolly in / out (note: the near plane keeps its absolute distance, so dollying in cuts the cube open) |
| `A` / `D` | orbit left / right around the cube |
| `Q` / `E` | lower / raise the camera |
| `P` | cycle split view -> perspective only -> orthographic only |
| `N` / `M` | move the near plane closer / farther by 0.5. Watch the foreground return |
| `F` / `G` | move the far plane closer / farther by 2. Pull it in and the back of the scene disappears |
| `R` | reset |
| `Esc` | quit |

In a window the animation follows the wall clock; in `--headless` mode and with `--frames N` the time is
`frame / 60 s`, so every run draws the same picture. The validated frame is frame 12 (t = 0.2 s).

## The transformation pipeline, written out

Matrices are **column-major**: the element in row `r`, column `c` is `m[4c + r]`. A GLSL `mat4` has the
same memory layout, which is why a `demo::math::Mat4` can be passed to `glUniformMatrix4fv` unchanged.
Vectors are columns and are multiplied from the right, so a product reads right to left:
`clip = P * V * M * p`.

**1. Model matrix** `M = T * R * S` (scale first, translate last). The cube is
`T(0, 1.9, 0) * R_y(0.8 t) * R_x(0.55)`:

```
        | 1 0 0 tx |          |  c 0 s 0 |   R_y(a): c = cos a, s = sin a
  T  =  | 0 1 0 ty |  R_y  =  |  0 1 0 0 |
        | 0 0 1 tz |          | -s 0 c 0 |   R_x(b): | 1  0   0  0 |
        | 0 0 0 1  |          |  0 0 0 1 |           | 0  c  -s  0 |
                                                      | 0  s   c  0 |
  S = diag(sx, sy, sz, 1)                              | 0  0   0  1 |
```

**2. View matrix** `V = look_at(eye, target, up)`. With `f = normalize(target - eye)`,
`s = normalize(f x up)` and `u = s x f` the camera basis, `V` first translates the eye to the origin and then
rotates the world into the camera frame, which looks down **-z**:

```
      |  s.x  s.y  s.z  -dot(s, eye) |
  V = |  u.x  u.y  u.z  -dot(u, eye) |
      | -f.x -f.y -f.z   dot(f, eye) |
      |   0    0    0        1       |
```

The camera orbits the target `(0, 1.5, 0)` at azimuth 32 degrees, elevation 17 degrees, radius 9.5.

**3. Projection matrix.** `perspective(fovy, aspect, n, f)` with `k = 1 / tan(fovy / 2)`:

```
      | k/aspect  0       0            0        |        clip.w = -z_eye   (the distance in front of the eye)
  P = |    0      k       0            0        |        z_ndc = -1 at z_eye = -n,  +1 at z_eye = -f
      |    0      0  (f+n)/(n-f)  2fn/(n-f)     |
      |    0      0      -1            0        |
```

`ortho(l, r, b, t, n, f)` is a scale and a translation that maps the box to the NDC cube, and `w` stays 1:

```
      | 2/(r-l)    0        0      -(r+l)/(r-l) |
  P = |   0     2/(t-b)     0      -(t+b)/(t-b) |
      |   0        0    -2/(f-n)   -(f+n)/(f-n) |
      |   0        0        0           1       |
```

**4. Clipping, perspective divide, viewport** (fixed function, ES 3.0 §2.18, §2.13, §2.13.1).
A vertex survives if `-w <= x, y, z <= w` in clip space. Primitives that straddle the boundary are
**clipped there, before the divide** (that is why `w` is allowed to be 0 or negative for vertices behind
the eye: they never reach the divide). Then

```
  (x_d, y_d, z_d) = (x_c, y_c, z_c) / w_c                       normalized device coordinates, [-1, 1]^3
  x_w = x + w/2 + (w/2) * x_d        y_w = y + h/2 + (h/2) * y_d   glViewport(x, y, w, h)
  z_w = (f-n)/2 * z_d + (n+f)/2                                 glDepthRangef(n, f), default [0, 1]
```

**A worked example from the validated frame.** The corner `(+0.8, +0.8, +0.8)` of the cube mesh, as
printed by the demo (perspective pane, viewport `(1, 129, 254, 126)`):

```
  object  (+0.800 +0.800 +0.800 1)
  world   (+0.965 +2.164 +0.959 1)   = M * object
  eye     (+0.310 +0.248 -8.039 1)   = V * world            (8.04 units in front of the eye)
  clip    (+0.423 +0.680 -7.033 8.039) = P * eye            (w = 8.039; inside: |x|,|y|,|z| <= w)
  ndc     (+0.053 +0.085 -0.875)     = clip.xyz / clip.w
  window  (134.68 197.33)  depth 0.0626
```

The same corner in the orthographic pane goes through the same `M` and `V`, but `w = 1`, so
`ndc = clip.xyz` and the window position is `(133.66, 68.51)` in a viewport that starts at `y = 1`.

## Central calls

| Call | Role |
|---|---|
| `glUniformMatrix4fv(loc, 1, GL_FALSE, m.data())` | uploads one stage of the chain (`u_model` per draw, `u_view` and `u_projection` per pane). `GL_FALSE` means "the data is column-major"; ES 3.0 also allows `GL_TRUE` for row-major data (ES 2.0 required `GL_FALSE`). The demo uploads both ways |
| `glViewport`, `glScissor` + `glClear` | each pane has its own viewport (the NDC to window mapping) and a scissored clear, the gizmo and the icon are two more viewports inside it |
| `glEnable(GL_DEPTH_TEST)`, `glDepthFunc`, `glClearDepthf` | hidden-surface removal. `GL_LESS` with depth cleared to 1 (the nearest surface wins) is what a scene needs; `GL_GREATER` with 0 is used by the validation to prove the depth test does what it says |
| `glGenBuffers/glBufferData/glVertexAttribPointer`, `glDrawArrays` | one buffer with ~1600 interleaved vertices (position, normal, color); each object is a `(first, count)` range |
| `glGetUniformfv` | reads a matrix back (validation) |

## Spec

- ES 3.0 §2.13 Coordinate Transformations; §2.13.1 Controlling the Viewport (the window-coordinate equations and `DepthRangef`).
- ES 3.0 §2.18 Primitive Clipping (the clip volume, polygons clipped against its boundary, new vertices created).
- ES 3.0 §2.12.6.1 Loading Uniform Variables In The Default Uniform Block (`UniformMatrix*fv`, `transpose`); §6.1.12 (matrices are returned in column-major order).
- ES 3.0 §3.6.1 Basic Polygon Rasterization (*point sampling*: a fragment is produced where its center lies inside the polygon; a shared edge produces it for exactly one of two polygons).
- ES 3.0 §4.1.5 Depth Buffer Test; §4.2.3 Clearing the Buffers; §2.1.6.2 (float to unsigned-normalized conversion: one of the two closest integers).
- GLSL ES 3.00 §4.1.6 Matrices; §5.10 Vector and Matrix Operations (`mat4 * vec4`, column-major); §7.1 Vertex Shader Special Variables (`gl_Position`).

## Shader concepts

`scene.vert` has the whole chain in four lines: `world = u_model * vec4(a_position, 1.0)`,
`eye = u_view * world`, `gl_Position = u_projection * eye`. The fourth component matters: `1.0` for
positions (so the translation columns apply), `0.0` for directions. Normals are brought to world space with
`mat3(u_model)` and `normalize`, which is only correct for rotations, translations and *uniform* scales (a
non-uniform scale needs the inverse transpose, i.e. a normal matrix: see `demo::math::normal_matrix` and the
lighting demo). `u_lit` switches the shading off for the gizmo and the icons. `scene.frag` writes the
interpolated color (`highp`, so that the 8-bit result is exact to within the conversion rounding).

## How it is validated

The run records 48 demo checks (plus the 5 runtime checks of every demo). The core is an **independent CPU model of the pipeline** (`reference_render` in
`main.cpp`): the same three float matrices are multiplied with the vertices, the polygon is clipped against the
near and far planes in clip space (Sutherland-Hodgman), divided by `w`, mapped through the viewport, rasterized
with the point-sampling rule, depth-tested and shaded in `double`. It shares nothing with the GL except the
vertex data and the matrices; the matrices themselves are checked by known answers first.

1. **Model sanity (CPU only).** `perspective()` maps the near plane to NDC z = -1 and the far plane to +1 with `w`
   equal to the distance, and the corner of the near rectangle to NDC (1, 1, -1); `ortho()` maps its box to the NDC
   cube with `w = 1`; `look_at()` sends the eye to the origin and the target to `(0, 0, -radius)`.
2. **Whole picture.** In both panes every pixel whose center is not within 0.6 px of a triangle edge (and not at an
   unresolvable depth tie) must match the CPU model within 2/255: 26169 of 32004 pixels in the perspective pane, 26286
   in the orthographic pane, on the reference run with a maximum error of 1. At least 60% of each pane must be
   comparable, so the check cannot pass by excluding everything.
3. **Named probes.** The three visible faces of the cube at a projected point near their centers, and at four points near
   the corners of each (12 probes per pane); the three **hidden** faces: their projected points must show the nearer
   face, not their own color; 8 to 9 ground tile centers of both checker colors; the three gizmo bars; the wedge and box icons;
   24 gutter pixels that prove the scissored clears did not spill.
4. **Perspective vs orthographic size.** For each of the three identical posts the CPU predicts, from the convex hull of
   its eight projected corners, which pixel rows of the column through its axis it covers, and the GPU must agree to 1
   pixel at each end: 43 / 35 / 29 px in the perspective pane (decreasing with distance), 45 / 45 / 44 in the orthographic
   pane (equal), and the demo asserts exactly these relations (`near > middle > far` and `spread <= 2`).
5. **Near-plane cut at the predicted window position.** For the nearest post and for the ground, a world line that
   crosses the near plane is intersected with it, the cut is projected, and the pixel column through it is scanned: the
   row where the colors stop must be within 1 px of the prediction (measured: within 0.2 px) in both panes. The probes
   are 8 px on the visible side and 6 px past the cut, away from the clip edge.
6. **Depth test, three ways.** Enabled: hidden faces stay hidden (3.). Disabled (`glDisable(GL_DEPTH_TEST)`): the
   whole perspective pane must match a painter's-order CPU model, and at least one hidden face's probe changes
   color. `GL_GREATER` with the depth cleared to 0: both panes match a "farthest surface wins" model and a back face
   shows its own color where `GL_LESS` hides it.
7. **Moving the clip planes.** Near plane 0.5 (triangles now reach behind the eye plane and must be clipped) and far plane
   at the middle of the cube: both panes must match a CPU model computed with those planes, and the prediction must differ
   from the default view in at least 500 pixels (so the check has power).
8. **Matrix upload.** A non-symmetric matrix is uploaded with `GL_FALSE` and with `GL_TRUE` (transposed data) and
   `glGetUniformfv` returns the identical 16 floats; the whole scene drawn with `GL_TRUE` still agrees with the CPU model.
9. **Negative controls (checks that must be able to fail).** Row-major data uploaded with `GL_FALSE` reads back
   transposed and the picture then disagrees with the CPU model in about 40000 pixels; a CPU model with the cube rotated
   0.4 rad too far disagrees in 755 pixels and fails at least one face probe; predicting the perspective pane with the
   orthographic matrix disagrees in 6562 pixels.

**Tolerances.** Colors: 2/255. ES 3.0 §2.1.6.2 converts a float color to "one of the two" nearest integers (1), and the
shader shades in `fp32` where the model uses `fp64` (1 more). Edges: ES 3.0 §3.6.1 produces a fragment where the pixel
center is inside the polygon, but vertices are snapped to a subpixel grid (at least 4 bits, `SUBPIXEL_BITS`) and computed
in `fp32`, so pixels closer than 0.6 px to an edge are excluded. Sizes and cut positions: 1 px, the pixel-center rule.
The probes were chosen away from the shared diagonal of every quad and from every edge (the reference marks the pixel
"unsafe" if another color could show there).

**Scope.** The probes are chosen for the 256 x 256 test surface and the default camera. Other window sizes
still get the whole-picture comparisons, but a probe may land behind another object (the validation prints a note, or fails,
which is then a property of the test geometry).

## Run

```
./build/debug/bin/es20-transformations-camera --headless --validate --capture camera.png
./build/debug/bin/es20-transformations-camera --headless --width 1024 --height 512 --capture wide.png
xvfb-run -a ./build/debug/bin/es20-transformations-camera --window --frames 120
```

## Pitfalls

- **Row-major vs column-major.** The most common bug. Write a matrix as a C array row by row and upload it with
  `GL_FALSE`, and the GL sees its transpose: translations land in the bottom row, every vertex ends up as garbage. The
  demo's negative control does exactly that. Either build matrices column by column (as `demo::math` does) or pass
  `GL_TRUE` (ES 3.0; not allowed in ES 2.0).
- **Order of multiplication.** `P * V * M * p` applies `M` first. `M * V * P` is a different, wrong matrix; so is
  `R * T` instead of `T * R` (rotate around the world origin instead of around the object's own center).
- **The near plane is not a free parameter.** `n` must be positive for a perspective projection (`n = 0` makes the matrix
  degenerate), and the ratio `f / n` decides the depth precision: `z_ndc` is a function of `1/z_eye`, so almost all of
  the depth range is spent close to the near plane. Choose `n` as large as the scene allows; do not use 0.001 "to be safe".
- **Clipping happens before the divide.** A triangle with one vertex behind the eye is clipped to a quad, not stretched.
  An object closer than the near plane is *cut*, not "too big": this demo puts the near plane in the scene on purpose.
- **NDC z runs from -1 to +1 in OpenGL**, from 0 to 1 in Direct3D and Vulkan. A projection matrix written for those
  maps the near half of the depth range outside the clip volume in GL.
- **The depth test needs three things**: a depth buffer in the context, `glEnable(GL_DEPTH_TEST)`, and a `glClear` of
  `GL_DEPTH_BUFFER_BIT` every frame. Forgetting the clear freezes the picture after the first frame (everything is
  "behind" what was drawn before); `GL_LESS` is the default function and works with the clear value 1.
- **Take the aspect ratio from the viewport you draw into**, not from the window: with two panes it is the pane
  (254 / 126 here), and a wrong aspect squashes the picture.
- **Normals are not positions.** `mat3(u_model)` is wrong under non-uniform scale or shear; use the inverse transpose.
  A model matrix with a negative determinant (a mirror) also reverses the winding, which matters with face culling
  (see `es20-culling-front-face`).
- **Orthographic does not mean "same size on the screen".** An orthographic box has to be sized by hand (here from the
  orbit radius); a perspective frustum is sized by its field of view. Matching them at one depth, as this demo does, is
  a choice.
