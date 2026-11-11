# es20-culling-front-face

**API introduced in:** OpenGL ES 2.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

A triangle has two sides, and the GL has to decide which one you are looking at. It does so from the
triangle's **signed area in window coordinates**, after the projection and the viewport transform,
and from one piece of state, `glFrontFace`. The result drives two things: **face culling**
(`glEnable(GL_CULL_FACE)` + `glCullFace`), which throws polygons away before rasterization, and the
fragment-shader input **`gl_FrontFacing`**, which lets a shader color the two sides differently.
This demo draws one small scene under every combination and checks each cell against a CPU model of
the rules.

## The rule (ES 3.0 §3.6.1)

1. Compute `a = 1/2 * sum(x_i * y_(i+1) - x_(i+1) * y_i)` from the *window-space* vertices
   (equation 3.6). Positive `a` means the vertices run counter-clockwise on screen (y up).
2. `glFrontFace(GL_CCW)` (initial) uses `a`; `glFrontFace(GL_CW)` reverses its sign. If the result is
   positive the polygon is **front-facing**, otherwise (including 0) **back-facing**.
3. With `GL_CULL_FACE` disabled (initial) every polygon is rasterized. With it enabled, front-facing
   polygons are rasterized only if the cull mode is `BACK`, back-facing ones only if it is `FRONT`;
   `FRONT_AND_BACK` removes every polygon. The initial mode is `BACK`.
4. Culling decides about *polygons*. **Points and lines are never culled**, not even with
   `GL_FRONT_AND_BACK`, and for their fragments `gl_FrontFacing` is always `true`: "all other
   fragments are considered front-facing" (§3.9.2.2).

Because the sign is taken in window space, *anything* that mirrors the picture reverses the winding:
a projection with a negative x or y scale, a model matrix with a negative determinant, or rendering
upside-down into a texture. A rotation by 180 degrees does not (determinant +1).

## What the image shows

256 x 256 default framebuffer, a grid of **8 columns x 7 rows**; every cell has its own `glViewport`
and its own backdrop shade. Green = the fragment shader saw `gl_FrontFacing == true`,
orange = `false`, backdrop only = nothing was rasterized.

| Column | State |
|---|---|
| 0 | `GL_CULL_FACE` off, `glFrontFace(CCW)`; `glCullFace(FRONT_AND_BACK)` is set but has no effect |
| 1 | `GL_CULL_FACE` off, `glFrontFace(CW)`; same |
| 2 / 3 / 4 | culling on, `CCW`, `glCullFace(BACK)` / `FRONT` / `FRONT_AND_BACK` |
| 5 / 6 / 7 | culling on, `CW`, `glCullFace(BACK)` / `FRONT` / `FRONT_AND_BACK` |

| Row | Primitive |
|---|---|
| 0 | triangle with its vertices listed counter-clockwise (`A, B, C`) |
| 1 | the same triangle listed clockwise (`A, C, B`) |
| 2 | the CCW triangle **mirrored in x by the vertex shader** (`u_flip = (-1, 1)`): it now runs clockwise on screen |
| 3 | the CW triangle, mirrored: counter-clockwise on screen |
| 4 | the CCW triangle **rotated by 180 degrees** (`u_flip = (-1, -1)`): winding unchanged, picture upside down |
| 5 | `GL_POINTS`: a square, drawn in every column |
| 6 | `GL_LINES`: a one-pixel line, drawn in every column |

The triangle is deliberately asymmetric so a mirror image is recognizable. Read the grid along a
column to see the effect of a state, along a row to see the effect of the primitive.

## Central calls

| Call | Role |
|---|---|
| `glFrontFace(CCW \| CW)` | which window-space orientation is "front" |
| `glCullFace(FRONT \| BACK \| FRONT_AND_BACK)` | which faces are discarded when culling is enabled |
| `glEnable/glDisable(GL_CULL_FACE)`, `glIsEnabled` | the switch |
| `glGetIntegerv(CULL_FACE_MODE, FRONT_FACE)` | state read-back |
| `glViewport` | maps NDC to each cell; a negative size is `INVALID_VALUE` |
| `glScissor` + `glClear` | per-cell backdrop |

## Spec

- ES 3.0 §3.6.1 Basic Polygon Rasterization (equation 3.6, `FrontFace`, `CullFace`, state defaults).
- ES 3.0 §3.9.2.2 Shader Inputs (`gl_FrontFacing`, "all other fragments are considered front-facing").
- ES 3.0 §2.13.1 Controlling the Viewport (window mapping; `INVALID_VALUE` for a negative size).
- ES 3.0 §3.4 Points (point size, pixel-center rule) and §3.5 Line Segments, for the probes.
- GLSL ES 3.00 §7.2 Fragment Shader Special Variables (`gl_FrontFacing`) and §7.1 (`gl_Position`,
  `gl_PointSize`).

## Shader concepts

`triangle.vert` forwards a 2D position, multiplied by the `u_flip` uniform, as clip coordinates with
`w = 1`; `facing.frag` selects `u_front_color` or `u_back_color` with the built-in input
`gl_FrontFacing` (`in bool`, GLSL ES 3.00). The same two shaders serve triangles, points (the vertex
shader writes `gl_PointSize`) and lines. A culled triangle never reaches the fragment shader at all.

## How it is validated

- **Plan.** For each of the 56 cells the CPU transforms the triangle exactly as the GPU will
  (flip, then the viewport mapping of the cell), computes the window-space signed area (equation 3.6),
  applies the front/back rule and the culling rule, and predicts one of: backdrop only, front color,
  back color. Points and lines are predicted as always drawn and always front-facing.
- **Pixels.** Four probe pixels per triangle (the centroid and three points at barycentric weights
  0.6/0.2/0.2 near the vertices, each at least 2 px from every edge, enforced at start-up) plus
  a cell corner that no primitive touches; points and lines use the pixel at their center and three
  pixels along the line. They are placed on pixel centers, so a one-pixel point or line (the only
  size ES guarantees) covers them. Flat colors are compared with a tolerance of 1 (8-bit conversion).
- **Model sanity.** One check states the four window-space signs the whole table rests on: CCW +,
  CW -, flipped by the x-mirror, unchanged by the 180-degree rotation.
- **Grid coverage.** The grid must contain front-facing, back-facing and culled cells, so a driver
  that ignored `glFrontFace` or `glCullFace` cannot pass by accident.
- **State.** Initial `CULL_FACE` (disabled), `CULL_FACE_MODE` (BACK), `FRONT_FACE` (CCW); each
  `glCullFace`/`glFrontFace` value read back; `glIsEnabled(CULL_FACE)` toggled.
- **Negative.** `glViewport(0, 0, -w, h)` and `glViewport(0, 0, w, -h)` both give `INVALID_VALUE`
  and leave the viewport unchanged, which is why mirroring must be done in the vertex shader or the
  projection.

## Run

```
./build/debug/bin/es20-culling-front-face --headless --validate --capture culling.png
```

## Pitfalls

- **Mirroring flips the winding.** A model matrix with a negative scale, a projection that flips y
  (common when targeting textures) or any reflection makes front faces back-facing. Switch
  `glFrontFace` for those draws or fix the transform; do not expect culling to "know".
- **`glCullFace` does nothing until `GL_CULL_FACE` is enabled** (columns 0 and 1).
- **`GL_FRONT_AND_BACK` is not "draw nothing"**: it culls triangles only; points and lines are still
  drawn (column 4/7, rows 5 and 6).
- **`gl_FrontFacing` is only meaningful for triangles.** For points and lines it is always true.
- **It only varies when back faces are rasterized**, i.e. with culling off or `FRONT` culling; with
  `BACK` culling every fragment you see is front-facing.
- **Zero-area triangles** are "back-facing" by the wording of the rule, but they produce no fragments.
- **Do not mirror with a negative viewport size**; it is a GL error (§2.13.1), not a reflection.
- The default is `CCW` for both OpenGL and OpenGL ES; some asset pipelines (and Direct3D-style
  matrices) use clockwise front faces, which is exactly what `glFrontFace(GL_CW)` is for.
