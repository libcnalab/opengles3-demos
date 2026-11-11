# es32-tessellation-domains

**Minimum:** OpenGL ES 3.2 · **Shaders:** GLSL ES 3.20 (`#version 320 es`) · **Run:** `es32-tessellation-domains [--headless --validate]`

## Purpose

Tessellation adds two programmable stages and one fixed-function stage between the vertex shader and the rasterizer. The
application draws **patches** (`PATCHES`, a fixed number of vertices each, `glPatchParameteri(PATCH_VERTICES)`). A
**tessellation control shader** (TCS) runs once per vertex of the *output* patch and decides how finely the patch is cut: it
writes `gl_TessLevelOuter[]` (the number of segments of each edge) and `gl_TessLevelInner[]`. The **primitive generator**
subdivides a triangle, a quad or a set of isolines accordingly, and the **tessellation evaluation shader** (TES) runs once per
generated vertex with its position in the domain, `gl_TessCoord`, and turns it into a real vertex.

This demo is about the generator itself: the three domains (`triangles`, `quads`, `isolines`), the three edge spacings
(`equal_spacing`, `fractional_odd_spacing`, `fractional_even_spacing`), the winding of the generated triangles (`cw`, `ccw`)
and `point_mode`. What the generator produces is *specified* (ES 3.2 §11.2.2), so the demo does not just look at a picture:
it captures the TES output with transform feedback and compares the generated vertices and primitives with a CPU
implementation of the subdivision rules.

## Reading the image

The 256 × 256 window holds nine cells, one tessellated patch each, bent into a bump (the TES adds a smooth dome to the
height) and viewed from above at an angle so that the structure is visible in 3D.

| | `equal_spacing` | `fractional_odd_spacing` | `fractional_even_spacing` |
|---|---|---|---|
| **triangles** | `ccw`, levels outer (2,3,4) inner 4 | `cw`, outer (2.6,3.4,4.2) inner 3.4 | `ccw`, outer (2.5,3.5,4.6) inner 3.4 |
| **quads** | `cw`, outer (2,3,4,3) inner (4,3) | `ccw`, outer (2.6,3.4,3.2,4.4) inner (3.4,2.6) | `cw`, outer (2.5,3.5,2.6,3.6) inner (3.4,2.5) |
| **isolines** | 4 lines × 6 segments | 4 lines × 5 segments (level 4.2) | 4 lines × 6 segments (level 4.2) |

Each cell is drawn in three passes:

1. **The generated triangles**, filled. The colour comes from `gl_FrontFacing`: **teal = counter-clockwise** (front-facing with the
   default `glFrontFace(CCW)`), **orange = clockwise**. That is the winding layout made visible.
2. **A wireframe** of the generated triangles. OpenGL ES has no polygon mode, so a small geometry shader (`wire.geom`) turns every
   triangle into a closed line strip. Isolines are lines already and need no geometry shader.
3. **The generated vertices as white dots** (the second program uses `point_mode`; its output is captured with transform feedback
   and re-drawn as points by a plain vertex shader). The extra pass is necessary: in core ES 3.2 the point size is 1.0 unless the
   *last vertex processing stage is a vertex shader* (§13.5), so a point mode TES cannot make big dots itself.

What to look for: in the equal column the dots are evenly spaced along each edge and on concentric inner triangles / a regular
inner mesh; in the fractional columns the spacing is regular except for two short segments that sit around the **middle** of
each edge (their length shrinks as the level gets further from the rounded value); the quad's inner mesh follows
`IL0` along the horizontal edges and `IL1` along the vertical ones; different outer levels per edge make the edges differently
fine.

## The pipeline

`patch.vert` passes the application's patch vertices on. `domain.tesc` is the control shader:

```glsl
layout(vertices = OUT_VERTICES) out;      // output patch size = number of TCS invocations
...
gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
t_pos[gl_InvocationID] = ...;             // per-vertex outputs, written at the invocation's own index
barrier();                                // all invocations have written; now invocation 0 may read t_pos[0..N-1]
if (gl_InvocationID == 0) {
    p_center = vec4(centroid of t_pos[], float(gl_PatchVerticesIn));   // per-patch output (patch out)
    gl_TessLevelOuter[0..3] = u_outer + u_bias * float(gl_PrimitiveID);
    gl_TessLevelInner[0..1] = u_inner + ...;
}
```

The `barrier()` is not decoration: invocation 0 reads what the other invocations wrote, which is undefined without it
(GLSL ES 3.20 §8.15). The centroid is computed there and travels to the TES as a `patch out` / `patch in` variable. The levels
come from uniforms, so the application can ask for any combination; `u_bias` adds `gl_PrimitiveID` to every level so that the
patches of one draw call get different levels.

`domain.tese` has the input layout `layout(TESS_LAYOUT) in;`. The application inserts a `#define TESS_LAYOUT quads,
fractional_odd_spacing, cw, point_mode` (and the matching domain macro) right after the `#version` line, so one source file
serves all 30-odd layout variants; the same trick sets the TCS output patch size (`OUT_VERTICES`). The TES interpolates
the patch corners with `gl_TessCoord` (barycentric for triangles, bilinear for quads and isolines), adds the bump and projects.
It also writes `gl_TessCoord`, `gl_PrimitiveID`, the levels it sees, `gl_PatchVerticesIn`, the patch centroid and `gl_Position` to
varyings that the validation captures.

## Central API calls

`glPatchParameteri(PATCH_VERTICES, n)` · `glDrawArrays(PATCHES, first, count)` · `glGetProgramiv(TESS_CONTROL_OUTPUT_VERTICES |
TESS_GEN_MODE | TESS_GEN_SPACING | TESS_GEN_VERTEX_ORDER | TESS_GEN_POINT_MODE)` · `glTransformFeedbackVaryings` /
`glBeginTransformFeedback(TRIANGLES | LINES | POINTS)` · `glBeginQuery(PRIMITIVES_GENERATED | TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN)`
· `glGetIntegerv(MAX_TESS_GEN_LEVEL | MAX_PATCH_VERTICES)`.

## Spec

ES 3.2: §10.1.12 Separate Patches · §11.2 Tessellation, §11.2.1 Tessellation Control Shaders, **§11.2.2 Tessellation Primitive
Generation** (with §11.2.2.1 triangles, §11.2.2.2 quads, §11.2.2.3 isolines and Figures 11.1-11.4), §11.2.3 Tessellation Evaluation
Shaders · §7.12 program queries · §12.1, §12.2.2 and Table 12.1, §12.3 transform feedback and primitive queries · §13.5 point size ·
§13.7.1 face culling · Tables 21.45 / 21.46 limits. GLSL ES 3.20: §4.4.1 *Tessellation Evaluation Inputs* (primitive mode, spacing,
ordering, `point_mode`, defaults), §4.4.2 *Tessellation Control Outputs* (`vertices = N`), `patch in` / `patch out` (§4.3.4, §4.3.6),
§7.1.2 / §7.1.3 the TCS / TES built-ins, §8.15 `barrier()`.

## The CPU model (what the spec says, written as code)

* **Levels.** `equal_spacing`: clamp to [1, max], round up to the integer *n*. `fractional_even_spacing`: clamp to [2, max], round up
  to even. `fractional_odd_spacing`: clamp to [1, max − 1], round up to odd. A patch is **discarded** if a relevant outer level is ≤ 0
  (triangles: outer 0-2, quads: outer 0-3, isolines: outer 0-1). A negative inner level is clamped, not discarded. An inner level of
  1 next to an outer level above 1 counts as 1 + ε (2 for equal and even spacing, 3 for odd).
* **Triangles.** The three edges (OL0 on u = 0, OL1 on v = 0, OL2 on w = 0) have OL segments. Inside are concentric triangles: with
  first inner level *n* ring *r* has *n − 2r* segments per edge, its corners sit at barycentric (a, a, 1 − 2a) with a = 2r / 3n and
  permutations, a ring of 0 segments is the centre point. Vertices: the boundary count plus 3(n−2) + 3(n−4) + … (+ 1 if *n* is even).
* **Quads.** Outer levels on u = 0 / v = 0 / u = 1 / v = 1. The inner mesh is the grid u = i/IL0, v = j/IL1 without its border; **IL0
  divides the horizontal axis (u)**, as Figure 11.3 shows for the levels (4, 2) and (7, 4).
* **Isolines.** `round(OL0)` lines at v = k/n (k < n), each cut into `round(OL1)` segments by the spacing; the spacing is ignored for
  the number of lines.
* **Triangle count.** For a triangulated disk with *V* vertices of which *B* are on the boundary, *F = 2V − B − 2* (Euler). The text
  requires triangles that cover the domain completely, without overlap, made from the subdivided edges: exactly that.

## How it is validated

Every capture draws with `RASTERIZER_DISCARD`, transform feedback and the two primitive queries; the records are mapped and compared.

* **Equal spacing, exact** (24 level combinations, plus the cw/ccw twins): the set of distinct `gl_TessCoord`s equals the CPU set, in both
  directions, with the exact number of vertices and primitives (`TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN` = `PRIMITIVES_GENERATED` = CPU
  count) — all-ones, inner 1 / 2 / 3 / 4, rounding of 2.1 / 3.5, outer < 1, degenerate inner rectangles and Figure 11.3 / 11.4 cases.
  Tolerance 1e-3 on coordinates: the spec gives no precision and fixed-point tessellators snap to 2⁻¹⁶; the measured largest
  deviation is 2.5e-05 on radeonsi, llvmpipe and zink (printed by every run). The distinct-vertex tolerance is 1e-4.
* **Winding and coverage:** the signed area of every generated triangle in the (u, v) plane of Figure 11.1 has the sign the layout asks
  for (`ccw` positive, `cw` negative), no triangle is degenerate, and the areas sum to the domain area (1/2 for triangles, 1 for quads).
* **Point mode:** one point per distinct vertex — the same set as the triangles or lines of the same patch, point count = vertex
  count (`TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN` in POINTS mode).
* **Fractional spacings** — only what the text defines: the counts follow from the rounded levels; every outer edge is cut into
  *n* segments, *n − 2* of one length and two of equal length, symmetric about the middle; with *n − f* = 0 all segments are equal; the
  ratio of the two additional segments to the others **decreases strictly with *n − f*** (measured: 1.00, 0.68, 0.42, 0.19, 0.07 for odd
  and 1.00, 0.67, 0.40, 0.18, 0.07 for even spacing at *n − f* = 0, 0.5, 1, 1.5, 1.8); opposite quad edges with equal levels are cut at
  identical positions. The *position* of the short segments is explicitly undefined and not asserted.
* **Level rules:** discarding (6 cases, both queries 0), irrelevant levels not discarding, negative inner levels clamped, the maximum:
  with `MAX_TESS_GEN_LEVEL` = 64 an isoline count of 164.5 gives exactly 64 lines, an odd segment count above the maximum gives 63, an
  even one 64; 17 hand-written rounding examples (`1.01 → 2`, `2.0 odd → 3`, `3.01 odd → 5`, `2.01 even → 4`, …).
* **Pipeline data:** `gl_PrimitiveID` (3 patches in one draw get levels 1, 2, 3 → 2, 8, 18 triangles), the levels as the TES sees them
  (unclamped, as written), `gl_PatchVerticesIn`, the `barrier()`-dependent centroid in a `patch out` variable, `gl_Position` against the
  CPU's TES; the patch sizes 3, 4, 5 (with a TCS output patch of 3), `MAX_PATCH_VERTICES` in and out, and vertices after the last full
  patch ignored.
* **State and errors:** the five `TESS_*` program queries for 19 programs, plus the defaults (a layout naming only `triangles` gives
  `EQUAL` and `CCW`); `INVALID_OPERATION` for non-`PATCHES` draws with a tessellation program, `PATCHES` without one, the queries on a
  program without the stage, and `BeginTransformFeedback` modes that do not match the TES output (Table 12.1); `INVALID_VALUE` for
  `PATCH_VERTICES` 0, −1 and `MAX + 1`, with the previous patch size still in effect; programs with `vertices = MAX + 1`, `vertices = 0`
  or no primitive mode do not build.
* **Pixels:** every CPU-computed vertex of the three equal-spacing cells appears as a white dot on its CPU-projected pixel (the
  projection repeats the TES arithmetic and the viewport transform; 16 + 18 + 28 = 62 dots); the filled triangles are teal exactly in the `ccw` cells
  and orange in the `cw` cells; the isoline cell shows its four lines and nothing between them; the cell corners keep the clear colour.
  A cull-face probe draws the same quad with `ccw` and `cw` layouts under `FrontFace(CCW|CW)` and `CULL_FACE(BACK)` and expects the patch
  exactly where the windings agree.
* **Negative controls (a check that cannot fail proves nothing):** the CPU model with `IL0` and `IL1` swapped is rejected by the captured
  vertices (deviation 0.14), a `ccw` capture is accepted as `ccw` and rejected as `cw`, and the corner pixels prove that "drawn here"
  is distinguishable from "not drawn here".
* **`gl_TessCoord` invariant** (GLSL ES 3.20 §7.1.3): all captured values (8466 in the default run) satisfy `x == 1.0 - (1.0 - x)` in float arithmetic.

## Findings and spec notes

* The three Mesa drivers (radeonsi, llvmpipe, zink) agree with the model to the last vertex, for all of the above; no deviation had to
  be recorded.
* **Figure 11.3 versus the prose of §11.2.2.2.** The text says the *u = 0 and u = 1 (vertical)* edges are cut using the *first* inner
  level, which would make `IL0` divide `v`. Figure 11.1 (the labels `IL0`, `IL1`) and Figure 11.3 (levels (4, 2) draw four horizontal and
  two vertical segments) say `IL0` divides `u`. The demo follows the figures (and the hardware); the negative control above pins it down.
* **`gl_PatchVerticesIn` in the TES** is the size of the TES's input patch, i.e. the **TCS output patch size** (§11.2.3: "equal to the
  tessellation control shader output patch size"), not `PATCH_VERTICES`. In the TCS it is `PATCH_VERTICES`. The demo captures both.
* `PATCH_VERTICES` is not an entry of the ES 3.2 state tables and meta-gl has no `GetParameter` for it, so its value is verified through
  its effects (`gl_PatchVerticesIn`, number of patches), not read back.
* GLSL ES 3.20 numbers the tessellation sections differently from the briefing note (§4.4.1.1 / §4.4.2.1 / §7.1.5-7.1.6): in the
  document used here the layouts are the unnumbered subsections *Tessellation Evaluation Inputs* of §4.4.1 and *Tessellation Control
  Outputs* of §4.4.2, and the built-ins are §7.1.2 and §7.1.3.
* On softpipe (a native ES 3.1 driver) the program cannot run: creating an ES 3.2 context fails, so the run is skipped.

## Pitfalls

* `barrier()` must be called unconditionally in `main()` of a TCS, never inside `if`.
* `gl_TessLevelOuter[3]` and `gl_TessLevelInner[1]` are undefined for triangles (and the third and fourth outer levels for isolines):
  reading them in the TES gives garbage; the demo compares only the levels that exist.
* `layout(vertices = N)` and `PATCH_VERTICES` are independent. Indexing `gl_in[]` beyond the patch size is undefined.
* A tessellation program draws only `PATCHES`, and `PATCHES` need a tessellation program.
* Point size is 1.0 in a tessellation or geometry shader; large dots need a vertex-shader pass.
* The winding is a property of the **TES layout**, not of the vertex order of the patch; with `FrontFace(CCW)` a `ccw` layout yields
  front faces only if the TES keeps the orientation of (u, v) (here the mapping is orientation-preserving).
* The tessellator guarantees nothing about the order of triangles or vertices, nor where fractional segments are; compare sets, not
  sequences.

## Controls (windowed)

`a` toggle the camera orbit · `w` wireframe · `p` vertex dots · `+` / `-` scale all tessellation levels · `r` reset.
