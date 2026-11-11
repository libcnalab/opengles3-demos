# es32-tessellation-terrain

**Minimum:** OpenGL ES 3.2 · **Shaders:** GLSL ES 3.20 (`#version 320 es`) · **Run:** `es32-tessellation-terrain [--window] [--headless --validate]`

## Purpose

A terrain is the classic use of tessellation: a coarse grid of patches that the GPU refines where the detail matters and
displaces with a height map. Here a 64 × 64 world-unit landscape is a 16 × 16 grid of **quad patches** (1024 patch vertices, one
`glDrawArrays(PATCHES, …)`), and three things happen inside the pipeline:

1. The **tessellation control shader** (`terrain.tesc`) chooses the level of detail of every patch edge from its distance to the
   camera: `level = clamp(u_lod_scale / distance, 1, u_max_level)`. The distance is measured to the middle of the *displaced* edge,
   so the shader fetches the height map itself (texture access in the TCS, §11.2.1.1).
2. The **primitive generator** subdivides each quad with `fractional_odd_spacing`: the mesh changes continuously with the level
   (new vertices grow out of the middle of an edge) instead of popping from one integer subdivision to the next.
3. The **tessellation evaluation shader** (`terrain.tese`) places every generated vertex on the terrain: bilinear position in the
   patch, height from the height map. The **fragment shader** lights every pixel from the same map (per-pixel normal by central
   differences), so the lighting does not depend on the level of detail — the level only changes the silhouette and how closely
   the surface follows the heights.

The **height map** is procedural (value noise, six octaves, one ridged term, lakes clamped flat), generated on the CPU and stored in
a 256 × 256 `R16F` texture (`glTexStorage2D` + `glTexSubImage2D(RED, HALF_FLOAT)`). It is uploaded as half floats computed by the CPU, so
the texture holds exactly the bits the CPU model reads. All three stages read it with `texelFetch` and do the bilinear
interpolation by hand: the result then does not depend on filtering precision, and a CPU evaluation of the same formula gives
the same number to float rounding.

**No cracks.** Two patches that share an edge must cut it identically or the terrain tears. The level of an edge is a function of
the edge's two corners only, evaluated in the same order by both patches, so the outer levels agree to the last bit; the inner levels
are free (`max` of the two opposite outer levels).

**`glPrimitiveBoundingBox` and `gl_BoundingBox`** (ES 3.2 §13.2) tell the implementation where the primitives of a patch can lie in
clip space, so that it can skip work. The box can only be an optimisation: with a **correct** box the picture must not change. The
demo renders the frame three ways and requires identical images: the default box state, a loose correct box set with
`glPrimitiveBoundingBox`, and a **tight per-patch box computed by the TCS** and written to `gl_BoundingBox` (the clip-space
min / max of the 8 corners of the patch's world-space box, which bounds every point inside) while the state box is set to something
absurdly small — a box written by the TCS replaces the state's value.

## Reading the image

Blue sky gradient on top; below it the landscape: flat blue lakes, sand and grass at low heights, grey rock on steep slopes, snow
above height 4.6, one directional sun and linear fog towards the horizon colour. The **left half** also shows the
tessellation: a translucent wireframe of the generated triangles (a geometry shader turns triangles into lines; core ES has no polygon
mode). Look for the coarse triangles on the far mountains and the finer ones close to the camera, the gradual change of
detail between them, and that no gap opens where patches of different level meet. The camera flies a loop above the terrain; the
captured frame is the one at 8 seconds of the fly-over.

## Central API calls

`glPatchParameteri(PATCH_VERTICES, 4)` · `glDrawArrays(PATCHES)` · `glTexStorage2D(R16F)` / `glTexSubImage2D(HALF_FLOAT)` ·
`glPrimitiveBoundingBox` / `glGetFloatv(PRIMITIVE_BOUNDING_BOX)` · `glGetIntegerv(MAX_TESS_*)` · `glGetProgramiv(TESS_*)` ·
transform feedback (`glTransformFeedbackVaryings`, `glBeginTransformFeedback`) with `PRIMITIVES_GENERATED` /
`TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN` queries.

## Shader notes

* GLSL has no `#include`: `terrain_common.glsl` (the height function, normal, material, fog and the sky colour) is inserted by the
  application after the `#version` line of the TCS, TES and FS together with `#define HEIGHTMAP_N` and `#define TERRAIN_HALF`, so
  all stages see one copy. The TCS bounding-box variant is the same source with `#define WRITE_BOUNDING_BOX`.
* `gl_BoundingBox` is written by **every** TCS invocation. A shader that writes it on some paths only leaves the components undefined
  on the others (§13.2). The box is the min / max over the 8 corners of the patch's world-space box of the clip-space x, y, z, w: clip
  space is a linear image of world space, so every point inside has clip coordinates inside those ranges.
* The shading model uses only `+`, `*`, `mix`, `smoothstep`, `normalize` and `length` — no `sin`, `pow` or `exp`, whose precision is
  implementation-defined — which is why a CPU evaluation can be compared to 2/255.
* The sky is an attribute-less full-window triangle (`gl_VertexID`), drawn before the terrain without depth.

## How it is validated

* **Limits.** `MAX_TESS_GEN_LEVEL` ≥ 64, `MAX_PATCH_VERTICES` ≥ 32, `MAX_TESS_PATCH_COMPONENTS` ≥ 120 and all 19 TCS / TES limits
  (`MAX_TESS_CONTROL_*` and `MAX_TESS_EVALUATION_*`: uniform components, texture image units, input / output / total components, uniform
  blocks, atomic counters and buffers, shader storage blocks, image uniforms) against the minimums of ES 3.2 Tables 21.45 and 21.46, and
  `MAX_COMBINED_TESS_{CONTROL,EVALUATION}_UNIFORM_COMPONENTS` against the Table 21.50 formula. Every run prints the table. The queries
  `TESS_CONTROL_OUTPUT_VERTICES` = 4, `TESS_GEN_MODE` = `QUADS`, `TESS_GEN_SPACING` = `FRACTIONAL_ODD`, `TESS_GEN_VERTEX_ORDER` = `CCW` and
  `TESS_GEN_POINT_MODE` = `FALSE` are read for the three terrain programs.
* **Geometry through transform feedback.** The TES output of the displayed frame (about 10 000 triangles, 30 000 records) is captured
  and compared with the CPU:
  * the levels computed by the TCS against `clamp(80 / distance, 1, 16)` (largest relative deviation 1.7e-7; the bound 1e-4 is a sanity
    bound — the spec gives shader arithmetic no accuracy);
  * the **number of triangles** (`TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN` = `PRIMITIVES_GENERATED`) against the §11.2.2 count for the CPU's
    levels rounded to odd: with *B* boundary vertices and an inner mesh of (a − 1)(b − 1) vertices a quad patch has 2V − B − 2
    triangles. Levels within 1e-3 of an odd integer, where the CPU and the GPU could round differently, make the check refuse to run
    (the smallest margin of the fixed camera is 0.0023);
  * every generated vertex against the CPU: bilinear in the corners by the captured `gl_TessCoord` (deviation 0) and the height from
    the CPU height map (5e-6, bound 1e-3: float arithmetic);
  * **no cracks**: all 480 shared edges of the 16 × 16 grid have equal outer levels and identical vertex positions on both sides.
* **The LOD depends on the distance.** The same patches are captured from four LOD viewpoints: the camera (9954 triangles), 25 units
  higher (4608), 120 units higher (512 = 256 patches × 2: every patch is a single triangle pair) and one unit above the ground at the
  origin (18224, the nearest edges clamped to the ceiling `gl_TessLevelOuter` = 16, which rounds to 17 segments). Each count equals the
  CPU's, and the sequence is strictly decreasing with distance.
* **Pixels.** The captured triangles are ray-cast on the CPU through a grid of pixels, one every 5 pixels (2601 probes at 256 × 256) (a pixel centre ray against
  the exact triangles: that is what the rasterizer samples). A probe counts when its ray hits well inside a triangle (barycentric margin
  ≥ 0.05) and the four neighbouring pixels hit a smooth surface (hit distance within 10 %): silhouettes and triangle edges are
  excluded. The colour of each such pixel — material by height and slope, lakes, sun, fog — is compared with the CPU lighting model
  (1008 probes, tolerance 2/255: 8-bit rounding of the GPU's float colour plus double against float arithmetic; the measured
  largest error is 1). Pixels whose ray and neighbours miss everything are compared with the sky gradient (1157 probes).
* **Primitive bounding box.** The state is queried (initial value, a round trip); the three renderings are bit-identical (0 pixels
  differ); a camera one second later differs in 39 373 pixels, so the comparison can fail. What happens with a deliberately wrong
  box and no `gl_BoundingBox` is only printed (§13.2: fragments outside may or may not be generated); all three Mesa drivers ignore the
  box (0 pixels differ).
* **Negative controls.** The ray-cast pixel probes against a model with the sun mirrored disagree on 898 of the 1008 probes; the
  bounding-box comparison is shown to detect a different image.

## Findings

* radeonsi, llvmpipe and zink agree with the CPU model everywhere, including the TCS-computed `gl_BoundingBox`; no deviation had to be
  recorded. None of them uses the bounding box to drop fragments (the too-small-box experiment changes nothing).
* llvmpipe and zink (lavapipe) mis-rasterize **lines whose end points lie far outside the viewport**: the first version of the wireframe
  overlay showed long streaks fanning across the sky when the geometry shader emitted the edges of terrain beside and behind the camera
  (all geometry-shader outputs are identical on radeonsi, which draws them correctly). The wireframe pass therefore drops triangles with a
  vertex more than two window half-widths outside, or behind `w = 0.5`. The validated image has no overlay and is unaffected; this is
  an observation, not an asserted check.
* `glGetIntegerv` cannot hold `MAX_COMBINED_TESS_*_UNIFORM_COMPONENTS` on these drivers (the formula gives 1.5 × 10¹⁰); the demo reads
  the limits with `glGetInteger64v`, where the value is exact.
* On softpipe (a native ES 3.1 driver) creating an ES 3.2 context fails, so the run is skipped.

## Pitfalls

* Neighbouring patches must derive an edge's level from the edge alone. Using the patch centre's distance for all four edges opens cracks.
* `fractional_odd_spacing` rounds *up to the nearest odd integer*: a level just above an odd number jumps to the next one, which is
  why the validation requires a margin to the rounding boundary.
* A texture without mipmaps needs a non-mipmapped `MIN_FILTER`, even for `texelFetch`: the default `NEAREST_MIPMAP_LINEAR` makes the
  texture incomplete and every fetch returns zero.
* Use the same `u_viewproj` the vertex positions are projected with for the `gl_BoundingBox` computation, and include the height range
  in the world-space box: a box that ignores the displacement is *incorrect* and allows an implementation to drop fragments.
* `glGetIntegerv` clamps to 32 bits; use `glGetInteger64v` for limits that can be larger.

## Controls (windowed)

`w` wireframe on the left half / the whole window / off · `l` tint patches by tessellation level (blue coarse, red fine) · `f` freeze the
LOD viewpoint while the camera keeps flying (the mesh stays, the picture moves, and the surface gets visibly coarse as you leave) ·
`b` cycle the bounding box mode (default / loose box / per-patch `gl_BoundingBox`): no visible change is the point · `+` / `-` LOD
distance scale · `space` pause · `r` reset.
