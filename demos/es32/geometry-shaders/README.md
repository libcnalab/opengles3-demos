# es32-geometry-shaders

Geometry shaders in OpenGL ES 3.2: a programmable stage between the vertex and the fragment shader that sees one whole
primitive (with its neighbours, for the adjacency types) and emits any number of new primitives of one output type.
The demo draws four small 3D scenes, one per use of the stage, and then proves with a CPU model, with transform
feedback and with the error rules of the specification that the stage does what the specification says.

**Minimum version:** OpenGL ES 3.2 and GLSL ES 3.20 (`#version 320 es`). ES 3.1 has no geometry shaders. On a
native ES 3.1 driver (softpipe) the executable reports `SKIP_UNSUPPORTED`.

## What you see (256x256, a 2x2 grid of 128x128 scenes)

| cell | input -> output | what the geometry shader does |
|---|---|---|
| top left (A) | `points` -> `triangle_strip, max_vertices = 4` | 151 POINTS (a spiral galaxy) become camera-facing quads; the fragment shader discards outside the unit circle and shades a sphere. The palette entry is `kPalette[gl_PrimitiveIDIn % 6]`. |
| top right (B) | `triangles` -> `triangle_strip, max_vertices = 15` | each of the 80 faces of an icosphere becomes a prism: a lifted cap (3 vertices) and three side quads (4 vertices each), separated by `EndPrimitive()`. The lift is the unnormalised face normal times a factor taken from `gl_PrimitiveIDIn % 3`; the fragment shader colours by the built-in `gl_PrimitiveID` that the geometry shader wrote. |
| bottom left (C) | `triangles_adjacency` -> `triangle_strip, max_vertices = 12` | a torus is first drawn normally (vertex + fragment shader), then again with an index list that carries each triangle's three neighbours. A front-facing triangle emits a thin screen-space quad for every edge whose neighbour faces away: the silhouette. |
| bottom right (D) | `triangles, invocations = 6` -> `triangle_strip, max_vertices = 3` | ONE triangle list (a cube) is placed six times: invocation `gl_InvocationID` rotates and moves copy k from the uniform arrays `u_place[k]`, `u_spin[k]`. |

Every program passes its data between the stages in interface blocks (`VS_OUT`, `GS_OUT`); the block *names* must match
across a stage boundary, the instance names (`gs_in[]`, `gs_out`, `fs_in`) are local. Animated in windowed mode
(time = frame / 60, frame 0 is the capture).

## Central calls and where the specification talks about them

* `glCreateShader(GL_GEOMETRY_SHADER)`, `glAttachShader`, `glLinkProgram`: ES 3.2 section 7.3. A program with a geometry
  shader must also have a vertex shader (unless separable), and the geometry shader must specify the input type, the output
  type and `max_vertices` or the link fails (sections 11.3.1 and 11.3.2).
* `glGetProgramiv` with `GEOMETRY_VERTICES_OUT`, `GEOMETRY_INPUT_TYPE`, `GEOMETRY_OUTPUT_TYPE`,
  `GEOMETRY_SHADER_INVOCATIONS`: section 7.12. They raise `INVALID_OPERATION` for a program without a geometry shader.
* Input primitive types and the draw modes they accept: section 11.3.1 and 11.3.4.6. `points` accepts POINTS; `lines`
  accepts LINES, LINE_STRIP and LINE_LOOP; `lines_adjacency` accepts LINES_ADJACENCY and LINE_STRIP_ADJACENCY; `triangles`
  accepts TRIANGLES, TRIANGLE_STRIP and TRIANGLE_FAN; `triangles_adjacency` accepts TRIANGLES_ADJACENCY and
  TRIANGLE_STRIP_ADJACENCY. Anything else raises `INVALID_OPERATION` and renders no fragment.
* Adjacency vertex order: sections 10.1.8 (`LINES_ADJACENCY`, vertices 1 and 4 are the neighbours), 10.1.9, 10.1.10
  (`TRIANGLES_ADJACENCY`: vertices 1, 3, 5 are the triangle, 2, 4, 6 the neighbours across the edges 1-3, 3-5, 5-1).
* `gl_PrimitiveIDIn` (section 11.3.4.3) counts primitives, not adjacency vertices; `gl_InvocationID` and the output order
  of instanced shaders (section 11.3.4.2: invocation 0's primitives first, all of one input primitive before the next).
* Transform feedback captures the **last vertex stage** (sections 11.1.2.1, 12.1, 12.2): the geometry shader's outputs.
  Block members are named `Block.member` (`GS_OUT.eye`). Triangle strips are decomposed into triangles in the order of
  section 10.1.5 (triangle i is (i, i+1, i+2) for even i and (i+1, i, i+2) for odd i). When a geometry shader is active
  the capture mode is compared with its **output** type, not with the draw mode (section 12.2.2).
* `PRIMITIVES_GENERATED` (section 12.3) counts what the geometry shader emitted, triangle by triangle.
* GLSL ES 3.20: layout qualifiers of geometry inputs and outputs (4.4.1, 4.4.2), `EmitVertex` / `EndPrimitive` (8.13),
  `gl_in[]`, `gl_PrimitiveIDIn`, `gl_InvocationID`, `gl_PrimitiveID` (7.1.4, 7.1.5). The inventory cites the function
  section as 8.16; in the 3.20 text used here it is 8.13.

## Shader concepts worth reading

* `shaders/billboard.geom` - one point in, one four-vertex strip out; `flat` block members; `gl_Position` of the vertex
  shader carries eye-space data because the geometry shader does the projection.
* `shaders/crystal.geom` - three `EndPrimitive()` calls per input triangle; only exact operations (subtract, cross product,
  multiplication by 0.5/1/1.5) so that the output can be compared bit for bit.
* `shaders/silhouette.geom` - the six `gl_in` entries of `triangles_adjacency`, `faces_eye()` on the triangle and on its
  neighbour `(b, a, c)`; the outline quad is built in clip space with a small z bias so it wins the depth test against the
  surface it lies on.
* `shaders/ring.geom` - `layout(invocations = 6)`; per-copy data from uniform arrays indexed by `gl_InvocationID`.

## How it is validated (`--validate`)

1. **Limits and queries.** All twelve `MAX_GEOMETRY_*` values are at least the minimum of ES 3.2 Table 21.47. The four
   programs report the input type, output type, vertices out and invocations that their sources declare; five small
   programs (one per input type) do the same; the four pnames on a program without a geometry shader raise
   `INVALID_OPERATION`.
2. **Pixels.** A CPU model re-implements the four geometry shaders (`cpu_model.hpp`), rasterizes the emitted triangles
   (perspective-correct, depth-tested) and is compared with every pixel of the cell. A pixel is only compared when its
   expected colour is unambiguous: every triangle that certainly covers it (edges at least 0.75 px away) must give the
   result, no nearer triangle that *might* cover it (within 0.75 px of an edge, because vertex positions are only specified
   to `SUBPIXEL_BITS` >= 4, i.e. 1/16 px) may have a colour more than 3/255 away, and a 1/16 px shift of the sample point
   may not change the colour by more than 2/255 (this drops the rim of the sphere impostors, where the shading has an
   unbounded slope). The tolerance is 3/255: 8-bit rounding (1) plus slack for the GPU's `normalize`, `sqrt` and `pow`,
   for which the specification gives no precision. Measured on radeonsi: at most 1/255 (cells A to C) and 2/255 (cell D)
   over about 11 600 to 14 500 compared pixels per cell; 11 to 29 percent of the pixels are skipped as ambiguous.
3. **Transform feedback, bit for bit.** The same programs capture their geometry shader output (varyings
   `GS_OUT.eye`, `GS_OUT.world`, `GS_OUT.normal`, `GS_OUT.edge`, `GS_OUT.side`, `GS_OUT.inst`, `GS_OUT.face`, ...).
   The input data is dyadic (positions are multiples of 1/16 or 1/32, sizes and lift are powers of two) and the exact
   passes use matrices whose entries are 0, 1 or such fractions, so every operation is exact in single precision and the
   CPU model must match bit for bit (no tolerance). Cell C captures vertex positions that are plain copies, cell D captures
   `gl_InvocationID` and `gl_PrimitiveIDIn` as integers, which proves the order of the six invocations. The number of
   captured words is exact (a sentinel after the buffer stays untouched), `PRIMITIVES_GENERATED` equals
   `TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN` and the number of triangles the model emits.
4. **Adjacency vertex order.** A geometry shader that writes `gl_VertexID` of every `gl_in[k]` and `gl_PrimitiveIDIn`
   shows, through transform feedback, which vertices LINES_ADJACENCY (8 vertices: two segments), LINE_STRIP_ADJACENCY
   (7 vertices: four segments, 3 vertices: none), LINES_ADJACENCY with 7 vertices (trailing vertices ignored) and
   TRIANGLES_ADJACENCY (12 vertices) deliver.
5. **Mismatch rule.** All 5 input types x 11 draw modes: exactly the 11 compatible pairs draw without an error, the other
   44 raise `INVALID_OPERATION` (rasterization discarded). In a 16x16 target a matching draw fills the framebuffer, a
   mismatching draw leaves it untouched.
6. **Transform feedback mode rule.** `BeginTransformFeedback(TRIANGLES)` with a GL_POINTS draw of a triangle_strip geometry
   shader is legal; `POINTS` and `LINES` raise `INVALID_OPERATION` at the draw.
7. **Link rules.** A valid control and `max_vertices = 256`, `invocations = 32` link. These do not link: no input layout,
   no output type, no `max_vertices`, no vertex shader, `max_vertices` above the limit, `gl_in[3]` for triangles, a block
   member type mismatch between VS and GS, a GS input block without a VS output, a GS input block array of the wrong size.
   `invocations` above the limit or zero must fail at *compile* time (GLSL ES 3.20 4.4.1). The programs fail at link or
   compile time as noted in the log.

**Negative controls** (each check can fail): the CPU palette shifted by one entry (A), prisms 25 percent taller (B), an
outline 2.4 times thicker (C), the invocation index shifted by one (D) all break the pixel comparison; one flipped bit in
the model breaks each transform feedback comparison.

## Pitfalls and findings

* `UseProgram` raises `INVALID_OPERATION` while transform feedback is active; set the program up before
  `BeginTransformFeedback` and issue only the draw call inside.
* Interface blocks match by block name; `flat` and `int` members need `flat` on the outputs and the inputs.
* A geometry shader output is only well defined after `EmitVertex()` for the members that were written *before* that call
  (all members are undefined afterwards), so the helper functions write every member each time.
* The strip order for transform feedback is (i, i+1, i+2) / (i+1, i, i+2); Mesa follows section 10.1.5.
* **Known deviation (Mesa: radeonsi, llvmpipe and zink):** `max_vertices` x output components above
  `MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS` still links (ES 3.2 section 11.3.4.4 says LinkProgram will fail). The check is
  strict and reported as XFAIL on "Mesa".
* The adjacency index list must be built from a closed, consistently wound manifold: each directed edge must have exactly
  one owner (`cpu_model.hpp: adjacency_indices`).
* Mesa reports `LAYER_PROVOKING_VERTEX` = `UNDEFINED_VERTEX` here; see `es32-layered-rendering`.

Run: `build/debug/bin/es32-geometry-shaders --headless --validate --capture out.png`
(without `--headless` a window opens; the scenes animate).
