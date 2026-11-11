# es30-occlusion-queries

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es30-occlusion-queries [--headless --validate]`

## Purpose

An occlusion query asks the GPU whether anything drawn between `glBeginQuery` and `glEndQuery` passed the depth test.
The classic use is **occlusion culling**: draw the big, cheap occluders first, then draw only the *bounding box* of each
expensive object inside a query, with color and depth writes switched off. If no sample of the box survived the depth
test, the object behind it is completely hidden and does not need to be drawn at all.

ES 3.0 has the boolean flavour of occlusion queries: `ANY_SAMPLES_PASSED` (exact: TRUE if and only if at least one
sample passed) and `ANY_SAMPLES_PASSED_CONSERVATIVE` (allowed to answer TRUE in some extra, implementation-dependent
cases in exchange for speed). Both use the generic query object API of ES 3.0 §2.14 (`glGenQueries`, `glBeginQuery`,
`glEndQuery`, `glDeleteQueries`, `glIsQuery`) and §6.1.7 (`glGetQueryiv`, `glGetQueryObjectuiv`). This demo exercises all
of it in a small city block and checks every answer against a CPU visibility model.

## What the image shows

The camera stands in a street and looks straight ahead. Two wide grey front buildings (the occluders) stand on a dark
ground plane under a light blue sky; six towers stand behind them:

| tower | where | query result |
|---|---|---|
| T0 (blue) | behind the left block | culled |
| T1 (yellow, tall) | behind the right block, but taller | visible above it |
| T2 (green) | in the street between the blocks | visible |
| T3 (purple) | behind the right block | culled |
| T4 (orange) | behind the left block's right edge | partly visible, so drawn |
| T5 (teal, tall) | close behind the left block | culled |

Visible towers are drawn normally (flat Lambert shading). Culled towers are **not drawn**; instead their bounding box
is shown as a translucent red "x-ray" box with a red outline on top of everything, so you can see what the queries
removed. With a window (`--window`), the occluders slide left and right and towers pop between culled and visible.
`space` pauses the motion, `o` removes the occluders (everything becomes visible).

The block always fits the surface: the 60° field of view spans the shorter side of the window, and the sky and ground
fill the rest.

## Central calls

```
glGenQueries(6, any)                                   // names only - no objects yet
draw ground + occluders                                // depth writes on
glColorMask(FALSE,FALSE,FALSE,FALSE); glDepthMask(FALSE)
for each tower i:
    glBeginQuery(GL_ANY_SAMPLES_PASSED, any[i])        // the first Begin creates the object
    draw tower i's bounding box                        // depth-tested only
    glEndQuery(GL_ANY_SAMPLES_PASSED)
glColorMask(TRUE...); glDepthMask(TRUE)
for each tower i:
    glGetQueryObjectuiv(any[i], GL_QUERY_RESULT, &r)   // waits for the result
    if r: draw tower i  else: draw its red x-ray box
```

`validate()` additionally uses `GL_ANY_SAMPLES_PASSED_CONSERVATIVE`, `GL_QUERY_RESULT_AVAILABLE`,
`glGetQueryiv(GL_CURRENT_QUERY)`, `glIsQuery`, `glDeleteQueries` and the `GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN`
target (only as a second query *type* for the error rules).

## Spec

- ES 3.0 §2.14 Asynchronous Queries: query names versus objects (`GenQueries` reserves names, "no object is associated
  with them until the first time they are used by BeginQuery"), the active query object name per query type, every
  `BeginQuery` / `EndQuery` error, deletion of an active query, and "the required state for occlusion queries is shared".
- ES 3.0 §4.1.6 Occlusion Queries: the samples-boolean state, set to FALSE by `BeginQuery`, set to TRUE when a sample
  passes the depth test, written as the query result; CONSERVATIVE "can additionally set the samples-boolean state to TRUE".
- ES 3.0 §6.1.7 Asynchronous Queries (state queries): `IsQuery`, `GetQueryiv(CURRENT_QUERY)`, `GetQueryObjectuiv`
  errors, `QUERY_RESULT_AVAILABLE` ordering ("if any query object returns a result available of TRUE, all queries of the
  same type issued prior to that query must also return TRUE"), "Repeatedly querying the QUERY_RESULT_AVAILABLE state ...
  is guaranteed to return true eventually", `QUERY_RESULT` forces completion.
- ES 3.0 Table 6.23 Query Object State; Appendix D.1.3 Deleted Object and Object Name Lifetimes ("Query objects are in use
  so long as they are active").
- ES 3.2 §4.2 Query Objects and Asynchronous Queries, and Appendix E.2: occlusion query objects "may be reused for either
  target" (a clarification of the ES 3.0 error rule, Bug 13342).
- ES 3.0 §4.1.5 Depth Buffer Test, §4.2.2 Fine Control of Buffer Updates (`ColorMask`, `DepthMask`), §4.1.7 Blending,
  §3.6.1 polygon rasterization and §3.5.1 line rasterization (probe margins), §2.1.6.2 color conversion (tolerances).

## Shader concepts

One program for everything: the vertex shader transforms the unit cube by an MVP matrix (each object is the cube,
translated and scaled), the fragment shader does flat Lambert shading with one directional light, or outputs the
uniform color unchanged (`u_lit = 0`) for the x-ray overlay. `layout(location)` vertex inputs, one user-defined
fragment output. The query pass uses the same program; its color output is discarded by `glColorMask`.

## How it is validated

**CPU visibility model.** The camera looks exactly along −z, so each box face perpendicular to z projects to an
axis-aligned rectangle of a single window depth. For each tower:

- *hidden* if an occluder's front face lies at least 1 world unit nearer than the tower's nearest point and that face's
  rectangle contains the tower's whole projected box with 2 pixels to spare. Every pixel there holds the occluder face
  or nearer ground (the occluders reach below the ground), so no sample can pass `DepthFunc(LESS)`;
- *visible* if at least one pixel center lies 2 pixels inside the tower's front face, inside the viewport, and 2 pixels
  outside the projection of every occluder that is not entirely behind that face;
- otherwise *ambiguous*, which is a failure: every configuration is designed to be unambiguous at every surface size
  (at least 200 pixels on the shorter side; smaller surfaces report SKIP_UNSUPPORTED).

**Query results** (6 towers × 2 targets each), in five configurations:

1. the scene, results read with the blocking `QUERY_RESULT` alone: `ANY_SAMPLES_PASSED` is TRUE exactly for the towers
   the model calls visible (3 hidden, 3 visible);
2. the scene again, results read only after `QUERY_RESULT_AVAILABLE` was polled to TRUE without any flush, with the
   ordering rule checked (all earlier queries of the same target are available too) and every query becoming available
   by polling alone;
3. **negative control:** the occluders pushed 30 units back. Their rectangles still overlap the towers on screen, but they
   are now behind them, so every tower that was culled must flip to visible;
4. no occluders: everything visible;
5. **negative control:** depth cleared to 0.0, nothing drawn. No sample can pass, so every `ANY_SAMPLES_PASSED` is FALSE,
   including the towers that are visible in the scene.

Configurations 1, 3, 4 and 5 read the results with the blocking `QUERY_RESULT` and are strict on every driver.

Every result must be exactly 0 or 1 (§4.1.6 writes "FALSE or TRUE"). `ANY_SAMPLES_PASSED_CONSERVATIVE` is checked only
where the specification defines it: TRUE whenever a sample passed. For hidden towers it may legally be TRUE, so its value
is printed, not asserted.

**Query object API**, on fresh names: `glIsQuery` FALSE for a generated name before its first `glBeginQuery`, TRUE after
it and after `glEndQuery`, FALSE after `glDeleteQueries`; `glGetQueryObjectuiv` on a generated-but-unused name, on the
active query (both pnames), and on a deleted name → `INVALID_OPERATION`; `CURRENT_QUERY` 0 outside, the query inside;
`INVALID_OPERATION` for `glBeginQuery` on an active target, on the other occlusion target while one is active, with a name
active on another target, with id 0, with a name never generated, with a deleted name, with an occlusion object on the
primitives target and with a primitives object on an occlusion target; `glEndQuery` without an active query; a failed
`glBeginQuery` creates and activates nothing. An occlusion query and a primitives query can be active together.
`glDeleteQueries` ignores 0 and unused names. A query with no draw inside is FALSE.

**The frame:** `render()`'s path draws exactly the towers the model calls visible, a probe pixel of each visible tower
equals the CPU-shaded tower color (±1, one float to 8-bit conversion), and a probe pixel of each culled tower equals the
occluder color blended once with the red overlay (±2, two conversions). Probes keep 2 pixels from every rectangle edge
and from every outline segment.

## Known driver deviations (XFAIL)

- **Mesa (all drivers): deleting an active query ends it.** ES 3.0 §2.14: "If an active query object is deleted its name
  immediately becomes unused, but the underlying object is not deleted until it is no longer active". Nothing resets the
  active query name, so the query stays active until `glEndQuery`. Mesa's `glDeleteQueries` performs an implicit end:
  `CURRENT_QUERY` reads 0 and the following `glEndQuery` raises `INVALID_OPERATION`.
- **Mesa (all drivers): occlusion objects cannot switch occlusion targets.** ES 3.2 §4.2 (clarifying the ES 3.0 rule):
  "occlusion query objects specified by either of the two targets ANY_SAMPLES_PASSED or ANY_SAMPLES_PASSED_CONSERVATIVE
  may be reused for either target". Mesa raises `INVALID_OPERATION` ("target mismatch"), and so does ANGLE (XFAIL on "ANGLE").
  Because two independent implementations agree, the reading was re-checked: ES 3.0 §2.14 already says "There are two query
  types supported by the GL" (primitive and occlusion), and the ES 3.1 change log (update of January 29, 2015, repeated in
  ES 3.2 Appendix E.2) states the intent: "make clear that occlusion query objects may be specified and reused with any mix of
  the two valid occlusion query targets (Bug 13342)". Both implementations key the type on the target. ANGLE does keep a
  deleted active query active, as §2.14 requires.
- **zink: polled results.** Once `QUERY_RESULT_AVAILABLE` has been polled to TRUE, `QUERY_RESULT` returns a partial
  sample count (for example 30 or 196 for a box covering thousands of pixels, and 0 for a visible box) instead of the
  samples-boolean; `glFinish` does not change it. zink also reports earlier queries unavailable while a later one is
  available, against §6.1.7. Results read with the blocking `QUERY_RESULT` alone are correct and checked strictly.

## Run

```
es30-occlusion-queries                       # window: sliding occluders; space pauses, o toggles the occluders
es30-occlusion-queries --headless --validate --capture out.png
es30-occlusion-queries --headless --validate --width 333 --height 777
```

## Pitfalls

- **Reading the result in the same frame stalls the pipeline.** `QUERY_RESULT` waits until the GPU has finished the box
  draws. The demo does it for clarity; a real renderer reads last frame's result (or polls `QUERY_RESULT_AVAILABLE` and
  keeps the previous decision while it is FALSE). §6.1.7 itself recommends waiting one to three frames.
- **Turn the writes off for the box pass.** With depth writes on, a box would occlude the boxes queried after it; with
  color writes on, the boxes would show up in the image.
- **Bounding boxes, not objects.** A box that passes means the object *may* be visible; the result is conservative in the
  right direction as long as the box encloses the object.
- **Names are not objects.** `glIsQuery` is FALSE and `glGetQueryObjectuiv` fails for a name from `glGenQueries` until
  the first `glBeginQuery`. The first `glBeginQuery` also fixes the query type (occlusion or primitives).
- **One occlusion query at a time.** `ANY_SAMPLES_PASSED` and `ANY_SAMPLES_PASSED_CONSERVATIVE` share their active state:
  beginning either while the other is active is `INVALID_OPERATION`. Queries of different types (occlusion and primitives)
  may overlap.
- **CONSERVATIVE may say TRUE for a hidden object.** Use it for culling decisions where a false positive only costs time,
  never for a decision that must be exact.
- **Do not trust a polled result blindly on every driver** (see the zink deviation above).
