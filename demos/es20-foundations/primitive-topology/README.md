# es20-primitive-topology

**API introduced in:** OpenGL ES 2.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

The `mode` argument of `glDrawArrays` decides what a list of vertices *means*. The same six
vertices are three disconnected segments, an open path or a closed polygon outline depending on
whether they are drawn as `GL_LINES`, `GL_LINE_STRIP` or `GL_LINE_LOOP`; eight vertices are two
triangles with two vertices left over, or a ribbon, or a fan. This demo draws every primitive type
that exists in OpenGL ES 3.0 core, side by side, and verifies **both** what each topology draws
and what it must *not* draw (the gaps), against a reference computed on the CPU from the
specification.

It also covers the rasterization parameters of points and lines: `gl_PointSize`, `gl_PointCoord`,
`glLineWidth`, and the implementation-dependent ranges `GL_ALIASED_POINT_SIZE_RANGE` and
`GL_ALIASED_LINE_WIDTH_RANGE` (only size 1 and width 1 are guaranteed to exist), and the `first`
and `count` arguments of `glDrawArrays`.

## The gallery

The image is a 4x4 grid of cells, each designed as 64x64 pixels, numbered 0-15 in reading order.
Each cell has its own viewport and scissor box, so large points and wide lines cannot leak into a
neighbour. All vertices live in one buffer object; each draw selects its range with `first` and `count`.

The expected images are exact per-pixel models, so the design is never shrunk or scaled by a
fraction. It is multiplied by one integer factor `k = max(1, floor(min(width / W0, height / H0)))`
(`W0 x H0` = 256x256, 512x128 or 128x512 for the 4x4, 8x2 and 2x8 arrangements; the largest `k`
wins, ties go to the aspect ratio closest to the surface), and the scaled gallery is centred. At
800x800 the cells are 192x192 pixels (`k = 3`). Every vertex keeps the property its check relies on:

- triangles and points larger than 1 are scaled plainly (`x -> k x`, `size -> k size`), so a
  point's square still has its edges on integer coordinates and no pixel center lies on an edge;
- line endpoints and size-1 points stay at the same offset inside a pixel: design pixel `i` becomes
  the `k x k` block starting at `k i`, and the vertex moves to its center pixel `k i + k/2`. A line
  through pixel centers (horizontal, vertical or 45 degrees) therefore still runs through pixel
  centers and is still checked exactly, and a size-1 point still lies strictly inside one pixel;
- line widths and size-1 points are **not** scaled: the rules they show (width rounding, exactly
  one fragment per column, one pixel per size-1 point) are statements about single pixels.

The CPU model is built from the same scaled vertices, the "don't care" margins stay in window
pixels (1 px around triangle edges, `w/2 + 2.5` px around sloped lines), and the named probes are
taken at the scaled positions. The numbers in the table and in the check names are design pixels.
**Minimum surface:** 256x256, 512x128 or 128x512. On a surface that fits none of them the demo
reports `SKIP_UNSUPPORTED`.

| | col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|---|
| **row 1** | **0** `GL_POINTS`: seven size-1 points at fractional positions, then sizes 2-11, one vertex per point | **1** one large point drawn by a fragment shader that shows `gl_PointCoord` (red = s, green = t) | **2** `GL_LINES`: 6 vertices = 3 separate segments | **3** `GL_LINE_STRIP`: the same 6 vertices = 5 connected segments |
| **row 2** | **4** `GL_LINE_LOOP`: the same 6 vertices, closed | **5** `GL_TRIANGLES`: 8 vertices = 2 triangles, 2 vertices ignored | **6** `GL_TRIANGLE_STRIP`: 6 vertices = a ribbon of 4 triangles | **7** `GL_TRIANGLE_FAN`: pivot + 5 rim vertices = 4 triangles, the wedge stays open |
| **row 3** | **8** wide lines: widths 3, 2.6, 1.4 and 0.4, plus a vertical width-3 line | **9** the largest point the implementation supports (at most 60 px) | **10** lines of general slope, checked against the spec's rules rather than exact pixels | **11** `glDrawArrays(mode, first, count)` with `first != 0` |
| **row 4** | **12** too few vertices: `LINES` with 3, `TRIANGLES` with 4, strips and fans with 1-2, and `count = 0` for every mode | **13** a 24-sided disc as a `TRIANGLE_FAN` | **14** a `LINE_LOOP` octagon (horizontal, vertical and 45-degree edges) | **15** a house that uses all seven primitive types, one draw call each |

Cells 2, 3 and 4 differ **only** in the `mode` argument. The vertices form a staircase whose
segments alternate horizontal and vertical: `LINES` draws the three horizontal ones, `LINE_STRIP`
adds the two vertical connectors, and `LINE_LOOP` adds the closing segment from the last vertex
back to the first.

## Central calls and the rules behind them

| Topic | Calls / built-ins | Rule (ES 3.0) |
|---|---|---|
| Points | `GL_POINTS`, `gl_PointSize` | Each vertex is a separate point (§2.7.1). A point covers the pixels whose **centers** lie inside the square of side `gl_PointSize` around the vertex (§3.4.1), so a size-1 point lights exactly the pixel that contains its window position. The size is clamped to `ALIASED_POINT_SIZE_RANGE`; the maximum is only required to be at least 1; writing no size, or a size <= 0, is undefined (§3.4). |
| `gl_PointCoord` | fragment built-in | `s = 1/2 + (x_f + 1/2 - x_w)/size`, `t = 1/2 - (y_f + 1/2 - y_w)/size` (equations 3.1, 3.2): `s` grows to the right, **`t` grows downwards**. Only meaningful for points. |
| Line topologies | `GL_LINES`, `GL_LINE_STRIP`, `GL_LINE_LOOP` | `LINES`: vertex pairs, an odd last vertex is ignored. `LINE_STRIP`: segment `i` joins vertices `i` and `i+1`; a single vertex produces nothing. `LINE_LOOP`: a strip plus a closing segment (§2.7.1). |
| Line rasterization | | Diamond-exit rule (§3.5.1). Implementations may deviate from it by one pixel under four rules, e.g. at most one fragment per column of an x-major line. Lines whose endpoints are pixel centers and that are horizontal, vertical or at 45 degrees have an unambiguous result in practice and are checked exactly. |
| Line width | `glLineWidth(w)` | `w <= 0` gives `INVALID_VALUE`. The width is rounded to the nearest integer (0 acts as 1) and clamped to the maximum of `ALIASED_LINE_WIDTH_RANGE`; the line is replicated in the minor direction, so an odd width `2k+1` centered on pixel centers covers `k` pixels on each side (§3.5.2.1). |
| Triangle topologies | `GL_TRIANGLES`, `GL_TRIANGLE_STRIP`, `GL_TRIANGLE_FAN` | `TRIANGLES`: vertices `3i..3i+2`, leftover vertices ignored. `TRIANGLE_STRIP`: each vertex after the second forms a triangle with the previous two; fewer than three vertices give nothing. `TRIANGLE_FAN`: each vertex after the second forms a triangle with the **first** vertex and the previous one; the fan is not closed (§2.7.1). |
| Triangle rasterization | | Pixels whose centers lie inside the polygon are produced ("point sampling"); a center exactly on a shared edge goes to exactly one of the two polygons (§3.6.1). Culling is disabled by default, so winding does not matter here. |
| Drawing | `glDrawArrays(mode, first, count)` | Vertices `first .. first+count-1` are used; `count = 0` transfers nothing; a negative count is `INVALID_VALUE` (§2.5, §2.9.3). |

## Shader concepts

`shaders/primitive.vert` is the only vertex shader. It takes `layout(location = 0) in vec2
a_position` (clip space, `w = 1`) and `layout(location = 1) in float a_point_size`, and writes
`gl_Position` and `gl_PointSize`. It does not know which primitive the vertex belongs to;
`gl_PointSize` is read only for points and ignored for lines and triangles. The shader always
writes it because a points draw that leaves it unwritten has an undefined size.

`shaders/solid.frag` writes a flat `u_color`. `shaders/point_coord.frag` writes
`vec4(gl_PointCoord, 0.0, 1.0)`; it is a separate program because `gl_PointCoord` is undefined
for lines and triangles. Both declare `precision mediump float;`, which every GLSL ES 3.00
fragment shader needs for floats, and the user-declared output replaces `gl_FragColor`.

## Implementation-dependent limits

Only point size 1 and line width 1 are guaranteed (ES 3.0 §3.4, §3.5). The demo queries both
ranges and builds its pictures from them, so it passes on any conforming implementation:

| | `ALIASED_POINT_SIZE_RANGE` | `ALIASED_LINE_WIDTH_RANGE` |
|---|---|---|
| radeonsi (reference machine) | [0.125, 2048] | [1, 2048] |
| llvmpipe | [1, 256] | [1, 255] |

Cell 0 only draws the sizes the range allows (after scaling by `k`), cells 1 and 9 use `min(max, 40k)`
and `min(max, 60k)`,
and cell 8 requests widths 3 and 2.6 only when the maximum is at least 3 (otherwise width 1). With a
maximum of 1 the demo still records 71 checks. The expected pixels are computed from the *effective*
width: rounded to an integer, 0 treated as 1, clamped to the maximum.

## Expected image

Cell 1 shows a smooth gradient: black top-left, red top-right, green bottom-left, yellow bottom-right
(`t` grows downwards). Cell 9 is one large orange square. The staircase shapes of cells 2-4 grow
from three separate bars to an open path to a closed one. The strip is a continuous ribbon, the fan
a three-quarter disc, and the house has a red roof, a brown door, a blue window, two chimney lines
and a yellow sun.

## Validation

`es20-primitive-topology --headless --validate` runs 73 checks. The reference is computed on the
CPU in `main.cpp` and shares nothing with the GPU path but the vertex positions:

1. **Topology.** `segments_of()` and `triangles_of()` cut the vertex list into primitives exactly as
   §2.7.1 describes (odd leftovers, strips, fans, loops, `first`, `count = 0`).
2. **Coverage.** Points cover the pixels whose centers lie in their square (ties within 0.05 px are
   ignored). Lines through pixel centers cover one pixel per column (row), exactly, and by rule 3 of
   §3.5.1 *no* neighbour in the same column; wide lines cover their `w` rows. Triangles cover the
   pixels whose centers lie inside by at least 1 px and nothing farther than 1 px outside.
3. **Don't care.** What the specification leaves to the implementation is not compared: pixels
   within 1 px of a triangle edge, the ends of a line segment, and everything within `w/2 + 2.5` px of
   a line that is not horizontal, vertical or 45 degrees. Everything else, lit and unlit, must match.
   Each cell is one check ("N pixels compared", most are 3 500-4 000 of 4 096 at `k = 1`; at `k = 3`
   most are 35 000-36 800 of 36 864).
4. **Named probes.** For the line and triangle cells the checks name the facts: the midpoint of
   every segment this topology builds is lit; the midpoints of the segments that *only another*
   topology would add (the strip's connectors in `LINES`, the closing segment in `LINE_STRIP`) are
   clear; the centroid of every triangle is covered, and the centroids of triangles that `TRIANGLES`
   would not build from the same vertices are not; points on edges shared by strip or fan triangles
   (3 px from any other edge) are covered, so there are no holes; the wedge of the fan is open.
5. **Line rules (cell 10).** For three segments of general slope the demo checks rule 3 (at most one
   fragment per column or row), rule 2 (at most one empty interior column) and rule 1 (every fragment
   within 2 pixels of the ideal row, which is the bound that the one-pixel deviation of rule 1 implies
   when the ideal row changes by one between columns).
6. **State and errors.** The point and line ranges contain 1 and are ordered, the initial line width
   is 1, `glLineWidth(0)` and `glLineWidth(-2)` raise `INVALID_VALUE` and leave the width unchanged,
   a width of 3 is returned by `GL_LINE_WIDTH` when allowed, and `glDrawArrays` with a negative count
   raises `INVALID_VALUE` while count 0 is legal.

Not tested, on purpose: the exact end pixels of a segment, even line widths (the perturbation of the
diamond-exit rule decides there), points larger than the range (clamping), triangles covered
*twice* on a shared edge (this needs blending to observe; the demo checks that there are no holes),
and multisampled rasterization (the default framebuffer is single-sampled).

## Pitfalls

- **`GL_LINE_LOOP` is not a polygon.** It is a line strip plus a closing segment; fill is a different primitive.
- **`GL_LINES` with an odd count ignores the last vertex**, and `GL_TRIANGLES` ignores a remainder of one or two vertices. Neither is an error.
- **A triangle strip or fan with fewer than three vertices, or a line strip with one, draws nothing**, silently.
- **`gl_PointSize` must be written.** Left unwritten, the point size is undefined. Only size 1 is guaranteed; query `ALIASED_POINT_SIZE_RANGE` before relying on more, and expect the value to be clamped.
- **`gl_PointCoord` has its origin at the top-left** (`t` grows downwards), unlike window coordinates, texture coordinates and framebuffer rows.
- **Wide lines are optional.** Only width 1 is guaranteed. Widths are rounded to integers, so `glLineWidth(2.6)` draws 3 pixels, and `glLineWidth(0.4)` draws 1.
- **Exact pixels of lines are implementation-dependent** within the four rules of §3.5.1. Do not compare wide or sloped lines pixel for pixel across drivers.
- **Face culling is off by default**, which is why all triangles here are drawn whatever their winding. A strip swaps the roles of two vertices on every other triangle so that all keep the winding of the first; with culling on, that is what makes a strip come out consistently.
- **`GL_POINTS` size is in window pixels and is not scaled by the viewport.** A point whose center is
  inside the viewport can extend beyond it, and the viewport does not clip it; only the scissor test does,
  which is why every cell here has its own scissor box.
- **Dithering** (`GL_DITHER`, on by default) can change the lowest bit of a color; the demo turns it off so flat colors are written exactly.
