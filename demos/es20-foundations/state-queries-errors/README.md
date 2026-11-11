# es20-state-queries-errors

**API introduced in:** OpenGL ES 2.0 (indexed queries: ES 3.0) · **Minimum context:** ES 3.0 · **Shaders:** none

## Purpose

Most GL bugs come down to "the state was not what I thought" or "the call failed and nobody
noticed". This demo covers the two tools for those questions: state queries and `glGetError`.
It checks what the specification says about each of them, not what one driver happens to do.

- **Initial state.** About 70 entries of the ES 3.0 state tables (§6.2, Tables 6.2-6.36) are
  read before the program changes anything. Each entry is read with all four simple queries
  (`glGetBooleanv`, `glGetIntegerv`, `glGetInteger64v`, `glGetFloatv`). Capabilities are also
  read with `glIsEnabled`.
- **Data conversions (§6.1.2).** When the query type differs from the state type, the result is
  converted. The values used here are chosen so that each conversion rule gives a different
  answer.
- **Indexed state (§6.1.1, §6.1.9).** The uniform-buffer and transform-feedback binding points,
  and `glGetStringi(EXTENSIONS)`.
- **Hints (§5.3)** and **dithering (§4.1.9)**.
- **The error model (§2.5).** Sticky flags, ignored commands, untouched output pointers, zero
  return values, and `INVALID_FRAMEBUFFER_OPERATION` (§4.4.4.4).

There are no shaders. The picture is a **dashboard**: one cell per state variable or error case.
A cell's fill is green when all of its GL query checks passed, red when one failed, and amber for
a documented driver deviation. The frame colour shows the group: blue for initial state, violet
for implementation minimums, cyan for conversions, orange for indexed state, grey for hints, olive
for dither, pink for error semantics, and khaki for incomplete framebuffers. Each cell is drawn
with `glScissor` + `glClear`, so its rectangle is exact at every surface size.

## What is demonstrated and checked

| Topic | Calls | Rule (spec) |
|---|---|---|
| Initial values | `glGet{Boolean,Integer,Integer64,Float}v`, `glIsEnabled` | Tables 6.2-6.36. Examples: `DITHER` TRUE and every other capability FALSE; `VIEWPORT`/`SCISSOR_BOX` = (0,0,w,h); `CULL_FACE_MODE` BACK; `DEPTH_FUNC` LESS; `BLEND_*` ONE/ZERO/FUNC_ADD; `COLOR_WRITEMASK` all TRUE; `DEPTH_CLEAR_VALUE` 1; `UNPACK_/PACK_ALIGNMENT` 4; `DRAW_BUFFER0` and `READ_BUFFER` BACK, `DRAW_BUFFER1` NONE; every binding 0. |
| Stencil masks | `STENCIL_(BACK_)VALUE_MASK`, `STENCIL_(BACK_)WRITEMASK` | §4.1.4 makes the initial mask 2^s − 1 with s ≥ the stencil depth, so the exact s is implementation-dependent. The check is that the value has the form 2^s − 1 and covers every stencil bit. Mesa returns 0xFF. |
| Implementation minimums | `MAX_TEXTURE_SIZE`, `MAX_UNIFORM_BUFFER_BINDINGS`, ... | Minimum columns of Tables 6.28-6.35 (e.g. 2048, 24, 16, 4). `UNIFORM_BUFFER_OFFSET_ALIGNMENT` is a *maximum* (≤ 256). The other three query types must agree with `GetIntegerv`. |
| boolean → int/float | `glColorMask(T,F,T,F)`, `glDepthMask(F)` | TRUE/FALSE read as 1/0 and 1.0/0.0. |
| float → int (rounding) | `glPolygonOffset(0.25, -2.75)`, `glSampleCoverage(0.75)` | Ordinary floating-point state is rounded to the nearest integer: 0 and −3, and 1 for `SAMPLE_COVERAGE_VALUE` (that value is *not* in the §6.1.2 list of linearly mapped state). `GetBooleanv` returns FALSE only for zero, so 0.25 reads as TRUE although it rounds to 0. |
| color/depth → int (linear) | `glClearColor`, `glBlendColor`, `glDepthRangef`, `glClearDepthf` | §6.1.2: *"unless the value is an RGBA color component, a DepthRangef value, or a depth buffer clear value. In these cases, the Get command converts the floating-point value to an integer according to the INT entry of table 4.5"*. That entry is Equation 2.4 with b = 32: one of the two integers nearest f·(2³¹−1). So 1.0 → 2147483647, 0.5 → 1073741823 or 1073741824. The same 0.25 reads as 0 from `POLYGON_OFFSET_FACTOR` and as about 2²⁹ from `COLOR_CLEAR_VALUE`. |
| Clamping on specification | `glDepthRangef(1.5,-2)`, `glClearDepthf(3)`, `glClearColor(-0.5,...)` | The depth range and the depth clear value are clamped to [0,1] when they are specified. The clear colour is *stored as floating-point* (§4.2.3) and reads back as −0.5. |
| 64 → 32-bit saturation | `MAX_ELEMENT_INDEX`, `MAX_SERVER_WAIT_TIMEOUT` | A value too large for the requested type returns the nearest representable value: `GetIntegerv` = clamp(`GetInteger64v`). |
| Indexed state | `glGetIntegeri_v`, `glGetInteger64i_v`, `glBindBufferRange/Base` | Every index starts at 0. A range binding reports its offset and size. `BindBufferBase` reports START = SIZE = 0 (the §6.1.9 footnote calls zero a sentinel). The generic binding changes too (§2.10.1.1). The 32-bit and 64-bit indexed queries are interchangeable (§6.2). |
| Index out of range | `glGetIntegeri_v(..., MAX)` | `INVALID_VALUE` (§6.1.1), and the output variable is not written (§2.5). |
| Extension strings | `glGetStringi(EXTENSIONS, i)` | Exactly the tokens of `glGetString(EXTENSIONS)` (§6.1.6). Index `NUM_EXTENSIONS` gives `INVALID_VALUE` and returns NULL. Name `RENDERER` gives `INVALID_ENUM` and returns NULL. |
| Hints | `glHint`, `GetIntegerv(*_HINT)` | Both targets start at DONT_CARE. FASTEST, NICEST and DONT_CARE read back. What a hint does is implementation-dependent, so nothing visual is asserted. |
| DITHER | `glEnable/glDisable(DITHER)` | Initially TRUE, and the toggles read back. A clear with a non-representable colour (0.5, 0.3, 0.7, 0.5) gives, at every pixel, one of the two representable neighbours. With dithering disabled the choice may not depend on the pixel position, so the image is uniform. |
| Sticky errors | `glLineWidth(-1)` then valid calls | The flag survives later valid commands, which still execute. Reading the error clears the flag. |
| First error kept | VALUE, OPERATION, ENUM in a row | `INVALID_VALUE` is reported. An implementation with several flag-code pairs may also report the others, in unspecified order (§2.5). The check accepts only those three codes. |
| Ignored commands | `glViewport/glScissor` (negative size), `glLineWidth(0)`, `glPixelStorei(3)`, `glBindTexture` (wrong target), `glBindVertexArray` (deleted name), `glBindBufferBase(ARRAY_BUFFER)`, `glBindBufferRange` (bad index/size/alignment), `glMapBufferRange` (past the end) | Each command raises the specified error, and the state it would have changed is re-read and found unchanged. |
| Zero return values | `glMapBufferRange`, `glUnmapBuffer`, `glGetStringi` | A command that fails returns zero: NULL or FALSE. |
| Incomplete framebuffers | `glCheckFramebufferStatus`, `glClear`, `glReadPixels` | An FBO without attachments is `INCOMPLETE_MISSING_ATTACHMENT`. Clearing it gives `INVALID_FRAMEBUFFER_OPERATION`, even with `RASTERIZER_DISCARD` enabled. An RGBA8 image on `DEPTH_ATTACHMENT` gives `INCOMPLETE_ATTACHMENT`. Then a red clear and a readback both fail: the client buffer is not written, and once the depth image is detached the colour image still holds its earlier blue. |

## Expected result

The window is covered by a grid of about 140 green cells. The grid adapts to the aspect ratio
and fills any surface of 64×64 or larger; smaller surfaces report `SKIP_UNSUPPORTED`. Unused
cells after the last one are dark. On radeonsi one pink-framed cell ("UnmapBuffer while
unmapped") is amber, because of the glthread deviation described below. The run also prints
implementation-dependent values: stencil mask bits, `UNIFORM_BUFFER_OFFSET_ALIGNMENT`, line and
point ranges, `MAX_TEXTURE_LOD_BIAS`, `MAX_ELEMENT_INDEX`, `MAX_SERVER_WAIT_TIMEOUT`,
`NUM_EXTENSIONS`, and whether the driver dithers. Mesa does not dither these clears; it picks
128 for 127.5.

## Validation

`es20-state-queries-errors --headless --validate` records 688 checks:

1. Every GL query check made in `init()` is replayed into the result. These are the actual
   tests.
2. The dashboard is read back. For each group, every cell's centre and inner corner must match
   its status colour, and its frame must match the group colour, within ±1. All colours are
   k/255, so Equation 2.3 gives k or a neighbour, and dithering can only choose between those
   neighbours.
3. Negative control: one cell is forced to "failed" and redrawn. It must turn red while its
   neighbour keeps its colour. It then returns to its real status. The conversion group also
   holds a built-in contrast: one input value must give different integers under the rounding
   rule and the linear rule.

`OUT_OF_MEMORY` is not tested. The spec gives no way to provoke it deterministically, and its
"command ignored?" entry in Table 2.3 is "Unknown".

## Pitfalls

- `glGetIntegerv(GL_COLOR_CLEAR_VALUE)` does **not** return 0..255. It returns colours scaled to
  2³¹−1. Read colours, depth range and the depth clear value with `glGetFloatv`.
- `glGetBooleanv` on float state is TRUE for any non-zero value, including values that round to
  0.
- A failed command leaves its output arguments alone. Initialise query outputs if you might read
  them after an error.
- `glGetError` reports only the *first* error since it was last called. Call it in a loop until
  `GL_NO_ERROR` to clear every flag. Do not read meaning into the order of the later codes.
- `INVALID_ENUM` for `glHint` cannot be produced through meta-gl. Its typed `HintTarget` and
  `HintMode` hold only legal values (META_GL_GAPS O1). This demo reaches `INVALID_ENUM` with
  values that are legal enums but invalid in context instead: `GetIntegeri_v(DEPTH_FUNC)`,
  `BindBufferBase(ARRAY_BUFFER)` and `GetStringi(RENDERER)`.
- Mesa's glthread (on by default on radeonsi) returns TRUE from `glUnmapBuffer` on an unmapped
  buffer, although the error is still raised. Run with `mesa_glthread=false` to see FALSE. The
  check is strict and reported as XFAIL on radeonsi (also in `es30-buffer-mapping`).
