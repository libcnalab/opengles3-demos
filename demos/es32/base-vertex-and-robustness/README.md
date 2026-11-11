# es32-base-vertex-and-robustness

**Minimum:** OpenGL ES 3.2 · **Shaders:** GLSL ES 3.20 · **Context:** robust access + `LOSE_CONTEXT_ON_RESET` ·
**Run:** `es32-base-vertex-and-robustness [--headless --validate]`

## Purpose

This demo covers two OpenGL ES 3.2 additions. Both are about where the GL reads from and where it writes to.

**Base vertex drawing.** Real programs pack many meshes into one vertex buffer and one index buffer. Before
ES 3.2, the indices of every mesh after the first had to be rewritten with the mesh's offset added, or a
`glVertexAttribPointer` call had to rebind the attributes before each draw.
`glDrawElementsBaseVertex`, `glDrawRangeElementsBaseVertex` and `glDrawElementsInstancedBaseVertex` add
`basevertex` to every index instead. In this demo, ten quads and two strip quads live in one 260-vertex buffer.
The one element array buffer stores a single quad pattern `{0,1,2, 0,2,3}` in all three index types, plus
one triangle strip. Each draw selects its mesh only through `basevertex`.

**Robustness.** The context is created with robust buffer access and the `LOSE_CONTEXT_ON_RESET` strategy.
The demo queries `CONTEXT_FLAGS`, the graphics reset status and the reset notification strategy. It
exercises the bounded queries `glReadnPixels` and `glGetnUniform{f,i,ui}v` with canary-filled buffers, and
issues two draws whose `basevertex` points far past the end of the vertex buffer.

## What happens, call by call

| Step | Calls | Notes |
|---|---|---|
| Context | `demo::Info::robust_context = true` | The headless runtime asks EGL 1.5 for `EGL_CONTEXT_OPENGL_ROBUST_ACCESS = TRUE` and `EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY = EGL_LOSE_CONTEXT_ON_RESET`. EGL 1.5 §3.7.1.5 says: "If the implementation does not support robust buffer access, context creation will fail." So a context that exists has robust access. |
| Vertex data | `glBufferData(ARRAY_BUFFER)`, `glVertexAttribPointer`, `glVertexAttribIPointer`, `glVertexAttribDivisor` | Per-vertex: position, colour, and `a_index`, the vertex's own position in the buffer (integer attribute). Per instance (divisor 1): an offset and a tint from a second buffer with 4 entries. |
| Index data | `glBufferData(ELEMENT_ARRAY_BUFFER)` | 56 bytes: the quad as ubyte (offset 0), ushort (8) and uint (20), and the ubyte strip `247 248 249 250 255 251 252 253 254` (44). The binding is VAO state. |
| Base vertex | `glDrawElementsBaseVertex(…, basevertex 4 / 8 / 12)` | Meshes 1 to 3, one index type each. |
| Range | `glDrawRangeElementsBaseVertex(start 0, end 3, …, basevertex 16)` | `start` and `end` bound the indices **before** `basevertex` is added (§10.5). |
| Instanced | `glDrawElementsInstancedBaseVertex(…, instancecount 4, basevertex 20)` | Per-vertex attributes are fetched at index + 20. The instanced attribute is fetched at `instance` (+ baseinstance, always 0 in ES). `basevertex` does not apply to it. |
| Vertex ID | the same three commands with `u_mode = 1` | The shader shows green when `gl_VertexID == a_index`. |
| Restart | `glEnable(PRIMITIVE_RESTART_FIXED_INDEX)`, `glDrawElementsBaseVertex(TRIANGLE_STRIP, …, basevertex 1)` | Index 255 restarts the strip. The index 254 becomes vertex 255 after the shift and must **not** restart it. |
| Robust access | `glScissor`, two `glDraw*BaseVertex` with `basevertex = 260` and `2^20` | Both are drawn only when `CONTEXT_FLAG_ROBUST_ACCESS_BIT` is set. The scissor test keeps their fragments in cell (0,2), and mesh 9 is drawn over that cell afterwards. |
| Queries | `glGetIntegerv(CONTEXT_FLAGS, RESET_NOTIFICATION_STRATEGY)`, `glGetGraphicsResetStatus`, `glReadnPixels`, `glGetnUniformfv/iv/uiv` | See validation. |

## Shader concepts

- `scene.vert` reads `a_index` as `in highp int` through `glVertexAttribIPointer`. With
  `glVertexAttribPointer`, the integer would be converted to float.
- `gl_VertexID` is "the integer index i implicitly passed by DrawArrays or one of the other drawing commands
  defined in section 10.5" (ES 3.2 §11.1.3.9). For the `*BaseVertex` commands, §10.5 defines it as the
  index stored in the element array **plus** `basevertex`. The ES 3.2 change log (appendix E.2) states that
  the vertex ID includes the basevertex value. It is defined because every enabled array has a buffer bound.
- `probe.frag` declares a `vec4`, an `ivec3`, a `uvec2` and a `mat2`. All four feed the output, so they stay
  active and have locations. The program is linked only so that the uniforms can be queried; it is never drawn.

## Expected result

The capture shows an 8 x 8 grid of 32-pixel cells on a dark blue-grey background, with the top-left cell
empty:

- row 2 from the top: blue, orange, light green and yellow squares (meshes 1 to 4)
- row 3: four cyan squares that get darker from left to right (one mesh, four instances, tints 1, ¾, ½, ¼)
- row 4: four green squares (every `gl_VertexID` matched; red would mean a mismatch)
- row 5: a light-blue and a pink square (the two strip quads on either side of the restart)
- row 6: a purple square (mesh 9 over the cell that received the out-of-bounds draws)

The empty top-left cell is where mesh 0 would appear. Every draw would land there if `basevertex` were ignored.

## Validation (`--validate`)

All 64 cells are compared with a **CPU model** that expands the same draw records the GPU received. The
model reads the index bytes and splits strips at the restart index, comparing it with the raw index before
`basevertex` is added (§10.3.4). It adds `basevertex` to get the vertex ID (§10.5), fetches instanced
attributes by instance number, and evaluates `gl_VertexID == a_index` itself. It then rasterizes the
triangles at pixel centres. Each cell is checked at a 3 x 3 grid of probes. Probes closer than 1.5 px to any
triangle edge are skipped, because edge pixels are implementation-dependent. The tolerance is ±1 per
channel: all colours are multiples of ¼, so the only error left is the 8-bit conversion.

Negative controls and mutations:

- Cell (0,7) must stay clear. On the GPU, `glDrawElements` and `glDrawElementsBaseVertex(basevertex 0)`
  with the same arguments do draw mesh 0 there, and `basevertex 8` moves the same call to mesh 2's cell. So
  the "stays clear" check would catch an ignored `basevertex`.
- A mutated CPU model that ignores `basevertex` disagrees with the image in all 15 drawn cells.
- A mutated model that compares the restart index after adding `basevertex` disagrees with cell (1,3).

Errors (each erroneous call would draw mesh 0 if it were executed, so the clear cell (0,7) also proves that
the commands were ignored, §2.3.1):

- `glDrawRangeElementsBaseVertex` with `end < start` gives `INVALID_VALUE` (§10.5).
- a count of -1 (`glDrawElementsBaseVertex`, `glDrawRangeElementsBaseVertex`) or an instancecount of -1
  gives `INVALID_VALUE` (the implicit sizei rule of §2.3.1).

Robustness:

- `CONTEXT_FLAGS` has `CONTEXT_FLAG_ROBUST_ACCESS_BIT` (§20.2). The debug bit is printed but not asserted:
  nothing in the spec forbids an implementation from returning a debug context unasked.
- `glGetGraphicsResetStatus()` returns `NO_ERROR` (§2.3.2), also after the out-of-bounds draws.
- `RESET_NOTIFICATION_STRATEGY` is one of `NO_RESET_NOTIFICATION` and `LOSE_CONTEXT_ON_RESET`, and equals the
  requested `LOSE_CONTEXT_ON_RESET` (EGL 1.5 §3.7.1.6).
- Robust buffer access, ES 3.2 §10.3.5, verbatim: *"When enabled, indices within the element array (see
  section 10.3.8) that reference vertex data that lies outside the enabled attribute's vertex buffer object
  result in undefined values for the corresponding attributes, but cannot result in application failure."*
  So the demo checks only that the two out-of-bounds draws return, raise no GL error and cause no reset,
  and that rendering afterwards is correct (mesh 9 over the same cell, and every other cell). It does
  **not** assert which values the out-of-bounds fetches return. The guarantee "zero or values from within
  the buffer" comes from the `KHR_robust_buffer_access_behavior` extension. Appendix F.1 of the ES 3.2 spec
  explicitly excludes it from core: robust buffer access is included "(preventing crashes), but without the
  additional behavior defined by GL_KHR_robust_buffer_access".
- `glReadnPixels` (§16.1.2) of a 4 x 4 RGBA/UNSIGNED_BYTE block with `PACK_ALIGNMENT` 4 needs exactly 64
  bytes. The buffer is 80 bytes, pre-filled with `0xA5`:
  - With `bufSize = 64`, the call succeeds. The bytes equal `glReadPixels` and show mesh 1's colour, and
    the 16 bytes behind `bufSize` are untouched.
  - With `bufSize = 63` or `0`, the call gives `INVALID_OPERATION` and the whole buffer is untouched (§2.3.1:
    "If the generating command modifies values through a pointer argument, no change is made to these values").
  - With `bufSize = -1`, the call should give `INVALID_VALUE`, or the `INVALID_OPERATION` that also applies
    (§2.3.1: "the GL implementation may choose to generate any of the applicable errors"), and leave the
    buffer untouched.
  - A 0 x 4 rectangle with `bufSize = 0` succeeds and writes nothing.
- `glGetnUniform{f,i,ui}v` (§7.12) on `vec4`, `ivec3`, `uvec2` and `mat2` (column-major): an exact
  `bufSize` succeeds with the uploaded values and nothing after them. A larger `bufSize` still returns
  exactly as many values as the type has. `bufSize` one byte short gives `INVALID_OPERATION` with `params`
  untouched. So does a `bufSize` equal to the component count, because `bufSize` counts bytes (gl.xml declares
  `params` with `len="bufSize / 4"`). A negative `bufSize` gives `INVALID_VALUE` or `INVALID_OPERATION` with
  `params` untouched. Location -1 gives `INVALID_OPERATION`.

## Known deviation

On all Mesa drivers here (25.0.7: radeonsi, llvmpipe, zink), `glReadnPixels` with `bufSize = -1` raises
**no** error and writes the 64 bytes. It behaves as if a negative size meant "unbounded", which contradicts
"An INVALID_VALUE error is generated if width, height, or bufSize is negative" (ES 3.2 §16.1.2). No
extension changes this rule. The two checks stay strict and are reported as XFAIL on drivers whose
renderer/version string contains "Mesa". `glGetnUniform*v` with a negative `bufSize` is handled correctly by
the same drivers.

## Spec references

ES 3.2 §10.5 Drawing Commands Using Vertex Arrays (DrawElementsOneInstance, vertex ID, *BaseVertex, the
DrawRangeElements errors) · §10.3.2 Vertex Attribute Divisors · §10.3.4 Primitive Restart (comparison before
basevertex) · §10.3.5 Robust Buffer Access · §11.1.3.9 Shader Inputs (gl_VertexID) · §2.3.1 Errors (implicit
sizei rule, ignored commands, untouched pointers, choice among applicable errors) · §2.3.2 Graphics Reset
Recovery · §7.12 Shader, Program, and Program Pipeline Queries (GetnUniform*) · §13.8.2 Scissor Test ·
§16.1.2 ReadPixels and ReadnPixels · §20.2 Context Queries (CONTEXT_FLAGS) · appendix E.2 change log
(vertex ID includes basevertex) · appendix F.1 (robust buffer access in core excludes
KHR_robust_buffer_access_behavior). EGL 1.5 §3.7.1.5 and §3.7.1.6. GLSL ES 3.20 §7.1 Built-In Language
Variables.

## Pitfalls

- `start` and `end` of `glDrawRangeElementsBaseVertex` describe the indices as stored, not the vertices that
  are actually fetched.
- `basevertex` does not move instanced attributes. To offset per-instance data you need a different buffer
  binding, because `baseinstance` does not exist in OpenGL ES.
- The restart index is compared with the raw index. With UNSIGNED_BYTE indices you cannot reach vertex
  `255 + k` via index 255. You can reach it via index 254 with `basevertex = k + 1`.
- A negative vertex ID (negative `basevertex` with small indices) is undefined (§10.5). The demo avoids it.
- Robust access does not make out-of-bounds data meaningful. It only promises that the program survives.
  Do not rely on zeros without `KHR_robust_buffer_access_behavior`.
- `bufSize` of `glGetnUniform*v` is in bytes, not in components.
- The mesa softpipe driver cannot create a robust context at all (`EGL_BAD_MATCH` for ES 3.0, 3.1 and 3.2
  with `EGL_CONTEXT_OPENGL_ROBUST_ACCESS`). On softpipe this demo therefore reports `BLOCKED_ENVIRONMENT`
  rather than `SKIP_UNSUPPORTED`.
