# es30-hello-triangle

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es30-hello-triangle [--headless --validate]`

## Purpose

The smallest complete OpenGL ES 3.0 program. It walks the whole programmable pipeline once:
vertex data is uploaded into a buffer object, a vertex array object describes how the vertex
shader reads it, a two-stage program transforms and colors the vertices, and the rasterizer
interpolates the colors across the triangle. Everything else in this suite builds on these steps.

## What happens, call by call

| Step | Calls | Notes |
|---|---|---|
| Compile | `glCreateShader`, `glShaderSource`, `glCompileShader`, `glGetShaderiv(COMPILE_STATUS)` | A failed compile is not a GL error. It only shows up in `COMPILE_STATUS` and the info log, so always check them. |
| Link | `glCreateProgram`, `glAttachShader`, `glLinkProgram`, `glGetProgramiv(LINK_STATUS)` | After linking, shaders can be detached and deleted. Deleting an attached shader is deferred until it is detached (ES 3.0 §2.12.1). |
| Upload | `glGenBuffers`, `glBindBuffer(ARRAY_BUFFER)`, `glBufferData(STATIC_DRAW)` | The usage hint is a performance hint, not a restriction. |
| Describe | `glGenVertexArrays`, `glBindVertexArray`, `glEnableVertexAttribArray`, `glVertexAttribPointer` | The VAO records each attribute's format, enable flag, and the buffer that was bound to `ARRAY_BUFFER` *when `glVertexAttribPointer` ran*. The `ARRAY_BUFFER` binding itself is not VAO state. |
| Draw | `glViewport`, `glClearColor`, `glClear`, `glUseProgram`, `glBindVertexArray`, `glDrawArrays(TRIANGLES, 0, 3)` | |

## Shader concepts

- `#version 300 es` must be the very first line.
- Vertex inputs are declared with `in` and `layout(location = N)`. The location is the generic
  attribute index passed to `glVertexAttribPointer`, so no `glGetAttribLocation` lookup is needed.
- The fragment shader declares `precision mediump float;`. Fragment shaders have **no default
  float precision**, so omitting it is a compile error.
- GLSL ES 3.00 has no `gl_FragColor`. A user-declared `out vec4` at location 0 feeds draw buffer 0.

## Expected result

A triangle on a dark-blue background with red (bottom-left), green (bottom-right) and blue (top)
corners, smoothly blended in between.

## Validation (`--validate`)

- `GL_CURRENT_PROGRAM` and `GL_VERTEX_ARRAY_BINDING` hold the objects the draw used, and the
  color attribute's stride read back with `glGetVertexAttribiv` equals `sizeof(Vertex)`.
- Two corner pixels equal the clear color (±1 for 8-bit conversion).
- Four interior pixels are compared with a CPU reference that evaluates the barycentric weights
  **at the pixel center**, where the rasterizer samples (ES 3.0 §3.6.1). The tolerance is ±3/255,
  which covers `mediump` arithmetic and the implementation's choice of rounding.
- The runtime additionally requires `glGetError() == GL_NO_ERROR` after setup, after rendering and
  after deleting every object.

## Spec references

ES 3.0 §2.9 Vertex Arrays · §2.10 Buffer Objects · §2.11 Vertex Array Objects ·
§2.12.1/§2.12.3 Shader and Program Objects · §3.6.1 Basic Polygon Rasterization ·
§4.2.3 Clearing the Buffers · §4.3.2 Reading Pixels.
GLSL ES 3.00 §4.3.4 Input Variables · §4.3.6 Output Variables · §4.3.8 Layout Qualifiers ·
§4.5.4 Default Precision Qualifiers.

## Pitfalls

- Passing a byte offset to `glVertexAttribPointer` only works while a buffer is bound to
  `ARRAY_BUFFER`. The pointer argument then means "offset into that buffer" (`demo::buffer_offset`).
- Without a bound VAO, a non-zero VAO cannot be described. With VAO 0, ES 3.0 still allows the
  legacy default vertex array.
- `glClearColor` values are converted to the buffer's fixed-point format. 0.10 × 255 = 25.5 may
  become 25 or 26, which is why the clear check allows ±1.
