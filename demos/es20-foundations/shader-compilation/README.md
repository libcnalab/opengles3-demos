# es20-shader-compilation

**API introduced in:** OpenGL ES 2.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

Shader and program objects have a life cycle with several non-obvious rules. Errors in this
area are often invisible: a failed compile does not raise a GL error, and deleting a shader does
not necessarily delete it. This demo exercises each rule and verifies the result.

The image is deliberately simple. The left half is drawn by *program A*, whose vertex shader was
assembled from three source strings. The right half is drawn by *program B*, loaded the usual way.
Both draw a full-screen triangle whose corners come from `gl_VertexID`, with no vertex attributes.

## What is demonstrated and checked

| Topic | Calls | Rule (spec) |
|---|---|---|
| Multi-string source | `glShaderSource(shader, 3, strings, {-1, -1, N})` | Strings are concatenated. A negative length means null-terminated; otherwise exactly N characters are used. `SHADER_SOURCE_LENGTH` includes the terminator (ES 3.0 §2.12.1, §6.1.12). |
| Source round trip | `glGetShaderSource` | Returns the concatenation; text past an explicit length is absent. |
| Compile failure | `glCompileShader`, `glGetShaderiv(COMPILE_STATUS)`, `glGetShaderInfoLog` | Reported by status and log only, never by `glGetError`. Cases: a syntax error, and a fragment shader without a float precision (GLSL ES 3.00 §4.5.4). |
| Link failure | `glLinkProgram`, `glGetProgramiv(LINK_STATUS)` | A `vec3` output linked to a `vec4` input fails (GLSL ES 3.00 §4.3.10). An ES 3.0 program without a fragment shader fails (ES 3.0 §2.12.3). |
| Using a failed program | `glUseProgram` | `INVALID_OPERATION`. |
| Attached shaders | `glGetAttachedShaders`, `glDetachShader` | A linked program keeps working after its shaders are detached. |
| Deferred shader deletion | `glDeleteShader`, `glIsShader`, `DELETE_STATUS` | An attached shader is only flagged; it disappears when the last program detaches it. `glDeleteShader(0)` is ignored. |
| Deferred program deletion | `glDeleteProgram`, `glIsProgram` | The current program is only flagged. It still renders, and is deleted once another program becomes current. |
| Validation | `glValidateProgram`, `VALIDATE_STATUS` | A `sampler2D` and a `samplerCube` on the same unit (both default to 0) make drawing fail with `INVALID_OPERATION`, and `glValidateProgram` reports `FALSE`. Moving one sampler with `glUniform1i` fixes it (ES 3.0 §2.12.9.7). |
| Precision | `glGetShaderPrecisionFormat` | Every precision type in both stages meets the GLSL ES 3.00 §4.5.1 minimum (e.g. `HIGH_FLOAT` range ≥ (126, 127), precision ≥ 23; `MEDIUM_FLOAT` ≥ (14, 14), ≥ 10). |
| Compiler presence | `glGetBooleanv(SHADER_COMPILER)`, `glReleaseShaderCompiler` | ES 3.0 always has a compiler. Releasing it is a hint and later compiles still work. |
| Shader binaries | `glShaderBinary`, `NUM_SHADER_BINARY_FORMATS` | ES defines no binary format, so the list may be empty (it is on Mesa). An unadvertised format gives `INVALID_ENUM` (ES 3.0 §2.12.2). |

## Expected result

Left half green, right half blue. The run prints the driver's compile and link logs. Their
wording is implementation-defined, so only their reported lengths are checked.

## Validation

48 checks: status values, queried state, error codes, the two halves' pixel colors (±1), and a
probe pixel drawn by the flagged-for-deletion program. Run:
`es20-shader-compilation --headless --validate`.

## Pitfalls

- Always check `COMPILE_STATUS` and `LINK_STATUS`. `glGetError` stays `GL_NO_ERROR` on failure.
- `glUniform*` writes to the **current** program. Set uniforms after `glUseProgram` (or use
  ES 3.1 `glProgramUniform*`).
- Sampler uniforms default to unit 0. Two different sampler types left at their defaults are a
  latent `INVALID_OPERATION`.
- A missing `precision mediump float;` in a fragment shader is a compile error, not a warning.
- Info-log text differs between drivers. Never parse it in application logic.

## Known driver deviations (XFAIL)

- **ANGLE** sets `VALIDATE_STATUS = TRUE` for a program whose `sampler2D` and `samplerCube` both use unit 0, although
  the following draw fails with `INVALID_OPERATION` as required. ES 3.0 §2.12.9.7 (ES 3.2 §11.1.3.11):
  *"ValidateProgram will check for all of the error conditions described in this section"*, and the sampler-type
  conflict is the first of them. XFAIL on "ANGLE"; the draw-time error is still checked strictly.
