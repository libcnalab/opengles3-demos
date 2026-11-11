# es31-program-pipelines

**Minimum:** OpenGL ES 3.1 · **Shaders:** GLSL ES 3.10 (`#version 310 es`) · **Run:** `es31-program-pipelines [--headless --validate]`

## Purpose

Until ES 3.0 a program object was a closed package: one vertex shader and one fragment shader linked together, and
changing either meant linking (or keeping) a whole new program for every combination. ES 3.1 adds **separable
programs** and **program pipeline objects**. A separable program may contain a single stage; a pipeline object then
says *which program supplies which stage*, and the stages of a bound pipeline can be exchanged with one call. This demo
shows how to build such programs, how to assemble and change pipelines at run time, how uniforms are addressed when
there is no single "current program", how the interface between two programs is matched and validated, and what the
API does when something is wrong.

## What the demo does

1. **Two ways to make a separable program.** `glCreateShaderProgramv(type, count, strings)` compiles one stage and
   returns a ready-to-use separable program in one call (it also sets `PROGRAM_SEPARABLE`, links, and detaches). The
   classic route is `glCreateProgram` + `glAttachShader` + **`glProgramParameteri(program, PROGRAM_SEPARABLE, TRUE)`
   before `glLinkProgram`**; the demo uses it for `progC`, a program that holds both a vertex and a fragment stage.
   A source that does not compile still makes `glCreateShaderProgramv` return a program name, but its `LINK_STATUS` is
   `FALSE` and its info log carries the shader's compile log, because the temporary shader object is already deleted.
2. **Pipelines and stages.** `glGenProgramPipelines`, `glBindProgramPipeline`, `glUseProgramStages(pipeline, stages,
   program)` with `VERTEX_SHADER_BIT`, `FRAGMENT_SHADER_BIT`, their union and `ALL_SHADER_BITS`. The pattern in a cell
   changes when only the vertex stage, or only the fragment stage, of the same bound pipeline is replaced.
3. **Uniforms without a current program.** `glUniform*` updates the *active program*: the program given to
   `glActiveShaderProgram(pipeline, program)` when a pipeline is bound, or the program of `glUseProgram` if there is
   one. `glProgramUniform*` names the program explicitly and needs no binding at all. The demo uses all 33
   `glProgramUniform*` entry points in a "uniform zoo" and reads every value back with `glGetUniform*`.
4. **Precedence.** A program made current with `glUseProgram` hides the bound pipeline, for drawing and for uniform
   updates; `glUseProgram(0)` brings the pipeline back.
5. **Interface matching and validation.** Between two programs nothing is checked at link time. The outputs of the vertex
   program must match the inputs of the fragment program by **location** (names are irrelevant when locations are
   declared), type and precision, and a program that was linked from several stages must be active for all of them or none.
   `glValidateProgramPipeline` sets `VALIDATE_STATUS`; a draw with an invalid pipeline raises `INVALID_OPERATION`.
6. **State and errors.** `glGetProgramPipelineiv` (`ACTIVE_PROGRAM`, `VERTEX_SHADER`, `FRAGMENT_SHADER`,
   `COMPUTE_SHADER`, `VALIDATE_STATUS`, `INFO_LOG_LENGTH`), `glGetProgramPipelineInfoLog`, deletion rules, and every
   `INVALID_OPERATION` / `INVALID_VALUE` the demo can reach without an invalid enum.

## Reading the image

The 256 x 256 surface has two parts (rows are numbered from the **bottom**).

**Rows 0-2 (bottom 96 pixels): the scene.** 8 columns of 32-pixel cells, cell number = row * 8 + column. Every cell is one
triangle covering the cell, drawn with a pipeline. The vertex stage supplies a *tint* (`u_tint`, a uniform of the vertex
program) and the coordinate `v_uv`; the fragment stage supplies a *background* (`u_bg`) and decides the split: the vertical
fragment program shows the tint on the left half, the horizontal one on the bottom half. The "mirror" vertex program flips
`u`, so the tint jumps to the right half.

| Cell | What it shows |
|---|---|
| **A0** | pipeline P1 = vertex A + fragment X: red left, olive right |
| **A1** | only the vertex stage replaced by B: blue, mirrored to the right |
| **A2** | only the fragment stage replaced by Y: pattern turns horizontal (blue bottom, teal top) |
| **A3** | vertex stage back to A: red bottom |
| **A4** | both stages from **one** two-stage program (`VERTEX\|FRAGMENT` bits): green left, magenta right |
| **A5** | `glProgramUniform3f` on that program while it is installed: yellow left (the picture follows at once) |
| **A6** | both stages replaced by single-stage programs again (blue bottom, teal top) |
| **A7** | `ALL_SHADER_BITS` with the two-stage program: same picture as A5 |
| **B0** | `glProgramUniform3f` on programs that are installed but neither current nor active: orange / dark blue |
| **B1** | `glActiveShaderProgram(vertex A)` + `glUniform3f(0, ...)`: violet tint |
| **B2** | the identical call with vertex B active changes B's tint instead: teal on the right |
| **B3** | `glActiveShaderProgram(fragment X)` + `glUniform3f(1, ...)`: the background becomes yellow |
| **B4** | `glUseProgram(progU)`: a solid orange fill although pipeline P3 is bound |
| **B5** | `glUseProgram(0)`: pipeline P3 again |
| **B6** | a second pipeline P2 (vertex A + fragment Y): the stages of P3 are untouched |
| **B7** | a vertex program whose outputs have **different names** but the same locations: orange left |
| **C0** | the horizontal fragment program built by `glCreateShaderProgramv` from **two** source strings |
| **C1-C3** | invalid pipelines (location, type, precision mismatch): the draw is rejected, the cell stays **empty** |
| **C4** | a vertex program that was `glDeleteProgram`-ed while attached to the pipeline: still used (violet) |
| **C5, C6** | pipelines that use only one stage of a two-stage program: rejected, cells stay **empty** |
| **C7** | scratch cell for the draws of valid pipelines in `validate()` (cleared again) |

The empty cells are the expected outcome of the draws that are *supposed* to fail; a coloured cell there would be a bug.

**Rows 3-7: the uniform zoo.** 38 cells of 32 pixels in reading order from the lower-left, the rest dark. Each cell checks
one uniform of the separable fragment program against the exact value that a `glProgramUniform*` call stored while the
program was never current: green = equal. All 38 are green.

## Central calls

| Call | Where |
|---|---|
| `glCreateShaderProgramv`, `glProgramParameteri(PROGRAM_SEPARABLE)`, `glLinkProgram` | `init()` |
| `glGenProgramPipelines`, `glBindProgramPipeline`, `glUseProgramStages`, `glActiveShaderProgram` | `render()` |
| `glProgramUniform*`, `glUniform3f`, `glUseProgram` | `render()`, `upload_zoo()` |
| `glValidateProgramPipeline`, `glGetProgramPipelineiv`, `glGetProgramPipelineInfoLog` | `check_validation()`, `check_pipeline_objects()` |

## Spec sections

OpenGL ES 3.2 (the numbering for ES 3.1 features): §7.3 Program Objects, §7.4 Program Pipeline Objects, §7.4.1 Shader
Interface Matching, §7.4.2 Program Pipeline Object State, §7.6.1 Loading Uniform Variables In The Default Uniform Block,
§7.12 Shader, Program, and Program Pipeline Queries, §11.1.3.11 Validation. GLSL ES 3.10: §4.4.1 and §4.4.2 (location
qualifiers on `in`/`out`), §4.4.3 (uniform locations), §9 Shader Interface Matching, in particular §9.2.1 footnote 2 and the
§9.2.2 table for separable programs (location, interpolation and precision must match).

## Shader concepts

* Every separable stage declares its interface with `layout(location = N)`: `out vec2 v_uv` at 0 and `out vec3 v_tint` at 1,
  and the same locations as inputs of the fragment programs. This is what lets *any* vertex program be combined with
  *any* fragment program; matching by name alone would tie the shaders to each other.
* Each program has its **own uniform name space**. The vertex programs declare `layout(location = 0) uniform vec3 u_tint`,
  the fragment programs `layout(location = 1) uniform vec3 u_bg`. Location 0 therefore means a different variable in every
  vertex program, which is exactly why `glUniform3f(0, ...)` needs to know the *active* program. Inside the two-stage
  program the stages share one name space, so the two uniforms must not collide: that is why `u_bg` is at 1.
* ES 3.1 vertex shaders need nothing special to be separable: `gl_Position` is part of the fixed interface. (The
  `gl_PerVertex` block and its redeclaration appear in GLSL ES **3.20**, not 3.10; see Pitfalls.)
* Mismatch shaders: `vs_pair_location.vert` (outputs at locations 3 and 4, names unchanged), `vs_pair_type.vert` (a `vec3`
  where the fragment program reads a `vec2`), `fs_mediump.frag` (`mediump` inputs against `highp` outputs).

## Validation

`validate()` reads pixels and state and compares them with CPU expectations:

* **Pixels.** Every drawn scene cell is probed at its four quadrant centres (away from the 0.5 split lines and the cell
  edges) against a model that takes the installed programs and their uniform values: `uv` is mirrored by vertex B, the
  split is on `u` or `v`, the tint or the background is chosen. Tolerance: ±1 per channel, the 8-bit quantization of
  the float colour (the clear colour 0.10 is 25.5 in 8-bit and rounds either way). Zoo cells are compared exactly
  (they are 0 or 1).
* **Readback.** All 43 stored zoo values (`glGetUniform{f,i,ui}v`), the three tint/background uniforms after each
  `glActiveShaderProgram`/`glUniform3f`/`glProgramUniform3f` combination, so that "which program was changed" is
  proved by values and not by a picture.
* **State.** `IsProgramPipeline` before and after the first bind, initial pipeline state (all names 0, `VALIDATE_STATUS`
  FALSE, log length 0), `GetProgramPipelineiv` after every `glUseProgramStages` variant, `PROGRAM_PIPELINE_BINDING`,
  deletion rules (a program held by a pipeline is flagged, not deleted).
* **Validation.** `VALIDATE_STATUS` and the draw outcome for three valid and five invalid pipelines; replacing the
  offending program makes the same pipeline valid (a mutation control).
* **Errors.** Non-separable and unlinked programs, shader and unknown names, never generated and deleted pipelines,
  `glUniform*` with no active program or an unknown location, `glCreateShaderProgramv` with a negative count.
* **Negative controls.** Changing one `u_tint` with `glProgramUniform3f` must change cell 0 and break the old expectation;
  a wrong `ivec3` must turn zoo cell 10 red while cell 11 stays green; the restored values restore the cells. The
  comparisons can therefore fail.

## Pitfalls

* `glUseProgramStages` requires a program that was **linked** with `PROGRAM_SEPARABLE` set. The flag only matters at link
  time (ES 3.2 §7.3: "does not take effect until after the next time LinkProgram ... is called"): setting it afterwards
  and not relinking leaves the program non-separable, and `glUseProgramStages` must fail with `INVALID_OPERATION`.
  **Mesa honours the flag immediately** and accepts the call (recorded as XFAIL, see `docs/KNOWN_LIMITATIONS.md`).
* A program that was linked from several stages must be active for **all** of them or none (§11.1.3.11). Taking only the
  vertex stage of a two-stage program and the fragment stage of another program is an invalid pipeline; so is replacing
  just one of its stages. Replace both (cells A4 to A6).
* Nothing is checked at link time between separable programs. Without `glValidateProgramPipeline` the first sign of a
  mismatch is `INVALID_OPERATION` at the first draw. The info log of a failed validation is not standardised: Mesa gives
  a message for the multi-stage case and none for location, type or precision mismatches, so only its *length
  consistency* is asserted and the text is printed.
* `glUniform*` with a bound pipeline but **no active program** is `INVALID_OPERATION`; `glGenProgramPipelines` only
  reserves a name, the object exists after the first bind, `glUseProgramStages` or `glActiveShaderProgram`.
* `glUseProgram` beats the pipeline silently. A forgotten `glUseProgram` of a previous frame makes pipeline code appear to
  do nothing.
* ES 3.1 says outputs match inputs "in name, type, and qualification, or by location"; ES 3.2 §7.4.1 and GLSL ES 3.10
  §9.2.1 (footnote 2) state that a present location qualifier alone decides. The demo follows the latter (a pair with
  equal names but different locations is invalid) and cites ES 3.2 numbering.
* The `gl_PerVertex` redeclaration rules mentioned with separable programs belong to GLSL ES 3.20 (and the
  separate-shader-objects extension for ES 3.0/3.1). `#version 310 es` has no such block, so this demo does not use it.
