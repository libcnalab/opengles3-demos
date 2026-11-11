# Known limitations

## Environment

- Reference machine: AMD Radeon 780M (radeonsi) and llvmpipe, Mesa 25.0.7, Linux 6.12, EGL 1.5.
  No non-Mesa OpenGL ES driver (ANGLE, Mali, Adreno, PowerVR, NVIDIA ES) has been tested.
  Results therefore say little about vendor-specific behavior.
- Minimum-version runs use `MESA_GLES_VERSION_OVERRIDE`, which lowers only the reported API version.
  No native ES 3.0-only driver is available, so "3.10 shaders are rejected by ES 3.0" is not
  verified anywhere. softpipe (native ES 3.1) verifies the 3.1-vs-3.2 boundary. On non-Mesa drivers
  set `-DOGLES3_TEST_MIN_VERSION=OFF` and `OGLES3_TEST_DRIVERS=gpu`.
- Windowed mode (SDL3) is tested under Xvfb; headless EGL pbuffers are the primary test path.
- Program binaries: `NUM_PROGRAM_BINARY_FORMATS` may be 0 on some drivers; such checks report
  the implementation's choice rather than failing.

## Driver deviations observed (Mesa 25.0.7, all Gallium drivers)

| Case | Rule | Observed |
|---|---|---|
| `shader-tests/glsl300/version-not-first-line.frag` | GLSL ES 3.00/3.10/3.20 §3.4: `#version` must be on the first line | a comment line before `#version` is accepted |
| `shader-tests/glsl300/integer-output-not-flat.vert` | GLSL ES 3.00 §4.3.6 (3.10 §4.3.6): integer vertex outputs must be `flat` | accepted; only integer *fragment inputs* are enforced |

| `diagnostics-capabilities` limits | ES 3.1 Table 20.x: `MAX_SAMPLES` ≥ 4, `MAX_FRAMEBUFFER_SAMPLES` ≥ 4 | softpipe reports 1 for both (no MSAA) while advertising ES 3.1 |

| `es20-uniforms` (38-way branch over 70+ uniforms) | correct execution | softpipe returns wrong comparison results for every branch after the third; same shader is correct on radeonsi, llvmpipe and zink, and each construct works in isolation on softpipe → softpipe backend miscompilation. Reproduce: `GALLIUM_DRIVER=softpipe LIBGL_ALWAYS_SOFTWARE=1 es20-uniforms --headless --validate` |

| `es20-vertex-attributes`, `es30-vertex-array-objects` (printed note, not asserted) | ES 3.0 §2.5: a command that generates an error has no effect on GL state | radeonsi/llvmpipe: a client-memory `glVertexAttribPointer` on a non-zero VAO correctly raises INVALID_OPERATION but still stores the pointer (`glGetVertexAttribPointerv` returns it) |
| `es20-vertex-attributes` (printed note, not asserted) | ES 3.0 §2.5: output parameters are not modified on error | `glGetVertexAttribiv(MAX_VERTEX_ATTRIBS, ...)` raises INVALID_VALUE but writes 0 to the output |

| `es20-depth-test` cell 13 | ES 3.0 §3.9.2.3 / ES 3.2 §14.2.3: fragment depth is clamped to [0,1]; "the depth range computation is not applied" | radeonsi clamps `gl_FragDepth` into [near, far] (llvmpipe and zink conform). Reported as XFAIL on "radeonsi" via `Check::expect_or_known_deviation` |

| `shader-tests/glsl300/multiple-outputs-{need,partial}-locations.frag` | ES 3.0 §3.9.2.3, GLSL ES 3.00 §4.3.8.2: with more than one fragment output, all need `layout(location)`; CompileShader must fail | Mesa compiles and links them, assigning locations itself (XFAIL on Mesa). Duplicate/out-of-range locations are rejected, but at link time rather than at CompileShader |
| **crash** (not testable in-process) | ES 3.0 §4.3.2.1: `glReadPixels` with `READ_BUFFER = NONE` → INVALID_OPERATION | radeonsi, llvmpipe and zink **segfault** in libgallium when the bound FBO also has an integer color attachment (e.g. COLOR0 = RGBA8, COLOR3 = RG32I). With only normalized attachments the error is raised correctly. `es30-multiple-render-targets` tests the rule on a normalized-only FBO and never makes the crashing call. A crash-isolating test (separate process) is a backlog item |

| `es30-buffer-mapping` | ES 3.0 §2.10.3.1: `glUnmapBuffer` on an unmapped buffer returns FALSE and raises INVALID_OPERATION | radeonsi returns TRUE (the error is raised); caused by Mesa glthread (passes with `mesa_glthread=false`). XFAIL on radeonsi |
| `es30-uniform-buffer-objects` | ES 3.0 §2.10.1.1: `glBindBufferRange` with a negative offset → INVALID_VALUE | radeonsi, llvmpipe, zink raise no error when the negative offset is a multiple of the alignment; drawing with that binding then crashes (llvmpipe segfault, radeonsi context loss). The demo never draws with it. XFAIL on all three |

| `es30-integer-float-textures` | ES 3.0 §3.8.3 / ES 3.2 §8.5: an invalid format/type/internalformat combination → INVALID_OPERATION | integer formats with type FLOAT (e.g. `R32I` + `RED_INTEGER` + `FLOAT`) raise INVALID_ENUM (desktop EXT_texture_integer behaviour). All Mesa drivers |
| `es30-integer-float-textures` | ES 3.0 §2.12.9.7: two active samplers of different types on one unit → INVALID_OPERATION at draw | `sampler2D` + `usampler2D` on unit 0 draws without error (Mesa compares only texture targets; `sampler2D` + `samplerCube` is caught, see es20-shader-compilation). All Mesa drivers |
| `es30-transform-feedback` | ES 3.0 §2.5: an erroring command has no effect | radeonsi raises INVALID_OPERATION for `glUseProgram` during active capture but still makes the program current |
| `es30-sync-objects` | ES 3.0 §6.1.8: `length` returns the number of integers actually written | `glGetSynciv` with `bufSize` 0 writes nothing but returns `length` 1. All Mesa drivers |
| `es30-framebuffer-blit-msaa` | ES 3.0 §4.3.3: multisample read framebuffer and non-identical formats → INVALID_OPERATION | an RGBA8 → SRGB8_ALPHA8 resolve raises no error. All Mesa drivers |
| `es30-framebuffer-invalidate` | ES 3.0 §6.1.13 / ES 3.2 §9.2.3: default framebuffer bound and attachment not BACK/DEPTH/STENCIL → INVALID_OPERATION | INVALID_ENUM is raised instead. All Mesa drivers |
| `es30-texture-3d-and-arrays` | ES 3.0 §2.1: commands execute in order; reads see all prior commands | zink: `glCopyTexSubImage3D` directly after scissored `glClear`s copies stale texels (a `glFlush` in between fixes it) |

Not deviations (verified during integration): Mesa accepting `glGenerateMipmap` on R32F and unsized
`RGB + FLOAT` uploads is correct, because the contexts expose `GL_OES_texture_float_linear`,
`GL_EXT_color_buffer_float` and `GL_OES_texture_float`. The ES 3.0 transform-feedback mode rule failing under
`MESA_GLES_VERSION_OVERRIDE=3.0` is an artifact of the override, since Mesa remains an ES 3.2 driver. The
sRGB decode accuracy is not specified, so it is measured but not asserted.

Because of the softpipe miscompilation row, softpipe is not used as a general correctness oracle. It runs only where
its native ES 3.1 boundary is informative (ES 3.2 executables must SKIP, plus version-gating tests
that opt in with `"extra_drivers": ["softpipe"]`).

These are reported as XFAIL only on the matching driver (`GL_VERSION` contains "Mesa", or
`GL_RENDERER` contains "softpipe"). Every other driver gets the strict check. Consequence: demos that
need a multisampled default framebuffer report BLOCKED_ENVIRONMENT on softpipe.

## Scope

- Not a Khronos conformance test; see `docs/VALIDATION.md` "Limits of this evidence".
- Extensions are out of scope (`docs/EXTENSIONS_POLICY.md`).
- Undefined behavior (rendering feedback loops, out-of-bounds access without robustness
  guarantees) is explained in documentation, never asserted in tests.

## Current coverage

The project is in an early phase; see `docs/generated/COVERAGE_SUMMARY.md` for current numbers
and `docs/AGENT_HANDOFF.md` for what is next. Anything not listed there as passing is not covered.
