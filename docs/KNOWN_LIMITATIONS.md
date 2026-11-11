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

| `es31-indirect-draw` (es-min only) | ES 3.1 §10.5/§12.1.2: any `Draw*Indirect` while transform feedback is active and not paused → INVALID_OPERATION (relaxed in ES 3.2 §12.2.2) | Mesa applies the ES 3.2 rule on ES 3.1 contexts too, including **natively on softpipe** (so not only an override artifact). XFAIL on "Mesa" when the context reports 3.1 |
| `es31-separate-vertex-format` | ES 3.2 Table 21.3, §2.2.2: `VERTEX_BINDING_OFFSET` is 64-bit state; `GetIntegeri_v` clamps | after `glBindVertexBuffer(..., 2^33+8, ...)` `GetInteger64i_v` returns 8 (truncated to 32 bits) and `GetIntegeri_v` is truncated instead of clamped. All Mesa drivers |

| `es31-image-load-store` | ES 3.2 §8.23: `glBindImageTexture` with a format not in Table 8.26 → INVALID_VALUE | RG32F, RG16F, R16F, R8, RG8, R11F_G11F_B10F accepted (desktop ARB_shader_image_load_store list); RGB8 rejected. No exposed extension adds these formats. All Mesa drivers |

| `es31-program-pipelines` | ES 3.2 §7.3/§7.4: `PROGRAM_SEPARABLE` takes effect only after the next successful link; `glUseProgramStages` on a program not linked as separable → INVALID_OPERATION | accepted as soon as the flag is set, without re-link. All Mesa drivers |
| `es31-program-interface-query` | ES 3.2 §7.3.1: `glGetProgramResourceiv` writes at most `count` values; `length` = values written | for `ACTIVE_VARIABLES` with count 3 of 5 Mesa writes all 5 and returns 5 (buffer-overrun hazard). All Mesa drivers |
| `es31-program-interface-query` | ES 3.2 §7.3.1 `REFERENCED_BY_*` | radeonsi reports a vertex-only uniform block as referenced by the fragment shader when the FS declares other blocks |
| `es31-ssbo-std430` | ES 3.2 §6.1.1: negative `glBindBufferRange` offset → INVALID_VALUE | same as the UBO row: no error for alignment-multiple negative offsets. All Mesa drivers |
| `es31-atomics` | GLSL ES 3.10 §4.4.5: an atomic counter without `binding` is bound to unit zero | Mesa rejects it ("atomic counters require explicit binding point", desktop ARB rule). All Mesa drivers |
| `es31-framebuffer-no-attachments` | GLSL ES 3.10 §4.4.1: static `gl_FragDepth` write with `early_fragment_tests` is an error | compiles. All Mesa drivers |
| `es31-framebuffer-no-attachments` | ES 3.2 §9.2.1: an attachment-less FBO has sample buffers iff `FRAMEBUFFER_DEFAULT_SAMPLES` ≠ 0 | `SAMPLE_BUFFERS`/`SAMPLES` read 0 with 4 default samples although rasterization is multisampled. All Mesa drivers |
| `es31-framebuffer-no-attachments` | ES 3.2 §7.11.1, GLSL ES 3.10 §7.1.2: scissor-failed fragments are not shaded; helper invocations have no side effects | llvmpipe/zink (lavapipe), multisampled: pixels outside the scissor sharing a 2×2 quad run with `gl_HelperInvocation` false and their atomics take effect. radeonsi and single-sample runs exact |

| `es32-debug-output` | ES 3.2 §18.1/§18.2: the callback receives a null-terminated message of `length` characters | with an explicit shorter `length` in `glDebugMessageInsert`/`glPushDebugGroup`, Mesa passes the caller's pointer unterminated at `length`. Read callback messages by `length`. All Mesa drivers |
| `es32-debug-output` | ES 3.2 Table 2.3: a command raising INVALID_VALUE is ignored | a label of `MAX_LABEL_LENGTH` characters raises INVALID_VALUE but is still stored (all 11 namespaces + sync). All Mesa drivers |

| `es32-float-targets-and-indexed-blend` | ES 3.2 §15.1.5.5: `IsEnabled(BLEND)` reports draw buffer 0 | radeonsi returns FALSE when only draw buffer 0 has blending enabled via `glEnablei`; llvmpipe/zink conform |
| `es32-float-targets-and-indexed-blend` | ES 3.2 §15.2.2: masked color components are not written | zink (lavapipe) with blending enabled and `glColorMask(T,F,T,T)` on all buffers blends and stores G; other mask combinations misbehave too (exact trigger not isolated) |

| `es32-geometry-shaders` | ES 3.2 §11.3.4.4: `GEOMETRY_VERTICES_OUT` × active output components may not exceed `MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS`; "LinkProgram will fail if it determines that the total component limit would be violated" | `max_vertices = 256` with 128 active output components per vertex (31 vec4 read by the FS + `gl_Position`): 256 × 128 = 32 768 > 4095 (radeonsi) / 1024 (llvmpipe, zink) links. The product is known at link time, so we read "if it determines" as satisfied. All Mesa drivers |
| `es32-layered-rendering` | ES 3.2 §9.2.8: `FramebufferTextureLayer` accepts cube map textures; layer selects the face (Table 8.24) | no error, but the previous attachment stays in place (the cube map is not attached). Layered `glFramebufferTexture` on a cube map works. All Mesa drivers |

Not deviations (verified during integration): Mesa accepting `glGenerateMipmap` on R32F and unsized
`RGB + FLOAT` uploads is correct, because the contexts expose `GL_OES_texture_float_linear`,
`GL_EXT_color_buffer_float` and `GL_OES_texture_float`. The ES 3.0 transform-feedback mode rule failing under
`MESA_GLES_VERSION_OVERRIDE=3.0` is an artifact of the override, since Mesa remains an ES 3.2 driver. The
sRGB decode accuracy is not specified, so it is measured but not asserted. An RGB32F color attachment reported as
`FRAMEBUFFER_UNSUPPORTED` (instead of `INCOMPLETE_ATTACHMENT`) is allowed: with two violated clauses the returned
token is implementation-dependent (ES 3.2 §9.4.2).

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
