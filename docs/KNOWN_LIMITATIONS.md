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
| `shader-tests/glsl300/version-not-first-line.frag` | GLSL ES 3.00 §3.4, 3.10/3.20 §3.3: `#version` must be on the first line | a comment line before `#version` is accepted |
| `shader-tests/glsl300/integer-output-not-flat.vert` | GLSL ES 3.00 §4.3.6 (3.10 §4.3.6): integer vertex outputs must be `flat` | accepted; only integer *fragment inputs* are enforced |

| `diagnostics-capabilities` limits | ES 3.1 Table 20.x: `MAX_SAMPLES` ≥ 4, `MAX_FRAMEBUFFER_SAMPLES` ≥ 4 | softpipe reports 1 for both (no MSAA) while advertising ES 3.1 |

| `es20-uniforms` (38-way branch over 70+ uniforms) | correct execution | softpipe returns wrong comparison results for every branch after the third; same shader is correct on radeonsi, llvmpipe and zink, and each construct works in isolation on softpipe → softpipe backend miscompilation. Reproduce: `GALLIUM_DRIVER=softpipe LIBGL_ALWAYS_SOFTWARE=1 es20-uniforms --headless --validate` |

| `es20-vertex-attributes`, `es30-vertex-array-objects` (printed note, not asserted) | ES 3.0 §2.5: a command that generates an error has no effect on GL state | radeonsi/llvmpipe: a client-memory `glVertexAttribPointer` on a non-zero VAO correctly raises INVALID_OPERATION but still stores the pointer (`glGetVertexAttribPointerv` returns it) |
| `es20-vertex-attributes` (printed note, not asserted) | ES 3.0 §2.5: output parameters are not modified on error | `glGetVertexAttribiv(MAX_VERTEX_ATTRIBS, ...)` raises INVALID_VALUE but writes 0 to the output |

| `es20-depth-test` cell 13 | ES 3.0 §3.9.2.3 / ES 3.2 §14.2.3: fragment depth is clamped to [0,1]; "the depth range computation is not applied" | radeonsi clamps `gl_FragDepth` into [near, far] (llvmpipe and zink conform). Reported as XFAIL on "radeonsi" via `Check::expect_or_known_deviation` |

| `shader-tests/glsl300/multiple-outputs-{need,partial}-locations.frag` | ES 3.0 §3.9.2.3, GLSL ES 3.00 §4.3.8.2: with more than one fragment output, all need `layout(location)`; CompileShader must fail | Mesa compiles and links them, assigning locations itself (XFAIL on Mesa). Duplicate/out-of-range locations are rejected, but at link time rather than at CompileShader |
| **crash** `test-driver-crash-readpixels-none` (child process) | ES 3.0 §4.3.2.1: `glReadPixels` with `READ_BUFFER = NONE` → INVALID_OPERATION | radeonsi, llvmpipe and zink **segfault** inside `glReadPixels` whenever the bound FBO has an integer color attachment: COLOR0 RGBA8 + COLOR3 RG32I, or RG32I alone, read with RGBA/UNSIGNED_BYTE or RGBA_INTEGER/INT. With only normalized attachments the error is raised correctly. The test re-executes itself, runs each case in a fresh child process and reports the crash as an XFAIL |

| `es30-buffer-mapping`, `es20-state-queries-errors` | ES 3.0 §2.10.3.1: `glUnmapBuffer` on an unmapped buffer returns FALSE and raises INVALID_OPERATION | radeonsi returns TRUE (the error is raised); caused by Mesa glthread (passes with `mesa_glthread=false`). XFAIL on radeonsi |
| `es30-uniform-buffer-objects` | ES 3.0 §2.10.1.1: `glBindBufferRange` with a negative offset → INVALID_VALUE | radeonsi, llvmpipe, zink raise no error when the negative offset is a multiple of the alignment; drawing with that binding then crashes (llvmpipe segfault, radeonsi context loss). The demo never draws with it. XFAIL on all three |

| `es30-integer-float-textures` | ES 3.0 §3.8.3 / ES 3.2 §8.5: an invalid format/type/internalformat combination → INVALID_OPERATION | integer formats with type FLOAT (e.g. `R32I` + `RED_INTEGER` + `FLOAT`) raise INVALID_ENUM (desktop EXT_texture_integer behaviour). All Mesa drivers |
| `es30-integer-float-textures` | ES 3.0 §2.12.9.7: two active samplers of different types on one unit → INVALID_OPERATION at draw | `sampler2D` + `usampler2D` on unit 0 draws without error (Mesa compares only texture targets; `sampler2D` + `samplerCube` is caught, see es20-shader-compilation). All Mesa drivers |
| `es30-transform-feedback` | ES 3.0 §2.5: an erroring command has no effect | radeonsi raises INVALID_OPERATION for `glUseProgram` during active capture but still makes the program current |
| `es30-sync-objects` | ES 3.0 §6.1.8: `length` returns the number of integers actually written | `glGetSynciv` with `bufSize` 0 writes nothing but returns `length` 1. All Mesa drivers |
| `es30-framebuffer-blit-msaa` | ES 3.0 §4.3.3: multisample read framebuffer and non-identical formats → INVALID_OPERATION | an RGBA8 → SRGB8_ALPHA8 resolve raises no error. All Mesa drivers |
| `es30-framebuffer-invalidate` | ES 3.0 §6.1.13 / ES 3.2 §9.2.3: default framebuffer bound and attachment not BACK/DEPTH/STENCIL → INVALID_OPERATION | INVALID_ENUM is raised instead. All Mesa drivers |
| `es30-texture-3d-and-arrays` | ES 3.0 §2.1: commands execute in order; reads see all prior commands | zink: `glCopyTexSubImage3D` directly after scissored `glClear`s copies stale texels (a `glFlush` in between fixes it) |

| `es31-indirect-draw` (es-min only) | ES 3.1 §10.5/§12.1.2: any `Draw*Indirect` while transform feedback is active and not paused → INVALID_OPERATION (relaxed in ES 3.2 §12.2.2) | Mesa applies the ES 3.2 rule on ES 3.1 contexts too, including **natively on softpipe** (so not only an override artifact). XFAIL on "Mesa" when the context reports 3.1 |
| `es31-separate-vertex-format` | ES 3.2 Table 21.3, §2.2.2: `VERTEX_BINDING_OFFSET` is 64-bit state; `GetIntegeri_v` clamps | after `glBindVertexBuffer(..., 2^33+8, ...)` `GetInteger64i_v` returns 8 (truncated to 32 bits) and `GetIntegeri_v` is truncated instead of clamped. All Mesa drivers Also ANGLE |

| `es31-image-load-store` | ES 3.2 §8.23: `glBindImageTexture` with a format not in Table 8.26 → INVALID_VALUE | RG32F, RG16F, R16F, R8, RG8, R11F_G11F_B10F accepted (desktop ARB_shader_image_load_store list); RGB8 rejected. No exposed extension adds these formats. All Mesa drivers |

| `es31-program-pipelines` | ES 3.2 §7.3/§7.4: `PROGRAM_SEPARABLE` takes effect only after the next successful link; `glUseProgramStages` on a program not linked as separable → INVALID_OPERATION | accepted as soon as the flag is set, without re-link. All Mesa drivers |
| `es31-program-interface-query` | ES 3.2 §7.3.1: `glGetProgramResourceiv` writes at most `count` values; `length` = values written | for `ACTIVE_VARIABLES` with count 3 of 5 Mesa writes all 5 and returns 5 (buffer-overrun hazard). All Mesa drivers |
| `es31-program-interface-query` | ES 3.2 §7.3.1 `REFERENCED_BY_*` | radeonsi reports a vertex-only uniform block as referenced by the fragment shader when the FS declares other blocks |
| `es31-ssbo-std430` | ES 3.2 §6.1.1: negative `glBindBufferRange` offset → INVALID_VALUE | same as the UBO row: no error for alignment-multiple negative offsets. All Mesa drivers |
| `es31-framebuffer-no-attachments` | GLSL ES 3.10 §4.4.1 / 3.20 §4.4.1.3: a static `gl_FragDepth` write with `early_fragment_tests` "is an error" (reported at compile or link time, GLSL ES §1.3) | compiles and links. All Mesa drivers and ANGLE |
| `es31-framebuffer-no-attachments` | ES 3.2 §7.11.1, GLSL ES 3.10 §7.1.2: scissor-failed fragments are not shaded; helper invocations have no side effects | llvmpipe/zink (lavapipe), multisampled: pixels outside the scissor sharing a 2×2 quad run with `gl_HelperInvocation` false and their atomics take effect. radeonsi and single-sample runs exact |

| `es32-debug-output` | ES 3.2 §18.1/§18.2: the callback receives a null-terminated message of `length` characters | with an explicit shorter `length` in `glDebugMessageInsert`/`glPushDebugGroup`, Mesa passes the caller's pointer unterminated at `length`. Read callback messages by `length`. All Mesa drivers |
| `es32-debug-output` | ES 3.2 Table 2.3: a command raising INVALID_VALUE is ignored | a label of `MAX_LABEL_LENGTH` characters raises INVALID_VALUE but is still stored (all 11 namespaces + sync). All Mesa drivers |

| `es32-float-targets-and-indexed-blend` | ES 3.2 §15.1.5.5: `IsEnabled(BLEND)` reports draw buffer 0 | radeonsi returns FALSE when only draw buffer 0 has blending enabled via `glEnablei`; llvmpipe/zink conform |
| `es32-float-targets-and-indexed-blend` | ES 3.2 §15.2.2: masked color components are not written | zink (lavapipe) with blending enabled and `glColorMask(T,F,T,T)` on all buffers blends and stores G; other mask combinations misbehave too (exact trigger not isolated) |

| `es32-geometry-shaders` | ES 3.2 §11.3.4.4: `GEOMETRY_VERTICES_OUT` × active output components may not exceed `MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS`; "LinkProgram will fail if it determines that the total component limit would be violated" | `max_vertices = 256` with 128 active output components per vertex (31 vec4 read by the FS + `gl_Position`): 256 × 128 = 32 768 > 4095 (radeonsi) / 1024 (llvmpipe, zink) links. The product is known at link time, so we read "if it determines" as satisfied. All Mesa drivers |
| `es32-layered-rendering` | ES 3.2 §9.2.8: `FramebufferTextureLayer` accepts cube map textures; layer selects the face (Table 8.24) | no error, but the previous attachment stays in place (the cube map is not attached). Layered `glFramebufferTexture` on a cube map works. All Mesa drivers |
| `es32-base-vertex-and-robustness` | ES 3.2 §16.1.2: `ReadnPixels` raises INVALID_VALUE if `bufSize` is negative (INVALID_OPERATION, the too-small-buffer rule, would also be acceptable) | `bufSize = -1` raises no error and writes the pixels. `glGetnUniform*v` rejects a negative `bufSize` correctly. All Mesa drivers |
| `es32-multisample-array-and-border-color` | ES 3.2 §8.10: integer `TEXTURE_BORDER_COLOR` values from `glTexParameteriv` are converted with equation 2.2 (`max(c / (2^31 - 1), -1)`), so 0 becomes exactly 0.0 | 0 is stored as 2^-32 (the pre-ES 3.0 `(2c + 1) / (2^32 - 1)` formula). All Mesa drivers |
| `es32-advanced-blend-and-sample-shading` | ES 3.2 §13.4.1: "When the sample shading fraction is 1.0, a separate set of fragment shader input values are evaluated for each sample … at the sample location" | radeonsi: a `SAMPLE_SHADING` + `MinSampleShading(1.0)` draw directly after a draw into a single-sample framebuffer (no flush in between) is shaded once per pixel with sample 0's inputs. A `glFlush` in between fixes it, and the displayed picture uses one; the unflushed order is the XFAIL check. llvmpipe and zink are correct |
| `es30-program-binary` | ES 3.0 §2.12.4: `length` must equal the binary's length, otherwise loading fails with LINK_STATUS FALSE | a binary passed with 16 extra trailing bytes loads with LINK_STATUS TRUE (Mesa `check_crc32` only rejects a header size larger than `length`). All Mesa drivers |
| `es30-program-binary` | ES 3.0 §2.5: an erroring command has no effect on GL state | `glProgramBinary` raising INVALID_ENUM (unlisted format) or INVALID_VALUE (`length` -1) still unlinks the program (`_mesa_clear_shader_program_data` runs before validation). All Mesa drivers |
| `es30-program-binary` | ES 3.0 §2.5: an erroring command does not modify values through pointer arguments | an erroring `glGetProgramBinary` (not linked, or `bufSize` too small) writes `*length = 0`. All Mesa drivers |
| environment (runtime check) | EGL 1.5 §3.5.2: a pbuffer has the requested `EGL_WIDTH`×`EGL_HEIGHT`; ES 3.0 §2.12.1: the initial viewport is the surface size | llvmpipe and zink (lavapipe): a surfaceless pbuffer from a 4-sample config (`EGL_SAMPLES` 4, `SAMPLE_BUFFERS` 1) yields a 0x0 default framebuffer; `glReadPixels` raises INVALID_OPERATION. radeonsi works. The runtime reports BLOCKED_ENVIRONMENT. So `ES30.MULTISAMPLE.DEFAULT` is exercised only on radeonsi, and es30-flat-centroid uses multisample renderbuffers instead |
| `es30-occlusion-queries` | ES 3.0 §2.14: deleting an active query frees its name, but the object stays active until it is no longer active | `glDeleteQueries` on an active query ends it: `CURRENT_QUERY` reads 0 and the following `glEndQuery` raises INVALID_OPERATION. All Mesa drivers |
| `es30-occlusion-queries` | ES 3.2 §4.2: an object created as ANY_SAMPLES_PASSED or ANY_SAMPLES_PASSED_CONSERVATIVE "may be reused for either target" | `glBeginQuery(ANY_SAMPLES_PASSED_CONSERVATIVE)` on an ANY_SAMPLES_PASSED object raises INVALID_OPERATION. All Mesa drivers and ANGLE. The ES 3.1 change log (Bug 13342) states the intent: "occlusion query objects may be specified and reused with any mix of the two valid occlusion query targets" |
| `es30-occlusion-queries` | ES 3.0 §6.1.7: once a query reports available, all earlier queries of the same type are available; an occlusion result is boolean | zink (lavapipe), only after QUERY_RESULT_AVAILABLE has been polled to TRUE: QUERY_RESULT returns a partial sample count (e.g. 30), or 0 for a visible box, and earlier queries still report unavailable. Blocking reads are correct. The polled behaviour is nondeterministic: in a given run many of the zink-matched polled checks pass (XPASS in generated/DEVIATIONS.md) |
| **crash** (worked around) `es30-glsl-texture-functions` | correct execution of a valid shader | a fragment shader with a `switch` over about 80 texture-lookup cases segfaults in libgallium on the first draw on llvmpipe and zink (the llvmpipe backtrace shows a recursion about 80 frames deep). The demo compiles the shader six times, once per sampler family (`#define GROUP n`), so no check needs an XFAIL |
| `es30-depth-textures` | ES 3.0 §3.8.13 / ES 3.2 §8.17: a sized depth(-stencil) texture with `TEXTURE_COMPARE_MODE` NONE and a non-NEAREST filter is incomplete, so it returns (0,0,0,1) | Mesa filters the texture and returns the filtered depth (e.g. 0.75). This affects 5 filter settings × 2 formats; no exposed extension relaxes the rule. All Mesa drivers |
| `es30-depth-textures` | ES 3.0 §3.9.2.1: an incomplete texture returns 0 for a shadow sampler | zink: a mipmap-incomplete DEPTH_COMPONENT32F returns 1.0 (the ALWAYS result) |
| `test-glsl-language` | GLSL ES 3.00 §5.8: "Expressions on the left of an assignment are evaluated before expressions on the right" | `y[j] = ++j` stores into `y[1]` (the index is evaluated after the right side). `x[i++] = i` is correct. GLSL 1.00 and 3.00. All Mesa drivers |
| `test-glsl-language` | GLSL ES 3.00 §6.1.1: arguments are evaluated once, left to right, and an `out` argument yields the l-value used for copy-out | `store(arr[j], ++j)` writes `arr[1]`. All Mesa drivers |
| `shader-tests/glsl300` texture2D case | GLSL ES 3.00 §8.8 lists only the new lookup functions; §6.1 requires a prototype or definition before a call | `texture2D()` compiles in a `#version 300 es` shader. All Mesa drivers |
| `shader-tests/glsl300` ternary l-value case | GLSL ES 3.00 §5.8: "The ternary operator (?:) is also not allowed as an l-value" | `(c ? a : b) = 1.0` compiles. All Mesa drivers |
| **crash** (avoided) `shader-tests` | linking `#version 310 es` with `#version 100` must fail cleanly | under `MESA_GLES_VERSION_OVERRIDE=3.0`, that link segfaults in libgallium (gpu, llvmpipe, zink). The case runs only in `test-glsl-language` on ES ≥ 3.1 |
| `test-glsl320-builtins` | GLSL ES 3.20 §4.3.9: a block's type qualifier "may optionally include layout qualifiers, the auxiliary storage qualifier patch, and the precise qualifier"; §4.9: precise on a block applies to each member | `layout(std430, binding = 2) precise writeonly buffer PreciseOut { ... };` (either qualifier order) fails with "syntax error, unexpected NEW_IDENTIFIER". All Mesa drivers |


### ANGLE 2.1.17841 (Vulkan backend, ES 3.1), matched on "ANGLE"

| Case | Rule | Observed |
|---|---|---|
| `es31-program-pipelines` | ES 3.2 §7.3/§7.4: `PROGRAM_SEPARABLE` takes effect only at the next successful link | same as Mesa: the flag set after the link is honoured immediately |
| `es31-program-pipelines` | ES 3.2 §11.1.3.11: `ValidateProgramPipeline(p)` validates *p* against the current state | validates the currently bound pipeline instead of *p* (its info log names the bound pipeline's error) |
| `es31-program-pipelines` | ES 3.2 §7.4: `UseProgramStages` with a shader-object name → INVALID_OPERATION | INVALID_VALUE |
| `es31-program-interface-query` | ES 3.2 §7.3.1.1: a name identifying an array element other than the first ("a[1]") does not match | `"o_id[1]"` returns the index of the `o_id[0]` entry |
| `es31-ssbo-std430` | ES 3.1/3.2 §7.8: a block larger than `MAX_SHADER_STORAGE_BLOCK_SIZE` "will fail to link" | the limit is 2^32-1 and a 2^32-byte block links |
| `glsl-compile-tests`, `test-glsl-language` | GLSL ES 3.00 §3.8 / 3.10 §3.7: defining an identifier containing `__` "does not itself result in an error" | rejected with the GLSL ES 1.00 wording (also in `#version 310 es`); `__` macro names only warn, which conforms |
| `glsl-compile-tests` | GLSL ES 3.00 §4.6.1: `invariant` "followed by a comma separated list of previously declared identifiers" | `invariant a, b;` is a syntax error |
| `glsl-compile-tests` | GLSL ES 3.00 §4.6.1 + §1.3: `invariant` after a use must be reported by link time at the latest | compiles and links |
| `glsl-compile-tests` | GLSL ES 3.00 §4.3.3: dFdx/dFdy/fwidth of a constant expression in an initializer return 0 (constant expression) | rejected ("assigning non-constant") |
| `es20-clear-viewport-scissor` | ES 3.0 §2.13.1: viewport width/height "are clamped to implementation-dependent maximums when specified" | `GL_VIEWPORT` returns the unclamped size (MAX + 1000) |
| `es20-framebuffer-objects` | ES 3.0 Table 6.15: `RENDERBUFFER_*_SIZE` initially 0 | a renderbuffer without storage reports R4 G4 B4 A4 |
| `es20-shader-compilation` | ES 3.0 §2.12.9.7: ValidateProgram checks the conflicting-sampler-types condition | `VALIDATE_STATUS` TRUE (the draw does fail with INVALID_OPERATION) |
| `es30-depth-textures` | ES 3.0 §3.8.15.1: Dref and Dt are clamped to [0, 1] for fixed-point depth formats | not clamped for DEPTH_COMPONENT24 / DEPTH24_STENCIL8 (DEPTH_COMPONENT16 is) |
| `es30-etc2-eac` | ES 3.0 §3.8.6: a sub-image whose size is not a block multiple is allowed when it reaches the level edge | 2x2 at (4,4) of a 6x6 texture → INVALID_OPERATION |
| `es30-etc2-eac` | ES 3.0 Appendix C.1.5 (eq. C.2, bit replication): 11-bit code 2047 is exactly 1.0 | 11-bit codes zero-filled to 16 bits: x·32/65535, 2047 → 0.99953 |
| `es30-etc2-eac` | ES 3.0 Appendix C.1.7 (eq. C.8) | signed R11/RG11 values 4 codes too far from zero (worked example: 324 instead of 320) |
| `es30-integer-float-textures` | ES 3.0 §3.8.3.2: RGB9_E5 shared exponent uses ⌊log2(maxc)⌋ | 4 of 64 texels get an exponent one too large (matches ⌊ln(maxc)⌋) |
| `es30-framebuffer-invalidate` | ES 3.0 §4.3.3, §4.4.2.4: a blit reads the attached 3D-texture slice | `glBlitFramebuffer` from slice > 0 copies slice 0 (`glReadPixels` reads the right slice) |
| `es30-framebuffer-invalidate` | ES 3.2 §9.2.3: `GetFramebufferAttachmentParameteriv` with an FBO bound: BACK → INVALID_ENUM; COLOR_ATTACHMENTm, m ≥ MAX → INVALID_OPERATION | the two errors are swapped |
| `es30-multiple-render-targets` | ES 3.0 §4.2.1: draw buffers correspond in order to the fragment colours | after `glDrawBuffers` on the bound FBO the next draw still uses the previous mapping (`DRAW_BUFFERi` reads correctly; re-binding the FBO fixes it) |
| `es30-pixel-buffer-objects` | ES 3.0 §4.3.2.4: a pack-buffer offset not divisible by the type size → INVALID_OPERATION | no error (the unpack rule is enforced) |
| `es30-transform-feedback` | ES 3.0 §2.15.2: ResumeTransformFeedback with a different program active → INVALID_OPERATION | resumes |
| `es31-stencil-texturing-and-level-queries` | ES 3.2 §8.11.3: compressed formats report the post-decompression component types | RGB8_ETC2 R/G/B types read NONE |
| `es31-stencil-texturing-and-level-queries`, `es31-multisample-textures` | ES 3.1 §8.8 / 3.2 §8.11.3: multisample textures have only level 0 | `GetTexLevelParameteriv(…, level 1)` raises no error |
| `es31-multisample-textures` | ES 3.2 §8.14.4: `GenerateMipmap` target list | `TEXTURE_2D_MULTISAMPLE` raises no error (INVALID_ENUM expected) |
| intermittent (not reproduced) `es31-image-load-store` | correct execution | one SIGSEGV of `es31-image-load-store:zink:es-min` during a full `ctest -j6` run on 2026-11-11. Not reproduced in 16 sequential and 90 parallel (3×15×2) repeats. No check is changed; if it recurs, isolate the call in a child process |

Not deviations (verified during integration): `es31-framebuffer-no-attachments` SAMPLE_BUFFERS/SAMPLES of an
attachment-less FBO with default samples 4 was our misreading. §9.2.3.1 derives the queried values only from
attachments, so the check now asserts only the §9.2.3.1 relation between them. `DEPTH_STENCIL_ATTACHMENT` with nothing
attached may give NONE (Mesa) or INVALID_OPERATION (ANGLE), since the text supports both. ANGLE conforms on three rules
where Mesa has XFAILs (default-framebuffer attachment query, deleting an active query, Draw*Indirect during
transform feedback); this supports our readings of them. The earlier `es31-atomics` row ("Mesa rejects an atomic counter
without binding") was our misreading: GLSL ES 3.20 §4.4.6 makes it a compile-time error (the change log calls it
a clarification), and Mesa and ANGLE both reject it correctly. Further: Mesa accepting `glGenerateMipmap` on R32F and unsized
`RGB + FLOAT` uploads is correct, because the contexts expose `GL_OES_texture_float_linear`,
`GL_EXT_color_buffer_float` and `GL_OES_texture_float`. The ES 3.0 transform-feedback mode rule failing under
`MESA_GLES_VERSION_OVERRIDE=3.0` is an artifact of the override, since Mesa remains an ES 3.2 driver. The
sRGB decode accuracy is not specified, so it is measured but not asserted. An RGB32F color attachment reported as
`FRAMEBUFFER_UNSUPPORTED` (instead of `INCOMPLETE_ATTACHMENT`) is allowed: with two violated clauses the returned
token is implementation-dependent (ES 3.2 §9.4.2).

Because of the softpipe miscompilation row, softpipe is not used as a general correctness oracle. It runs only where
its native ES 3.1 boundary is informative (ES 3.2 executables must SKIP, plus version-gating tests
that opt in with `"extra_drivers": ["softpipe"]`).

Per-run evidence for every deviation check (which runs failed, which matched runs passed, and which are es-min
override artifacts) is generated in [generated/DEVIATIONS.md](generated/DEVIATIONS.md). A 2026-11-11 audit of
that table found every "Mesa"-matched deviation reproduced on radeonsi, llvmpipe and zink.

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
