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

Because of the last row, softpipe is not used as a general correctness oracle. It runs only where
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
