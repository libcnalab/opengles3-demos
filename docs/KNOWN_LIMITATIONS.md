# Known limitations

## Environment

- Reference machine: AMD Radeon 780M (radeonsi) and llvmpipe, Mesa 25.0.7, Linux 6.12, EGL 1.5.
  No non-Mesa OpenGL ES driver (ANGLE, Mali, Adreno, PowerVR, NVIDIA ES) has been tested.
  Results therefore say little about vendor-specific behavior.
- Minimum-version runs use `MESA_GLES_VERSION_OVERRIDE`; on non-Mesa drivers only `es-max`
  runs are meaningful (set `-DOGLES3_TEST_MIN_VERSION=OFF`).
- Windowed mode (SDL3) is tested under Xvfb; headless EGL pbuffers are the primary test path.
- Program binaries: `NUM_PROGRAM_BINARY_FORMATS` may be 0 on some drivers; such checks report
  the implementation's choice rather than failing.

## Scope

- Not a Khronos conformance test; see `docs/VALIDATION.md` "Limits of this evidence".
- Extensions are out of scope (`docs/EXTENSIONS_POLICY.md`).
- Undefined behavior (rendering feedback loops, out-of-bounds access without robustness
  guarantees) is explained in documentation, never asserted in tests.

## Current coverage

The project is in an early phase; see `docs/generated/COVERAGE_SUMMARY.md` for current numbers
and `docs/AGENT_HANDOFF.md` for what is next. Anything not listed there as passing is not covered.
