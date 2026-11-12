# Focused review brief (1.0.0-rc.2)

You are the independent reviewer of the changes between release candidates 1.0.0-rc.1 and 1.0.0-rc.2 of
opengles3-demos. You did not write them. Find **new, actionable, technically supported problems** in these changes.
Do not restart the earlier audits of the OpenGL ES specification (`docs/audits/`); do not report style.

## The change set

`git diff v1.0.0-rc.1..<reviewed commit>` (the reviewed commit is given in your prompt). In short:

1. Clang 19 `-Wdouble-promotion` and unused-entity fixes in about 30 demo/test sources (`static_cast<double>` at
   the implicit conversion sites, a few helpers, removed dead code) and a `headless-ci-clang` preset.
2. `es30-multisample-default-framebuffer` runs on llvmpipe and zink in an SDL3 window under `xvfb-run` instead of a
   surfaceless pbuffer (`"window_drivers"` in its `demo.json`, `cmake/OglesDemo.cmake`, `CMakeLists.txt`), which
   makes it FULLY_VALIDATED.
3. `common/src/runtime.cpp`: an extra buffer swap when a window resize arrives in the validated frame.
4. `demos/es30/lighting`: sweeps skip sphere pixels within `|N.L| < 0.01` of a light's terminator; surfaces below
   200 px report SKIP_UNSUPPORTED.
5. Documentation (shader-tests README, learning path, meta-gl maintainer note), the 3-core build limit, and the
   rc.2 evidence in `docs/evidence/1.0.0-rc.2/`.

## Focus

1. **Numerical behaviour of the Clang fixes.** Does any change alter a value, move arithmetic between float and
   double, or change which overload is called? (The author compared object code and per-check results; verify
   independently where you doubt it.)
2. **Multisample validation.** Is running the default-framebuffer demo in an X11 window a legitimate test of the
   default framebuffer, with unchanged assertions? Could a window run PASS where the demo's subject was not
   exercised (for example a single-sample window)? Is the fallback without SDL3/xvfb-run honest
   (BLOCKED_ENVIRONMENT, not PASS)? Is the classification by `tools/coverage/generate_coverage.py` still correct?
3. **The lighting and runtime changes:** do they hide real failures, make a check vacuous, or weaken a tolerance?
4. **Evidence integrity:** do the rc.2 manifests and `results.csv` match the commits, compilers and counts claimed
   in `docs/FINAL_RELEASE_READINESS.md`, `docs/FINAL_COMPLETION_REPORT.md` (if present), `README.md`, `CHANGELOG.md`
   and `docs/evidence/README.md`? Is anything claimed that no evidence supports? Is the rc.1 evidence preserved?
5. **GPU safety:** can any default run (including the new window runs) reach the hardware GPU through ANGLE or
   zink, or run the opt-in faulting case?
6. **meta-gl:** the suite still requires `333d777` and no meta-gl change was made; nothing bypasses missing typed
   APIs (no raw enum casts outside `tests/negative`).
7. **Documentation consistency** of the changed documents (counts, commands, presets, limits).

## Rules

- **Do not compile anything** (the owner's CPU limit is in use by the evidence builds). Built executables matching
  the reviewed sources are in the build directory named in your prompt; run them only with
  `EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe` (or under `xvfb-run -a` with
  `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe` for windowed runs). Never use the hardware GPU, never set
  `OGLES3_RUN_GPU_FAULT_CASES`, never point ANGLE or zink at a hardware Vulkan driver.
- Modify nothing except your report `./RC2_REVIEW.md` in your worktree; do not commit. Do not touch meta-gl or any
  other repository.
- **At most 10 findings**, high-confidence only, each with severity (BLOCKER, HIGH, MEDIUM, LOW, INFORMATIONAL;
  BLOCKER/HIGH only for a demonstrated defect that makes a release claim false, hides a failure, breaks a clean
  build, or risks the GPU), exact file and line, explanation, evidence (command and output, or quoted lines),
  impact and recommended resolution. Zero findings is an acceptable result.
- End with a one-paragraph verdict: are there unresolved release blockers?
