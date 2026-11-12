# opengles3-demos

An educational and validation suite for the **OpenGL ES 2.0–3.2 programmable core API** and
**GLSL ES 1.00 / 3.00 / 3.10 / 3.20**, written in C++23 against the
[meta-gl](https://github.com/libcna/meta-gl) type-safe wrapper.

- **Independent executables.** Each demo is one small program with its own shaders, README and
  manifest. There is no engine; every OpenGL ES call is visible in the demo's `main.cpp`.
- **Validated, not just run.** Each demo checks its observable result (pixels against a CPU
  reference, buffer readback, state queries, expected errors) on real drivers, at its minimum ES
  version and at the driver's maximum.
- **Specification-derived coverage.** The 358 core entry points come from the Khronos registry (`gl.xml`);
  semantic features and GLSL language items are tracked in separate inventories (`inventory/`).

> **Release candidate 1.0.0-rc.1** (prepared 2026-11-12, not yet tagged). 83 executables: 76 demo programs (including
> `diagnostics-capabilities`), 6 GL tests and `glsl-compile-tests`, plus one CPU-only unit test. 82 are FULLY_VALIDATED and 1 is VISUALLY_VERIFIED
> (`es30-multisample-default-framebuffer`; Mesa's software EGL path cannot create its multisampled pbuffer). All
> 358 core entry points have validated evidence. The readiness report, with the evidence behind every number and
> the remaining limitations, is [docs/FINAL_RELEASE_READINESS.md](docs/FINAL_RELEASE_READINESS.md).
> This is **not** a Khronos conformance suite.

## Which OpenGL ES versions it covers

| | API | shading language | how |
|---|---|---|---|
| ES 2.0 | the ES 2.0 entry points and features (`demos/es20-foundations/`) | GLSL ES 1.00 (compile/link tests only) | exercised on ES 3.x contexts; no ES 2.0-only context is created |
| ES 3.0 | `demos/es30/` | GLSL ES 3.00 | Mesa drivers (radeonsi, llvmpipe, zink), and native ES 3.0 contexts through ANGLE |
| ES 3.1 | `demos/es31/` | GLSL ES 3.10 | Mesa drivers, softpipe (natively ES 3.1), ANGLE (natively ES 3.1) |
| ES 3.2 | `demos/es32/` | GLSL ES 3.20 | Mesa radeonsi, llvmpipe and zink. No other ES 3.2 implementation was available |

Extensions are out of scope ([docs/EXTENSIONS_POLICY.md](docs/EXTENSIONS_POLICY.md)).

## Requirements

- Linux, CMake ≥ 3.23, Ninja, a C++23 compiler. Tested: GCC 14.2 (all presets, warning-free). Clang 19.1 builds
  and passes the headless matrix but emits `-Wdouble-promotion` warnings, so `headless-ci` (`-Werror`) needs GCC
- EGL and an OpenGL ES 3.x driver (tested: Mesa 25.0.7 radeonsi, llvmpipe, zink on lavapipe, softpipe)
- **meta-gl** source checkout at `../../libcna/meta-gl` (relative to this directory), on `develop` at commit
  **`333d777`** (the tested revision). Older revisions lack enumerators that 17 executables need; configuring
  against one stops with an error. Details and the override `-DOGLES3_METAGL_SOURCE_DIR=...`:
  [docs/METAGL_DEPENDENCY.md](docs/METAGL_DEPENDENCY.md).
- Optional: SDL3 for windowed mode; Python 3 for the coverage and evidence tools; an ANGLE build for the `angle`
  test driver

## Build

```sh
git clone https://github.com/libcna/meta-gl ../../libcna/meta-gl
git -C ../../libcna/meta-gl checkout 333d777fadbe53613a9c2eead358ffbfe6a2ef33

cmake --preset debug
cmake --build --preset debug              # 6 parallel jobs (the presets cap compilation at 6 cores)
```

Presets: `debug`, `release`, `sanitize` (ASan/UBSan for CPU-side code), `headless-ci` (warnings as errors, no
SDL3, llvmpipe only).

## Run one demo

```sh
./build/debug/bin/es30-hello-triangle                     # window (SDL3); Esc closes it
./build/debug/bin/es30-hello-triangle --headless --validate --capture tri.png
./build/debug/bin/es30-hello-triangle --help              # every option
```

Options: `--help`, `--validate`, `--headless`, `--window`, `--frames N`, `--width N`, `--height N`,
`--capture FILE.png`, `--seed N`, `--debug`, `--result-json FILE`.
Exit status: 0 PASS, 1 FAIL, 77 SKIP_UNSUPPORTED or BLOCKED_*, 2 usage error. A run prints its checks, one line
each, and a final `RESULT:` line. The reason for a FAIL is in the failing check's line.

Where to start learning: [docs/DEMO_CATALOG.md](docs/DEMO_CATALOG.md) lists every executable in learning order, from
`es20-clear-viewport-scissor` and `es30-hello-triangle` to compute, geometry and tessellation shaders. The ★ entries
are 3D scenes; `es30-interactive-3d-scene` is an explorable capstone (mouse look, WASD, technique toggles 1–6).

## Run the automated tests

```sh
ctest --preset debug                       # the whole matrix (about 700 runs, under 2 minutes here)
ctest --preset debug -L es30 -L llvmpipe   # a subset by labels (group, driver, mode)
ctest --preset debug -R es30-instancing    # one executable on every driver
cmake --build --preset debug --target coverage   # regenerate docs/generated/ from the results
```

Each executable is registered once per driver (`gpu`, `llvmpipe`, `zink`, `softpipe` for ES 3.2 executables,
`angle` when found) and version mode (`es-max`, `es-min`, `native`, `native-min`), plus one llvmpipe run at
333x777. CTest shows SKIP_UNSUPPORTED and BLOCKED_* as "Skipped"; the precise outcome is in
`build/debug/test-results/<test>.json`. [docs/VALIDATION.md](docs/VALIDATION.md) explains the matrix.

## Software rendering (no GPU)

Every demo runs on Mesa's software drivers; the test matrix sets the same variables:

```sh
EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe ./build/debug/bin/es31-compute-basics --headless --validate
GALLIUM_DRIVER=zink ...        # GL on Vulkan (lavapipe)
GALLIUM_DRIVER=softpipe ...    # a native ES 3.1 driver; ES 3.2 executables SKIP there
MESA_GLES_VERSION_OVERRIDE=3.0 # lowers the reported version only (see VALIDATION.md "es-min")
```

## ANGLE

ANGLE is an independent GL ES front-end and shader compiler. The build looks for ANGLE's `libEGL.so` /
`libGLESv2.so` in `~/Android/Sdk/emulator/lib64/gles_angle` (the Android SDK emulator ships one, ANGLE 2.1.17841)
or in `-DOGLES3_ANGLE_DIR=<dir>`, and then adds the `angle` driver to the matrix: `native` (ANGLE's highest version,
ES 3.1) and `native-min` (a genuine ES 3.0 context for every ES 3.0 executable). ANGLE runs on **lavapipe**
(software Vulkan) by default. That tests a second front-end and compiler, not a second rasterizer: zink uses the
same lavapipe.

## Tests that need caution

- **Never point ANGLE at the real GPU on a machine whose desktop uses it.** One ANGLE/RADV case (a non-layered
  3D-slice `imageStore` in `es31-image-load-store`) caused GPU page faults and desktop freezes. It is not run on
  ANGLE unless `OGLES3_RUN_GPU_FAULT_CASES=1` is set, and `OGLES3_ANGLE_VULKAN_ICD` defaults to lavapipe.
- Driver crash cases (`test-driver-crash-readpixels-none`, parts of `es30-program-binary`, `es30-immutable-storage`
  and `es31-image-load-store`) run in child processes with timeouts; a crash there is recorded, not fatal.
- Compile with at most 6 parallel jobs (the presets do).

## What FULLY_VALIDATED means, and what it does not

An executable is **FULLY_VALIDATED** when every registered run has a fresh result written by the current
executable, each is a validated PASS or a justified SKIP_UNSUPPORTED, gpu and llvmpipe pass at the minimum and the
maximum version, and the manifest declares a real validation method. A PASS requires at least one recorded check
and no failed check. A known driver deviation is an XFAIL: it is recorded with a spec quote and matched to one
named driver, and it is never loosened for the others. Details: [docs/VALIDATION.md](docs/VALIDATION.md).

It does **not** mean Khronos conformance, correctness on drivers outside the tested set (all tested
implementations are Mesa or ANGLE on Mesa's lavapipe), GPU memory safety (the sanitizers cover CPU code), or that
`es-min` emulates a real lower-version driver (it only lowers the reported version). Undefined behaviour is
documented, never asserted. Known driver deviations and environment limits:
[docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

## Documentation

| Document | Content |
|---|---|
| [docs/FINAL_RELEASE_READINESS.md](docs/FINAL_RELEASE_READINESS.md) | release identity, evidence, audit, decision |
| [CHANGELOG.md](CHANGELOG.md) | release history and release notes |
| [docs/DEMO_CATALOG.md](docs/DEMO_CATALOG.md) | every executable in learning order |
| [docs/VALIDATION.md](docs/VALIDATION.md) | what a PASS means, test matrix, tolerances, reproducibility |
| [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | driver deviations (XFAIL) and environment limits |
| [docs/API_COVERAGE.csv](docs/API_COVERAGE.csv) | all 358 core entry points with evidence (generated) |
| [docs/generated/](docs/generated/COVERAGE_SUMMARY.md) | coverage summary, per-executable, feature, GLSL and deviation status (generated) |
| [docs/FEATURE_COVERAGE.md](docs/FEATURE_COVERAGE.md), [docs/GLSL_COVERAGE.md](docs/GLSL_COVERAGE.md) | inventory methods |
| [docs/METAGL_DEPENDENCY.md](docs/METAGL_DEPENDENCY.md), [docs/META_GL_GAPS.md](docs/META_GL_GAPS.md) | meta-gl revision, gaps and their fixes |
| [docs/evidence/](docs/evidence/) | machine-readable evidence of the release-candidate matrices |
| [docs/audits/](docs/audits/README.md) | independent review reports and their disposition |
| [docs/references/README.md](docs/references/README.md) | Khronos specifications and registry |
| [docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md), [AGENTS.md](AGENTS.md) | maintenance state; rules for contributors and AI agents |

## License

MIT — see [LICENSE](LICENSE). No third-party code is vendored. The entry-point inventory (`inventory/`) is derived
from the Khronos registry `gl.xml` (Apache 2.0; source and checksum in `inventory/registry_source.txt`); the
specifications themselves are not redistributed.
