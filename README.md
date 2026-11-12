# opengles3-demos

An educational and validation suite for the **OpenGL ES 3.0, 3.1 and 3.2** core API and
**GLSL ES 3.00 / 3.10 / 3.20**, written in C++23 against the
[meta-gl](../../libcna/meta-gl) type-safe wrapper.

- **Independent executables.** Each demo is one small program with its own shaders, README and
  manifest. There is no engine; every OpenGL ES call is visible in the demo's `main.cpp`.
- **Validated, not just run.** Each demo checks its observable result (pixels against a CPU
  reference, buffer readback, state queries, expected errors) on real drivers, at its minimum ES
  version and at ES 3.2.
- **Specification-derived coverage.** The 358 core entry points come from the Khronos registry;
  features and GLSL language items are tracked separately (`docs/`).

> **Status (2026-11-11):** 82 executables: 75 demos, 6 tests and `glsl-compile-tests`.
> - **API:** all 358 core entry points of ES 2.0/3.0/3.1/3.2 have passing, validated evidence.
> - **Semantic features:** every ES 3.0-3.2 row is validated; one ES 2.0 row (feedback loops) has undefined
>   results and is documented only.
> - **GLSL ES:** every 1.00/3.00/3.10/3.20 inventory row is validated.
>
> The suite runs on radeonsi, llvmpipe and zink, each at the maximum and the minimum ES version; ES 3.2
> executables also run on softpipe. Driver deviations from the spec are recorded as XFAIL with spec quotes
> ([docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md)). Live numbers:
> [docs/generated/COVERAGE_SUMMARY.md](docs/generated/COVERAGE_SUMMARY.md).
> This is not a Khronos conformance suite.

Where to start: [docs/DEMO_CATALOG.md](docs/DEMO_CATALOG.md) lists every executable in learning order. The ★
entries are 3D scenes: camera, lighting, normal and shadow mapping, deferred shading, post-processing, skybox
and fog, compute particles, tessellated terrain, and geometry shaders.

## Requirements

- Linux, CMake ≥ 3.23, Ninja, a C++23 compiler (GCC 14 / Clang 19 tested)
- EGL + an OpenGL ES 3.x driver (Mesa radeonsi/llvmpipe tested)
- meta-gl source checkout at `../../libcna/meta-gl`, **on `develop` at `333d777` or later** (earlier revisions
  lack enumerators that 17 executables need; see
  [docs/METAGL_DEPENDENCY.md](docs/METAGL_DEPENDENCY.md)). Override the path with `-DOGLES3_METAGL_SOURCE_DIR=...`.
- Optional: SDL3 for windowed mode; Python 3 for the coverage tools

## Build and run

```sh
cmake --preset debug
cmake --build --preset debug              # jobs: 8 (owner's CPU limit)
ctest --preset debug                       # every demo × {gpu, llvmpipe, zink} × {es-max, es-min}

./build/debug/bin/es30-hello-triangle                 # window (SDL3)
./build/debug/bin/es30-hello-triangle --headless --validate --capture tri.png
./build/debug/bin/es30-hello-triangle --help
```

Presets: `debug`, `release`, `sanitize` (ASan/UBSan for CPU-side code), `headless-ci`
(llvmpipe only, warnings as errors, no SDL3).

## Common options

`--help`, `--validate`, `--headless`, `--window`, `--frames N`, `--width N`, `--height N`,
`--capture FILE.png`, `--seed N`, `--debug`, `--result-json FILE`.
Exit status: 0 PASS, 1 FAIL, 77 SKIP_UNSUPPORTED / BLOCKED_*, 2 usage error.

## Documentation

| Document | Content |
|---|---|
| [docs/DEMO_CATALOG.md](docs/DEMO_CATALOG.md) | every executable in learning order |
| [docs/API_COVERAGE.csv](docs/API_COVERAGE.csv) | all 358 core entry points with evidence (generated) |
| [docs/FEATURE_COVERAGE.md](docs/FEATURE_COVERAGE.md) | semantic feature inventory and method |
| [docs/GLSL_COVERAGE.md](docs/GLSL_COVERAGE.md) | GLSL ES coverage strategy |
| [docs/VALIDATION.md](docs/VALIDATION.md) | what a PASS means, test matrix, tolerances |
| [docs/META_GL_GAPS.md](docs/META_GL_GAPS.md) | meta-gl audit, gaps and fixes |
| [docs/METAGL_DEPENDENCY.md](docs/METAGL_DEPENDENCY.md) | exact meta-gl revision required, behaviour on public `develop` |
| [docs/EXTENSIONS_POLICY.md](docs/EXTENSIONS_POLICY.md) | core-only policy, future extension module |
| [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | environment and scope limits |
| [docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md) | current state and next tasks |
| [AGENTS.md](AGENTS.md) | rules for contributors and AI agents |

## License

MIT — see [LICENSE](LICENSE).
