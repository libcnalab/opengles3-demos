# AGENTS.md — operating rules for every agent (Claude Code, Codex, humans)

These rules are shared by all agents. `CLAUDE.md` only imports this file; do not create
product-specific rules elsewhere. Read `docs/AGENT_HANDOFF.md` before starting any work.

## Mission

`opengles3-demos` is a suite of **independent C++23 executables** that teach and validate the
OpenGL ES 3.0 / 3.1 / 3.2 core API and GLSL ES 3.00 / 3.10 / 3.20 through the **meta-gl**
library (`../../libcna/meta-gl`). Every claim of coverage must be backed by a validated run.

## Hard rules

1. **meta-gl only.** Demos call `metagl::gl*`. Never call native `gl*` (the build links only
   libEGL and compiles with `GL_GLES_PROTOTYPES=0`, so direct calls fail to build). Never load
   GL function pointers yourself.
2. **meta-gl changes** are authorized by the owner (2026-11-11). Prefer additive,
   backward-compatible changes; breaking changes only when the correct ES semantics require
   it. Work on a meta-gl branch, commit only the files you changed (never stage unrelated
   working-tree changes), run meta-gl's tests (`-DMETAGL_BUILD_TESTS=ON -DMETAGL_BUILD_GPU_TESTS=ON`),
   check dependents (easy-gl, cna-engine, ...), and record every gap and fix in
   `docs/META_GL_GAPS.md`. Never push.
3. **Do not modify easy-gl, CNA, cna-engine** or other projects. Never add an easy-gl dependency.
4. **OpenGL ES, not desktop GL.** No fixed function, no desktop-only calls/enums, no desktop
   GLSL. A `#version 310 es` feature must not appear in a `#version 300 es` shader, etc.
5. **Core only.** Extensions are out of scope except for the diagnostics report
   (`docs/EXTENSIONS_POLICY.md`). Never use an extension to stand in for a core feature.
6. **No fabricated results.** A demo is PASS only if it ran with `--validate` and recorded at
   least one meaningful check. Unsupported -> `SKIP_UNSUPPORTED`; meta-gl limitation ->
   `BLOCKED_META_GL` (throw `demo::BlockedByMetaGl`) with a META_GL_GAPS.md entry.
7. **Spec first.** Semantics come from the Khronos specs (ES 3.0/3.1/3.2, GLSL ES
   3.00/3.10/3.20, gl.xml). Cite sections in README.md and demo.json. ES 2.0/3.0 features cite
   ES 3.0 numbering; ES 3.1/3.2 features cite ES 3.2 numbering.
8. **Small independent programs.** No engine, no renderer abstraction. GL calls stay visible
   in each `main.cpp`. Repetition across demos is acceptable and intended.

## Layout

```
common/            tiny runtime: CLI, EGL/SDL3 context, checks, readback, PNG, math (demo::)
demos/<group>/<d>/ one executable per directory: main.cpp, shaders/, README.md, demo.json, CMakeLists.txt
  groups: es20-foundations (ES 2.0 API features), es30, es31, es32, diagnostics
shader-tests/      GLSL ES compile/link (positive + negative) runner
tests/             integration / negative API tests (separate processes)
inventory/         registry-derived command/enum lists + feature/GLSL inventories (CSV)
tools/registry/    gl.xml extraction;  tools/coverage/ coverage generator
docs/              coverage, validation, gaps, handoff (docs/generated/ is generated)
```

## Adding a demo (the normal unit of work)

1. Create `demos/<group>/<name-without-prefix>/` with:
   - `CMakeLists.txt` containing `ogles_add_demo()` (discovered automatically; do not edit shared CMake files)
   - `demo.json` — copy the schema of `demos/es30/hello-triangle/demo.json`. `name` must be
     `<group-prefix>-<dir>` (e.g. `es30-instancing`), `min_es` the true minimum version,
     `api` maps every core entry point the demo meaningfully exercises to *how it is verified*,
     `features` / `glsl_features` use IDs from `inventory/*.csv` (add new IDs there if missing).
   - `main.cpp` implementing `demo::Demo` (`init/render/validate/shutdown`) and `main()` with a
     `demo::Info{... .source_dir = DEMO_SOURCE_DIR}`.
   - `shaders/*.vert|frag|comp|geom|tesc|tese` with the correct `#version`.
   - `README.md`: purpose, min ES / GLSL version, central API calls, spec sections, shader
     concepts, expected result, how to run, how it is validated, pitfalls. Real prose; no templates.
2. `validate()` must check observable results: pixels against a CPU reference, buffer/texture
   readback, state queries, or expected GL errors. Calling a function is not coverage.
3. Build and run: `cmake --build --preset debug && ctest --preset debug -R <name>`.
   All matrix runs (gpu/llvmpipe/zink × es-max/es-min; softpipe for ES 3.2 executables) must PASS,
   or SKIP_UNSUPPORTED where the driver is below `min_es` (softpipe is natively ES 3.1).
4. Look at the capture (`build/debug/test-results/<test>.png`). Only after actually viewing it
   add `"visual_review": {"date": "YYYY-MM-DD", "by": "<agent>", "notes": "..."}` to demo.json.
5. `cmake --build --preset debug --target coverage` and commit the regenerated docs.

## Commands

```
cmake --preset debug && cmake --build --preset debug     # configure + build
ctest --preset debug                                     # full matrix
ctest --preset debug -L es30 -L llvmpipe                 # subset by labels
./build/debug/bin/<demo> --help                          # options of any demo
./build/debug/bin/<demo> --headless --validate --capture out.png
cmake --build --preset debug --target coverage           # regenerate coverage docs
```

## Conventions

- Compilation uses at most 8 CPU cores in total (owner's rule). The build presets set `jobs: 8`. When several
  agents build in parallel, each passes `-j 2` (or a share such that the total stays <= 8).

- C++23, `namespace mgl = metagl;`, typed enums/handles, no raw `GLenum` casts except where a
  test deliberately exercises an invalid value (and then only in `tests/negative`).
- Warnings are errors in the `headless-ci` preset; keep the build warning-free.
- Commit messages: imperative summary + body explaining *why*. No AI attribution lines.
- Work in small batches; build and run before committing; update `docs/AGENT_HANDOFF.md`.
- Parallel agents: one worktree/branch per agent, only touching their own demo directories
  plus their own rows in `inventory/*.csv`. Shared infrastructure (`common/`, `cmake/`,
  `tools/`) is changed only by the integrating agent.
