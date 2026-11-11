# Agent handoff

_Keep this file current. It is the first thing any agent reads._

## Current phase

**Phase 1 (infrastructure) complete; Phase 2 (ES 2.0 foundations) starting.** Last update: 2026-11-11.

## Done

- Phase 0: meta-gl audit (358/358 core wrappers; enum-domain gaps G1–G3 found and fixed on meta-gl
  branch `opengles3-demos/enum-gaps`, commit `146e171`; see `docs/META_GL_GAPS.md`), environment
  survey, registry-derived inventory (`inventory/`), feature + GLSL inventories.
- Phase 1: CMake + presets, meta-gl via `add_subdirectory`, `common/` runtime (EGL pbuffer +
  SDL3 window, CLI, checks, PNG capture), CTest matrix (gpu/llvmpipe × es-max/es-min), result
  JSON, coverage generator, docs.
- Demos: `es30-hello-triangle`, `es20-shader-compilation` (FULLY_VALIDATED, captures viewed).
- `shader-tests/` (`glsl-compile-tests`): 37 data-driven GLSL ES 1.00-3.20 compile/link cases with
  `@expect/@min-es/@spec/@feature/@xfail` annotations; two Mesa deviations recorded as XFAIL.
- Test matrix extended to 4 drivers (gpu, llvmpipe, softpipe = native ES 3.1, zink). Finding: the
  `MESA_GLES_VERSION_OVERRIDE` es-min runs lower only the API version; docs corrected accordingly.

## Environment (reference machine)

- AMD Radeon 780M radeonsi + llvmpipe, Mesa 25.0.7, EGL 1.5 surfaceless platform with pbuffers.
- `MESA_GLES_VERSION_OVERRIDE=3.0|3.1` gives genuine lower-version contexts (GLSL 3.00/3.10).
- SDL3 3.4.0 in `/usr/local`; windowed runs tested with `xvfb-run -a`.
- Khronos specs are not committed. Fetch them into a scratch directory when needed
  (URLs in `docs/references/README.md`).

## Build / test

```
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
cmake --build --preset debug --target coverage
```

## Next tasks (in order)

1. Merge the parallel foundation-demo branches (clear/scissor, topology, vertex attributes, indexed
   drawing, depth, stencil, blending, culling) after review + full 4-driver matrix.
2. More shader-test cases (3.10 image/atomic/memory-qualifier rules, 3.20 geometry/tess negatives).
3. Stage 1–2 demos from `docs/DEMO_CATALOG.md` (es20-clear-viewport-scissor,
   es20-shader-compilation, es20-vertex-attributes, es20-indexed-drawing, es20-primitive-topology,
   es20-uniforms, es20-culling-front-face, es20-depth-test, es20-stencil-test, es20-blending).
4. `tests/negative/` death tests for meta-gl contract violations (`std::terminate`).
5. Stage 3–4 (textures, framebuffers, ES 3.0 data paths), then `glsl300-builtins` via transform feedback.
6. `diagnostics-capabilities`.

## Parallel work / Codex

From ~2026-11-11 22:00 CET the Codex weekly limit resets; the owner wants Codex CLI used heavily.
Delegate **one demo per Codex task** in its own worktree (`git worktree add ../ogl3-wt-<name> -b demo/<name>`),
giving it AGENTS.md, this file, the hello-triangle demo as the template, and the exact feature IDs.
The integrating agent reviews, runs the full matrix, views the capture, and merges.

## Known issues / open questions

- None blocking. Review O5 in META_GL_GAPS.md (location-vs-index overload of `glGetActiveAttrib`)
  before the vertex-attributes demo.
