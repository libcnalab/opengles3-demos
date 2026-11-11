# Agent handoff

_Keep this file current. It is the first thing any agent reads._

## Current phase

**Phase 1 (infrastructure) complete; Phase 2 (ES 2.0 foundations) starting.** Last update: 2026-11-11.

## Done

- Phase 0: meta-gl audit (358/358 core wrappers; enum-domain gaps G1–G3 found and fixed on meta-gl
  branch `opengles3-demos/enum-gaps`, commit `146e171`; see `docs/META_GL_GAPS.md`), environment
  survey, registry-derived inventory (`inventory/`), feature + GLSL inventories.
- Phase 1: CMake + presets, meta-gl via `add_subdirectory`, `common/` runtime (EGL pbuffer +
  SDL3 window, CLI, checks, PNG capture), CTest matrix, result
  JSON, coverage generator, docs.
- Demos (all FULLY_VALIDATED, captures viewed): `es30-hello-triangle`, `es20-shader-compilation`,
  `es20-uniforms` (all 42 glUniform* variants), `es20-textures-2d`, `diagnostics-capabilities`.
- Merged from subagents: `es20-blending`, `es20-culling-front-face`.
- Tests: `glsl-compile-tests` (44 cases), `test-metagl-contracts` (fork/SIGABRT death tests +
  context-loss cycle), `test-glsl300-builtins` (42 numeric built-in cases via transform feedback).
- meta-gl gaps G1–G5 fixed on meta-gl branch `opengles3-demos/enum-gaps` (`146e171`, `4021048`).
- `shader-tests/` (`glsl-compile-tests`): 37 data-driven GLSL ES 1.00-3.20 compile/link cases with
  `@expect/@min-es/@spec/@feature/@xfail` annotations; two Mesa deviations recorded as XFAIL.
- Test matrix: gpu, llvmpipe, zink × es-max/es-min; softpipe (native ES 3.1) only for ES 3.2
  executables and opt-ins (`extra_drivers`), because it miscompiles complex shaders.
- Findings so far: es-min override lowers only the API version; Mesa accepts a comment before
  `#version` and non-flat integer VS outputs; softpipe MAX_SAMPLES = 1 and a miscompilation
  (all in KNOWN_LIMITATIONS.md).

## Environment (reference machine)

- AMD Radeon 780M radeonsi + llvmpipe, Mesa 25.0.7, EGL 1.5 surfaceless platform with pbuffers.
- `MESA_GLES_VERSION_OVERRIDE=3.0|3.1` lowers the reported API/GLSL version only (compiler still accepts
  newer `#version`s). softpipe is a native ES 3.1 driver; zink runs on lavapipe.
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
3. Remaining Stage 1–3 demos from `docs/DEMO_CATALOG.md` (es20-uniforms, textures, mipmaps,
   cube map, framebuffer objects, copy-tex-image, read-pixels, ...).
4. `tests/negative/` death tests for meta-gl contract violations (`std::terminate`).
5. Stage 3–4 (textures, framebuffers, ES 3.0 data paths), then `glsl300-builtins` via transform feedback.
6. `diagnostics-capabilities`.

## In flight (2026-11-11 ~14:30)

Claude subagents in `.claude/worktrees/agent-*` (branches `worktree-agent-*`), one per pair of demos:
clear-viewport-scissor + primitive-topology; vertex-attributes + indexed-drawing; depth-test +
stencil-test; mipmaps + cube-map; framebuffer-objects + copy-and-read-pixels; vertex-array-objects +
instancing; multiple-render-targets + sampler-objects. Integration procedure: merge the branch, re-run
`ctest -R <names>` on the full matrix (zink was added after the first batch branched), view the
captures, fix or report issues, regenerate coverage, remove the worktree.

## Parallel work / Codex

From ~2026-11-11 22:00 CET the Codex weekly limit resets; the owner wants Codex CLI used heavily.
Use `tools/agents/codex_demo.sh <name> <task-file>`: it creates `../opengles3-demos-worktrees/<name>`
on branch `codex/<name>` and runs `codex exec` (workspace-write sandbox, high reasoning effort) with
`docs/agent-briefs/DEMO_BRIEF.md`. Take tasks in order from `docs/agent-briefs/BACKLOG.md`.
The integrating agent reviews, runs the full matrix, views the capture, and merges.

## Known issues / open questions

- None blocking. Review O5 in META_GL_GAPS.md (location-vs-index overload of `glGetActiveAttrib`)
  before the vertex-attributes demo.
