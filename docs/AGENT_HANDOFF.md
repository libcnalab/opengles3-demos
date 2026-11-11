# Agent handoff

_Keep this file current. It is the first thing any agent reads._

## Current phase

**All backlog demos merged; phase 6 (independent audit) next, phase 7 verification green.** Last update: 2026-11-11 20:30.

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

1. Integrate the running agents (below) one by one: merge, full matrix, view captures, verify every
   claimed deviation against the spec and extensions, record it, regenerate coverage, remove the worktree.
2. Codex (after 22:00 CET): ES 3.0 items 6, 10, 14, 16-20 and item 35, then 57, 59 and 62 if they are still open.
3. More 3D: fog, particles (CPU and compute), skybox.
4. Phase 6 audit (coverage gaps: run `coverage` and read docs/generated/COVERAGE_SUMMARY.md), then phase 7
   final verification on a clean build.

## In flight (2026-11-11 23:50)

**State**
- 83 executables. Full matrix 563/563: gpu, llvmpipe, zink × es-max/es-min, softpipe for ES 3.2, and ANGLE on
  lavapipe.
- 82 FULLY_VALIDATED, 1 VISUALLY_VERIFIED (es30-multisample-default-framebuffer, BLOCKED_ENVIRONMENT on
  llvmpipe/zink).
- **Window-size sweep** (`tools/sweep_sizes.sh`, 6 sizes, llvmpipe) is clean.
- **Clean-clone reproducibility** is verified (VALIDATION.md).
- **meta-gl:** branch `opengles3-demos/enum-gaps` at 333d777 is pushed. O8 is fixed (contract tests now real) and
  O12 is resolved by docs (owner's decision).

**Done tonight**
- **Codex phase 6 audit:** 19 findings, all fixed.
- **Codex educational review:** 10 accepted and fixed, 1 rejected.
- **Size sweep:** 25 executables fixed, plus es31-image-load-store.
- **Deviation matcher:** no longer lets ANGLE's backend name excuse ANGLE.
- **GPU safety:** ANGLE runs on lavapipe; GPU-faulting cases are opt-in only.
- Archived reports are in docs/audits/.

**Running**
- Codex independent review of every demo changed after the audit (`git diff 5bd59a6..HEAD`), per
  docs/agent-briefs/DIFF_REVIEW_BRIEF.md. Three groups; reports in ../opengles3-demos-worktrees/diffreview-*/DIFF_REVIEW.md.
- Sanitizer build plus Mesa matrix (`-E :angle:`).

**Known follow-ups**
- es20-clear-viewport-scissor and es20-primitive-topology draw their 64-px panel gallery at native size, so larger
  windows are not filled. The models are pixel-exact, so scaling them is a larger rewrite.
- Inventory has no IDs for shader fog, interactive input determinism or the barycentric wireframe
  (es30-interactive-3d-scene).
- softpipe and ANGLE give only ES 3.1 natively; there is still no native ES 3.0 driver.

## Parallel work / Codex

From ~2026-11-11 22:00 CET the Codex weekly limit resets; the owner wants Codex CLI used heavily.
Use `tools/agents/codex_demo.sh <name> <task-file>`: it creates `../opengles3-demos-worktrees/<name>`
on branch `codex/<name>` and runs `codex exec` (workspace-write sandbox, high reasoning effort) with
`docs/agent-briefs/DEMO_BRIEF.md`. Take tasks in order from `docs/agent-briefs/BACKLOG.md`.
The integrating agent reviews, runs the full matrix, views the capture, and merges.

## Known issues / open questions

- None blocking. Review O5 in META_GL_GAPS.md (location-vs-index overload of `glGetActiveAttrib`)
  before the vertex-attributes demo.
