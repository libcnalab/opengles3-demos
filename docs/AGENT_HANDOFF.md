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

## In flight (2026-11-11 21:00): continuation directive (rigour over statistics)

The owner's continuation directive sets the priorities: correctness, gaps, test reliability, reproducibility,
independent validation, then education.

**Done in this phase**
- **meta-gl:** edits are authorized again and the branch `opengles3-demos/enum-gaps` is pushed to origin.
  docs/METAGL_DEPENDENCY.md documents it: with public `develop`, 17 executables do not build.
- **Evidence:**
  - Result JSON records the executable, run time and override value.
  - A run writes `CRASHED` first, so crashes cannot vanish.
  - The generator ignores stale results (code vs. exe mtime, run-time inputs vs. run time), lints `api` claims
    against the code, and writes docs/generated/DEVIATIONS.md (per-run XFAIL/XPASS evidence).
- **ANGLE** (Android SDK build 2.1.17841, native ES 3.1, Vulkan/RADV) runs as driver `angle`, mode `native`.
  Triage done for 22 executables:
  - Most failures were real ANGLE deviations (KNOWN_LIMITATIONS, ANGLE table).
  - Test bugs fixed: es31-indirect-draw was missing a barrier before reusing the command buffer; es31-ssbo-std430
    asserted length() rounding.
  - Two of our readings were wrong and are corrected: atomic counters without binding; SAMPLE_BUFFERS of an
    attachment-less FBO.
  - Three readings were re-verified because ANGLE agrees with Mesa: occlusion-query reuse,
    PROGRAM_SEPARABLE after link, 64-bit VERTEX_BINDING_OFFSET.
- **MSAA:** resolve-dependent hard checks in three demos were demoted to printed observations or replaced by
  per-sample checks (texelFetch on sampler2DMS, stencil/occlusion counting). VALIDATION.md is updated.

**Since 21:00**
- ANGLE triage is complete. All 556 runs pass (now 563, after the capstone demo).
- **GPU safety:** an ANGLE/RADV case faulted the shared GPU and froze the owner's desktop (72 amdgpu lines,
  20:23-20:38). ANGLE now runs on lavapipe by default (`OGLES3_ANGLE_VULKAN_ICD`). Known GPU-faulting cases are
  opt-in only (`OGLES3_RUN_GPU_FAULT_CASES=1`). Rule in VALIDATION.md.
- **Compile limit is now 6 cores** (presets jobs 6; agents -j 2; Codex -j 1).
- **Capstone es30-interactive-3d-scene** is merged: mouse/WASD camera, technique toggles 1-6, 50 checks on every
  driver. New runtime input: `Demo::on_mouse`, `Env::key_held`, `Env::frame_seconds`.

**Owner decisions (2026-11-11, before the night):**
- O12: correct the meta-gl docs, not the code (done, 333d777 pushed).
- Codex audit findings: fix them autonomously after verifying each against the spec, run the full matrix, commit
  locally, and summarise in the morning.

**Running**
- (done) A Claude subagent on meta-gl O8: a mock loader so meta-gl's own contract death tests prove their contracts.
  Branch opengles3-demos/enum-gaps, not pushed yet; the integrator reviews and pushes.
- Codex phase 6 audits: background job from 22:05, three waves of two (es20+es31, es32+es30a, es30b+tests), -j 1.
  Reports go to ../opengles3-demos-worktrees/audit-*.report.md. Triage each finding against the spec.

## Parallel work / Codex

From ~2026-11-11 22:00 CET the Codex weekly limit resets; the owner wants Codex CLI used heavily.
Use `tools/agents/codex_demo.sh <name> <task-file>`: it creates `../opengles3-demos-worktrees/<name>`
on branch `codex/<name>` and runs `codex exec` (workspace-write sandbox, high reasoning effort) with
`docs/agent-briefs/DEMO_BRIEF.md`. Take tasks in order from `docs/agent-briefs/BACKLOG.md`.
The integrating agent reviews, runs the full matrix, views the capture, and merges.

## Known issues / open questions

- None blocking. Review O5 in META_GL_GAPS.md (location-vs-index overload of `glGetActiveAttrib`)
  before the vertex-attributes demo.
