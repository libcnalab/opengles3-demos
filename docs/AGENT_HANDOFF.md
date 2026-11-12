# Agent handoff

_Keep this file current. It is the first thing any agent reads._

## Current phase

**Release candidate 1.0.0-rc.2 prepared; functionality frozen; waiting for the owner.** Last update: 2026-11-12.

The suite is complete within its declared scope. Reports: [FINAL_RELEASE_READINESS.md](FINAL_RELEASE_READINESS.md)
and [FINAL_COMPLETION_REPORT.md](FINAL_COMPLETION_REPORT.md). Release history: [../CHANGELOG.md](../CHANGELOG.md).
Nothing is running and no worktree or agent task is open. Nothing has been pushed or tagged remotely; the owner
decides about publishing 1.0.0 and about the meta-gl policy question ([METAGL_MAINTAINER_NOTE.md](METAGL_MAINTAINER_NOTE.md)).

Do not add demos or features. Fix only demonstrated defects (with a regression check where possible), rerun the
matrices with `tools/release/collect_evidence.py` from a clean clone, and update the readiness report.

## State

- 83 executables (76 demo programs, 6 GL tests, `glsl-compile-tests`) plus the CPU test `unit-deviation-matcher`,
  all 83 FULLY_VALIDATED. `es30-multisample-default-framebuffer` runs on llvmpipe and zink in an SDL3 window under
  `xvfb-run` (`"window_drivers"`), because Mesa's surfaceless EGL gives those drivers a 0x0 multisampled pbuffer.
- 358/358 core entry points, 226 semantic features and 94 GLSL items accounted for (generated/COVERAGE_SUMMARY.md).
- GCC 14.2 and Clang 19.1.7 strict builds (`headless-ci`, `headless-ci-clang`) are warning-free.
- Evidence of rc.2 in `docs/evidence/1.0.0-rc.2/`; rc.1 evidence preserved in `docs/evidence/1.0.0-rc.1/`.
- meta-gl: required and tested at `develop` `333d777` (contains G1–G8). **Agents do not modify meta-gl**
  (AGENTS.md rule 2). The policy decision about the AI-authored G1–G8 commits is the owner's.

## Environment (reference machine)

- AMD Radeon 780M radeonsi + llvmpipe, Mesa 25.0.7, EGL 1.5 surfaceless platform with pbuffers.
- `MESA_GLES_VERSION_OVERRIDE=3.0|3.1` lowers the reported API/GLSL version only (the compiler still accepts
  newer `#version`s). softpipe is a native ES 3.1 driver; zink runs on lavapipe.
- ANGLE 2.1.17841 from `~/Android/Sdk/emulator/lib64/gles_angle`, on lavapipe (never RADV by default: one
  ANGLE/RADV case faulted the GPU on 2026-11-11 and froze the desktop).
- SDL3 3.4.0 in `/usr/local`; windowed runs tested with `xvfb-run -a`.
- Khronos specs are not committed. Fetch them into a scratch directory when needed
  (URLs in `docs/references/README.md`).
- At most 3 CPU cores for compilation in total (presets: `jobs: 3`; parallel agents and Codex `-j 1`).

## Build / test

```
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
cmake --build --preset debug --target coverage
python3 -I tools/release/collect_evidence.py --build build/debug --label debug --out <dir>
```

## Maintenance guidance

- **A new driver deviation:** confirm it against the spec text, keep the strict check, add an XFAIL matched to the
  narrowest driver name (`demo::deviation_matches`), add a KNOWN_LIMITATIONS row with the spec quote.
- **A new driver or machine:** run the matrix; anything that fails is either a demo defect (fix it), a driver
  deviation (above), or an environment limit (BLOCKED_ENVIRONMENT, with a KNOWN_LIMITATIONS row). Never loosen a
  check to make a driver pass.
- **A meta-gl change is needed:** record it in META_GL_GAPS.md with spec evidence and a proposed fix; a human writes
  the meta-gl change. Bump `OGLES3_METAGL_REQUIRED_COMMIT` and METAGL_DEPENDENCY.md when the suite starts to need it.
- **GPU safety:** a case that faults or hangs a hardware GPU never runs by default; it needs an explicit opt-in
  variable (`OGLES3_RUN_GPU_FAULT_CASES=1`). Bisect such cases on llvmpipe/lavapipe, not on the GPU. After risky
  experiments check `journalctl -k | grep amdgpu`.

## Open follow-ups (not release blockers)

- A meta-gl release tag containing G1–G8, so METAGL_DEPENDENCY.md can pin a tag instead of a branch commit.
- Evidence from a non-Mesa ES 3.2 driver and from ANGLE on a hardware Vulkan driver (needs other hardware or a
  machine whose display does not depend on the tested GPU).
- O5 in META_GL_GAPS.md (location-vs-index overload of `glGetActiveAttrib`) is a meta-gl API observation; the demos
  use the index overload.
- Three executables validate only their documented frame; an explicit `--frames N --validate` can FAIL on a correct
  render (KNOWN_LIMITATIONS.md, "Scope"). Making their probes pose-independent would be a redesign of their checks.

## Parallel work / Codex

`tools/agents/*.sh` launch Codex in its own worktree (`../opengles3-demos-worktrees/<name>`, branch `codex/<name>`)
with a brief from `docs/agent-briefs/`. The integrating agent reviews, runs the matrix, and merges. The final
release review used `docs/agent-briefs/RELEASE_REVIEW_BRIEF.md`.
