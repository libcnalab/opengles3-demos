# Project status: evidence-based (2026-11-12, 02:00)

This report is checked against the acceptance criteria of the owner's continuation directive (§17). Numbers come
from `docs/generated/COVERAGE_SUMMARY.md` for source revision a689f78 and from the runs listed. It does not claim
Khronos conformance.

## Numbers

| | total | with passing evidence | notes |
|---|---|---|---|
| Core API entry points (gl.xml, ES 2.0-3.2) | 358 | 358 (all FULLY_VALIDATED) | each `api` claim is linted against the code |
| Semantic features (ES 2.0/3.0/3.1/3.2) | 87/78/30/31 | 86/78/30/31 | ES20.FBO.FEEDBACK_LOOP is DOCUMENTED_ONLY (undefined results); ES30.MULTISAMPLE.DEFAULT is VISUALLY_VERIFIED |
| GLSL ES features (1.00/3.00/3.10/3.20) | 4/51/23/16 | 4/50/23/16 | GLSL300.UNDEFINED_BEHAVIOR is DOCUMENTED_ONLY |
| Executables | 83 | 82 FULLY_VALIDATED, 1 VISUALLY_VERIFIED | 75 demos, 7 tests, glsl-compile-tests |

## Runs behind the numbers

The debug matrix gives **700/700**:
- gpu (radeonsi), llvmpipe and zink, each at es-max and es-min;
- softpipe (native ES 3.1) for the ES 3.2 executables and opt-ins;
- **ANGLE** 2.1.17841 on lavapipe, in mode `native` (ES 3.1) and in `native-min`, which creates genuine ES 3.0 contexts;
- one llvmpipe run at 333x777 per executable.

Other runs:
- **Sanitizer** (ASan/UBSan): Mesa matrix 480/480.
- **headless-ci** (-Werror, llvmpipe, softpipe, ANGLE and size runs): clean build, 340/340 (2026-11-12 02:10).
- **Window sizes:** `tools/sweep_sizes.sh` is clean at 256x256, 333x777, 1000x200, 200x1000, 640x400 and 800x800 on
  llvmpipe. The only exception is the known BLOCKED_ENVIRONMENT of the multisampled default framebuffer.
- **Clean clones** of both repositories reproduce 563/563. That was before the size runs were added, with the same
  executables.

## Acceptance criteria (directive §17)

| # | criterion | status |
|---|---|---|
| 1 | All 358 core functions accounted for | met: registry-derived inventory; 358/358 with evidence |
| 2 | Coverage backed by meaningful evidence | met within the method: validate() checks results; stale results are rejected; `api` claims are linted; 4 independent review rounds found and fixed unmet claims (docs/audits/) |
| 3 | ES 3.0/3.1/3.2 feature categories accounted for | met: see the numbers above |
| 4 | GLSL ES 1.00-3.20 systematically reviewed | met: inventory complete; Codex audit plus GLSL language tests; misreadings corrected (atomic counters without binding, early_fragment_tests compile or link) |
| 5 | Version-specific functionality used correctly | met: shader `#version` per stage audited; **native ES 3.0 evidence** via ANGLE `native-min` (exact ES 3.0 contexts) for all 54 min_es 3.0 executables, besides Mesa's override-based es-min |
| 6 | Every executable builds in its configuration | met: debug, sanitize, headless-ci, clean clone |
| 7 | Every demo has meaningful validation | met: negative controls required and audited; vacuity checks enforced |
| 8 | Destructive negatives isolated | met: crash cases run in re-exec child processes with timeouts; GPU-faulting cases are opt-in only, after the 2026-11-11 desktop freezes |
| 9 | Driver deviations documented and narrowly scoped | met: KNOWN_LIMITATIONS; per-run evidence in generated/DEVIATIONS.md; matching fixed so ANGLE's backend name cannot excuse ANGLE |
| 10 | Implementation-dependent behaviour not asserted | met after this phase: MSAA resolve, filter and derivative precision, and the conditional link rule are now observations; the remaining assumptions are listed in VALIDATION.md |
| 11 | Missing capabilities reported as unsupported | met: SKIP_UNSUPPORTED / BLOCKED_ENVIRONMENT / BLOCKED_META_GL, never PASS |
| 12 | Visual demos correct and understandable | met: captures reviewed per demo; contact sheet at 640x400 reviewed; capstone es30-interactive-3d-scene |
| 13 | Reproducible from a clean checkout | met: VALIDATION.md "Reproducibility" |
| 14 | meta-gl dependency explicit | met: docs/METAGL_DEPENDENCY.md. The suite needs meta-gl `develop` at `333d777` or later (the merged work branch); with the pre-merge `795bcdc`, 17 executables do not build |
| 15 | Accurate educational documentation | met for the reviewed set: Codex educational review applied (10 fixes); READMEs carry "Read this first" routes for the dense demos |
| 16 | Generated reports agree with the repository and evidence | met: generator records the source revision and rejects stale results; CRASHED results cannot vanish |
| 17 | Unresolved limitations identified | see below |
| 18 | Architecture remains small and comprehensible | met: no engine; `common/` grew only by CLI/EGL/ANGLE selection, input, traceability and the crash marker |
| 19 | No unrelated repository modified | met: only opengles3-demos and meta-gl (authorized branch, merged into `develop` on the owner's request); easy-gl, CNA and cna-engine untouched |
| 20 | Status report free of exaggeration | this document |

## Unresolved limitations

- **Native ES 3.0 evidence comes from ANGLE alone** (`angle:native-min`, exact ES 3.0 contexts). Mesa's es-min
  remains an override, and softpipe is native ES 3.1.
- **Mesa and ANGLE share lower layers.** ANGLE's backend here is lavapipe, the same Vulkan driver as zink.
  Independence covers the GL front-end and the GLSL compiler, not rasterization. No other vendor's driver was
  available.
- **Multisampled default framebuffer:** Mesa's EGL path gives llvmpipe and zink a 0x0 pbuffer, so the executable
  stays VISUALLY_VERIFIED (FULLY_VALIDATED requires llvmpipe). Evidence exists from radeonsi and from ANGLE (ES 3.1 and
  native ES 3.0).
- **Assumptions beyond the normative text** remain, each documented in VALIDATION.md: per-sample checks replaced the
  resolve average; the stretched NEAREST display-blit mapping is assumed (presentation only).
- **The ANGLE/RADV GPU-fault case** (es31-image-load-store, 3D-slice image store) runs only with
  `OGLES3_RUN_GPU_FAULT_CASES=1`.
- **Two pixel-exact galleries** (es20-clear-viewport-scissor, es20-primitive-topology) scale by integer factors only.
  At sizes such as 640x400 they stay at 1x, centred.

## Independent reviews held

All reviews are in docs/audits/, each with its disposition:
1. Codex phase 6 audit (6 groups): 19 findings.
2. Codex educational review (4 groups): 10 accepted, 1 rejected.
3. Codex diff review of the post-audit fixes: 5 findings.
4. Codex diff review of the fill-window changes: 4 findings.

ANGLE triage in four groups was a further independent check of the suite's spec readings. It corrected four of
them: atomic counters without binding; SAMPLE_BUFFERS of an attachment-less FBO; early_fragment_tests reported at
compile or link time; length() rounding.
