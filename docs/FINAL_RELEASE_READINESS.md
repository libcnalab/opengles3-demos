# Final release readiness: opengles3-demos 1.0.0-rc.1

Prepared 2026-11-12 under the final stabilization directive. Every number below is from a run listed in section B;
numbers from earlier reports are labelled as historical. This is not a Khronos conformance claim.

## A. Release identity

| | |
|---|---|
| proposed version | **1.0.0-rc.1** (no earlier tags or releases exist; nothing is tagged or pushed yet) |
| tested revision | opengles3-demos **`9050c5a`** (full hash in the manifests): the evidence was collected at this commit from a clean clone. Its executables were compiled from clean clones at `2544bad`; `9050c5a` changes no compiled source, shader, CMake file or test registration (only documentation and the two Python tools), and ninja confirmed every build up to date |
| release-candidate commit | the commit that adds the final version of this report, directly on top of `9050c5a`; it changes documentation and evidence files only |
| meta-gl | **`333d777fadbe53613a9c2eead358ffbfe6a2ef33`** on `develop` (`v0.4.1-9-g333d777`), cloned from <https://github.com/libcna/meta-gl> for the runs. Configure fails without it (`OGLES3_METAGL_REQUIRED_COMMIT`) |
| supported and tested | Linux x86-64 (Debian 13, kernel 6.12); GCC 14.2 (C++23); CMake 3.31 + Ninja; Mesa 25.0.7: radeonsi (AMD Radeon 780M), llvmpipe, zink on lavapipe, softpipe; EGL 1.5 surfaceless pbuffers; SDL3 3.4.0 windows (X11, tested under Xvfb); ANGLE 2.1.17841 (Android SDK emulator build) on lavapipe |
| partly supported | Clang 19.1.7: builds and passes the headless matrix, but with 189 warnings, so the `-Werror` preset fails with Clang (section D, finding 10) |
| expected to work, not tested | other Linux distributions; other conformant ES 3.0–3.2 drivers (deviations would show as FAIL or need a narrowly matched XFAIL); Wayland |
| not supported | desktop OpenGL; ES 2.0-only contexts (ES 2.0 features run on ES 3.x contexts); Windows and macOS (not built or tested; the headless path needs EGL) |

## B. Validation results

### Executed for this candidate (2026-11-12, clean clones, evidence in [evidence/1.0.0-rc.1/](evidence/README.md))

| matrix | build | registered | PASS | SKIP_UNSUPPORTED | BLOCKED_ENVIRONMENT | FAIL | duration |
|---|---|---|---|---|---|---|---|
| standard (`debug`) | 0 warnings, 83 + 1 executables | 701 | 669 | 26 | 5 | **0** | 07:33:38–07:35:08 |
| sanitizer (`sanitize`, ASan + UBSan, full matrix incl. ANGLE) | 0 warnings | 701 | 669 | 26 | 5 | **0** | 07:35:08–07:40:24 |
| strict (`headless-ci`, `-Werror`, llvmpipe + softpipe + ANGLE + size) | **0 warnings, no errors** | 395 | 365 | 26 | 3 | **0** | 07:40:25–07:41:29 |
| Clang 19.1.7 (`headless-ci` without `-Werror`) | 189 warnings; with `-Werror`: fails | 395 | 365 | 26 | 3 | **0** | 07:41:29–07:42:35 |

The same four matrices had first been collected at `2544bad` (06:56–07:28, identical counts). They were collected
again at `9050c5a` after the independent review fixed the evidence collector (section H), so the committed manifests
come from the corrected tool.

The registered count includes the CPU-only `unit-deviation-matcher` (PASS, no result JSON). Breakdown of the
standard matrix by driver and mode (from `results.csv`):

| driver | mode | PASS | SKIP_UNSUPPORTED | BLOCKED_ENVIRONMENT |
|---|---|---|---|---|
| gpu (radeonsi) | es-max (ES 3.2) / es-min (override) | 83 / 70 | 0 | 0 |
| llvmpipe | es-max / es-min / 333x777 | 82 / 69 / 82 | 0 | 1 / 1 / 1 |
| zink (lavapipe) | es-max / es-min | 82 / 69 | 0 | 1 / 1 |
| softpipe (native ES 3.1) | es-max / es-min | 4 / 4 | 13 (ES 3.2 executables) | 0 |
| ANGLE on lavapipe | native (ES 3.1) / native-min (ES 3.0) | 70 / 54 | 13 (ES 3.2 executables) | 0 |

- **Every SKIP_UNSUPPORTED** is an ES 3.2 executable on an ES 3.1 implementation (softpipe, ANGLE): "requires
  OpenGL ES 3.2, context is 3.1".
- **Every BLOCKED_ENVIRONMENT** is `es30-multisample-default-framebuffer` on llvmpipe or zink: "default framebuffer
  is 0x0, requested 256x256 pbuffer (samples=4)" (section D).
- **Known driver deviations:** 650 checks in 202 runs were matched XFAILs, each listed with its spec quote in
  KNOWN_LIMITATIONS.md and per run in [generated/DEVIATIONS.md](generated/DEVIATIONS.md).
- **Clean build:** both repositories were freshly cloned (meta-gl from GitHub) into an empty directory; all three
  presets configured and built there without warnings; the coverage generator on that tree gives 82
  FULLY_VALIDATED + 1 VISUALLY_VERIFIED, with 700 of 700 registered GL runs backed by a fresh result.
- **ANGLE:** native ES 3.1 (83 runs) and genuine ES 3.0 contexts (54 runs; GL_VERSION "OpenGL ES 3.0.0", GLSL ES
  3.00), on lavapipe only.
- **Viewport stress:** each of the 83 executables at 333x777 on llvmpipe: 82 PASS, 1 BLOCKED_ENVIRONMENT (the
  multisampled pbuffer above).
- **Windowed mode:** `es30-hello-triangle`, `es30-interactive-3d-scene`, `es31-compute-particles` and
  `es32-tessellation-terrain` ran in SDL3 windows (X11 under Xvfb, llvmpipe) at 640x400 with `--validate`: all PASS;
  the captures were viewed. Keyboard and mouse control was not driven automatically.
- **GPU:** the kernel log was read (`"kernel_log": "read"`) and shows no amdgpu fault, timeout or reset during any
  of the runs (recorded per manifest).
  The last such event on this machine is the known ANGLE/RADV incident of 2026-11-11 20:38.

### Fault injection (validation infrastructure, 2026-11-12, throwaway clone)

| injected into `es30-hello-triangle` | observed |
|---|---|
| wrong expected clear colour | FAIL, exit 1, "2 check(s) failed" |
| `validate()` returns without checks | FAIL, "validate() recorded no checks" |
| a failed check followed by `demo::Skip` | FAIL, "1 check(s) failed before: injected skip" (SKIP before the fix) |
| `std::abort()` in `render()` | exit 134, result JSON `CRASHED` |
| 150 s sleep (> 120 s CTest timeout) | CTest "Timeout", result JSON `CRASHED` |
| a registered run's result file deleted | generator: "Registered runs without a fresh result", executable demoted from FULLY_VALIDATED |
| a FAIL result of an unregistered test id | generator: ignored and listed |
| meta-gl `main` (`dd3b300`) as dependency | configure stops: "does not contain the required commit 333d777…" |

Deviation matching is covered by the CTest unit test `unit-deviation-matcher` (real renderer strings).

### Historical, not re-executed here

The 2026-11-12 02:00 report gave "700/700" (debug), "480/480" (sanitizer, Mesa runs only) and "340/340"
(headless-ci) at `a689f78`–`a5d9604`. Read correctly, "700/700" meant no failure, with 31 skipped runs. The
sanitizer preset as a whole had never passed: every ANGLE run aborted at startup (fixed in `2544bad`, section D).

## C. Coverage

From [generated/COVERAGE_SUMMARY.md](generated/COVERAGE_SUMMARY.md) (source revision `9050c5a`, meta-gl `333d777`):

| | total | with validated evidence | other |
|---|---|---|---|
| core API entry points (gl.xml, ES 2.0/3.0/3.1/3.2 = 142/104/68/44) | 358 | 358 FULLY_VALIDATED | — |
| semantic features (ES 2.0/3.0/3.1/3.2 = 87/78/30/31) | 226 | 224 FULLY_VALIDATED | 1 VISUALLY_VERIFIED (ES30.MULTISAMPLE.DEFAULT), 1 DOCUMENTED_ONLY (ES20.FBO.FEEDBACK_LOOP: undefined results) |
| GLSL ES items (1.00/3.00/3.10/3.20 = 4/51/23/16) | 94 | 93 FULLY_VALIDATED | 1 DOCUMENTED_ONLY (GLSL300.UNDEFINED_BEHAVIOR) |
| executables | 83 | 82 FULLY_VALIDATED | 1 VISUALLY_VERIFIED (`es30-multisample-default-framebuffer`) |

No executable is partially validated, unsupported everywhere, or BLOCKED_META_GL. An API entry point counts only
where a manifest says how its effect is verified, and a lint checks that the executable calls it. FULLY_VALIDATED
means every registered run is a validated PASS or a justified SKIP_UNSUPPORTED, with gpu and llvmpipe passing at the
minimum and maximum version (VALIDATION.md). The GLSL claims rest on the built-in function tests (transform-feedback
readback against CPU references), the language-rule tests and the data-driven compile/link cases. Rejections tied
to a version are asserted only on native lower-version contexts (softpipe ES 3.1, ANGLE ES 3.1 and ES 3.0).

## D. Final audit

Scope: the validation infrastructure (result classification, deviation matching, the coverage generator, CMake
registration), the dependency, the documentation, GPU safety, and spot checks of the normative questions the
directive lists. Four earlier independent review rounds (`docs/audits/`) had covered the demos' spec readings
(MSAA resolve, filtering, derivative and float precision, geometry-shader link rules, blit scaling). Their
dispositions were checked rather than redone. The derivative checks (`es30-glsl-texture-functions`) assert only the
slopes of affine functions within a stated sanity bound and the §8.9 definition of `fwidth`. The resolve, filtering
and conditional-link rules remain observations.

| # | severity | finding | resolution |
|---|---|---|---|
| 1 | HIGH | `ctest --preset sanitize` failed all 137 ANGLE runs ("ASan runtime does not come first in initial library list"): the preload of ANGLE's libGLESv2 preceded the ASan runtime. The documented "480/480" covered the Mesa runs only | fixed `2544bad`: the ASan runtime is preloaded first (path from the compiler); the full sanitizer matrix now passes |
| 2 | MEDIUM | `demo::Skip`/`BlockedByMetaGl` raised after a failed check turned the run into SKIP (`common/src/runtime.cpp`). No current demo did this; latent | fixed `9e075cb`: FAIL, with the failure count in the reason; fault-injected |
| 3 | MEDIUM | the coverage generator never compared results with the registered runs: a run that was not executed, filtered out or stale did not stop FULLY_VALIDATED, contrary to VALIDATION.md | fixed `9e075cb`: reads `ctest --show-only=json-v1`; missing runs block FULLY_VALIDATED; unregistered results are ignored; fault-injected |
| 4 | MEDIUM | an empty deviation tag matched every driver (`std::string::find("")`), so a demo whose driver lookup found nothing would excuse all drivers. No current call site reached it | fixed `9e075cb`: `demo::deviation_matches` is one function for demos and shader tests; empty matches nothing; `unit-deviation-matcher` pins the rules, including that no Mesa or backend name excuses ANGLE |
| 5 | MEDIUM | documentation contradicted the repository: README (82 executables, "jobs: 8", status of 2026-11-11), VALIDATION.md (ANGLE "on RADV", "no native ES 3.0 driver", "all four drivers are Mesa"), KNOWN_LIMITATIONS ("early phase", "No non-Mesa driver (ANGLE ...)"), META_GL_GAPS O2 (pre-G8 behaviour), AGENT_HANDOFF (phase of 2026-11-11 20:30) | rewritten; README is now the entry point for new developers |
| 6 | LOW | configuring against a meta-gl checkout without the fixes gave 17 obscure compile errors | fixed `9e075cb`: configure-time check of `333d777` |
| 7 | LOW | every generated summary said "with uncommitted changes": the generator checked `git status` after rewriting its own outputs | fixed in the release-candidate commit (generator only; no effect on results) |
| 8 | INFO | CTest's "100% tests passed" counts skips as passes | reports now give PASS/SKIP/BLOCKED separately (VALIDATION.md "Counting") |
| 9 | INFO (policy) | the G1–G8 meta-gl commits the suite needs are AI-authored (owner-authorized, merged into `develop` on request), while meta-gl's committed AI Contribution Policy v2 forbids AI-authored contributions | not an agent decision; no further meta-gl change was made. Documented in METAGL_DEPENDENCY.md "Policy status" with the options for the maintainer |
| 10 | MEDIUM | the README claimed "GCC 14 / Clang 19 tested", but no Clang build was on record. Measured now: Clang 19.1.7 builds and passes the headless matrix, but emits 189 warnings in 30 files (178 `-Wdouble-promotion`), so the strict preset fails with Clang | documented as a limitation (README, KNOWN_LIMITATIONS); GCC 14.2 is the tested strict compiler. Not fixed during the freeze: the fixes are casts in 30 validated demo files, with no effect on behaviour. Follow-up for after 1.0.0 |

**Remaining limitations** (all in KNOWN_LIMITATIONS.md):
- `es30-multisample-default-framebuffer` is VISUALLY_VERIFIED, not FULLY_VALIDATED: Mesa 25.0.7's surfaceless EGL
  path gives llvmpipe and zink a 0x0 multisampled pbuffer. It passes on radeonsi and on ANGLE (ES 3.1 and native
  ES 3.0), which places the defect in Mesa's EGL pbuffer path.
- All tested implementations are Mesa or ANGLE over Mesa's lavapipe. ANGLE adds an independent front-end and
  compiler, not an independent rasterizer. ES 3.2 has Mesa evidence only. No other vendor's driver was available.
- Mesa's `es-min` lowers only the reported version. Native lower-version evidence is softpipe (ES 3.1) and ANGLE
  (ES 3.1, ES 3.0).
- ANGLE ES 3.0 refuses output-array capture in transform feedback (recorded XFAIL, ANGLE-only).
- Two assumptions beyond the normative text are documented in VALIDATION.md (no resolve-average assumption remains;
  the stretched NEAREST display-blit mapping is assumed for presentation only).
- Undefined behaviour (feedback loops, the GLSL undefined-behaviour row) is documented, never asserted.

**Release-blocking issues:** none known after the fixes above and the review fixes in section H.

## E. Reproducibility

```sh
mkdir -p root/libcna root/libcnalab && cd root
git clone https://github.com/libcna/meta-gl libcna/meta-gl
git -C libcna/meta-gl checkout 333d777fadbe53613a9c2eead358ffbfe6a2ef33
git clone https://github.com/libcnalab/opengles3-demos libcnalab/opengles3-demos   # then check out the RC commit
cd libcnalab/opengles3-demos
# (the RC commits are local until the owner pushes them; until then clone the local repository instead)

for p in debug sanitize headless-ci; do
  cmake --preset $p && cmake --build --preset $p
  python3 -I tools/release/collect_evidence.py --build build/$p --label $p --out ../evidence/$p   # or: ctest --preset $p
done
cmake --build --preset debug --target coverage      # docs/generated/: 82 FULLY_VALIDATED + 1 VISUALLY_VERIFIED
```

Requirements outside the repositories: Mesa with the radeonsi (or another hardware), llvmpipe, zink and softpipe
drivers and the lavapipe Vulkan ICD; EGL; SDL3 (optional); Python 3; an ANGLE build for the `angle` runs (found in
`~/Android/Sdk/emulator/lib64/gles_angle` or `-DOGLES3_ANGLE_DIR`). On another machine the `gpu` runs exercise that
machine's driver, and the numbers can differ.

## F. GPU safety

Safeguards, verified in the source:
- The known GPU-faulting case (`es31-image-load-store`, non-layered 3D-slice `imageStore` on ANGLE/RADV) is not run
  on ANGLE at all unless `OGLES3_RUN_GPU_FAULT_CASES=1` (`demos/es31/image-load-store/main.cpp`). The variable was
  not set in any run (recorded per manifest).
- ANGLE's Vulkan driver defaults to lavapipe (`OGLES3_ANGLE_VULKAN_ICD`, recorded per manifest), so the matrix never
  sends ANGLE work to the hardware GPU.
- Driver crash cases run in re-executed child processes with timeouts. CTest kills any run after 120 s, and a
  killed run leaves a `CRASHED` result (fault-injected).
- Compilation stays at ≤ 6 jobs (presets; this work used `-j 2`–`-j 3` while another session compiled).

Outstanding risks: Mesa's own driver crashes in child processes (`test-driver-crash-readpixels-none` on radeonsi)
are CPU-side segfaults, not GPU faults, but they do run on the hardware driver. A run without a GPU fault is not
proof of GPU safety: a different kernel, firmware or Mesa version can fault where this one did not. The suite's
negative tests deliberately feed invalid input to the driver.

## G. Decision

**READY WITH DOCUMENTED LIMITATIONS**, for the release candidate 1.0.0-rc.1. The independent review (section H)
found two HIGH issues and one MEDIUM issue in the evidence collector; all three were fixed and verified, and the
evidence was collected again with the fixed tool. No BLOCKER or HIGH issue remains open.

Evidence: the code under release builds cleanly from fresh clones in all three configurations. The standard,
sanitizer and strict matrices run with zero failures, and their machine-readable evidence is tied to the exact
revisions. The validation infrastructure was fault-injected and catches wrong values, empty validation, late
skips, crashes, timeouts, missing runs and a wrong meta-gl revision. Every API, feature and GLSL inventory row is
accounted for, and every non-PASS run has a capability or environment reason. The known GPU-faulting case stays
opt-in.

The limitations are explicit: one VISUALLY_VERIFIED executable; Mesa-only ES 3.2 evidence; ANGLE only on lavapipe;
strict (`-Werror`) builds need GCC.
Before 1.0.0 final the owner should decide the meta-gl policy question (finding 9) and preferably have meta-gl
tagged, so this suite can pin a tag rather than a `develop` commit.

## H. Independent final review

Reviewer: Codex (a different model and agent from the primary implementation agent), in its own worktree at
`4c6d788`, with [RELEASE_REVIEW_BRIEF.md](agent-briefs/RELEASE_REVIEW_BRIEF.md) (at most 15 high-confidence
findings, each with file, line, evidence, impact, severity and resolution; llvmpipe only, `-j 1`). Report:
[audits/2026-11-12-codex-release-review.md](audits/2026-11-12-codex-release-review.md).

The reviewer reconciled every evidence total with the manifests and this report. It confirmed that
`2544bad..4c6d788` changes no executable or CMake implementation. It built and ran `unit-deviation-matcher` and
`es30-hello-triangle` (llvmpipe), both PASS. It reported nothing against the executables, result classification,
GPU safety, policy or documentation. Its three findings concern `tools/release/collect_evidence.py`; each was checked
against the code by the primary agent and accepted:

| severity | finding | resolution (`9050c5a`) | verified by |
|---|---|---|---|
| HIGH | the revision came from the script's checkout while CTest ran in any `--build` directory, so another checkout's results could carry this revision | refuse unless the build's `CMAKE_HOME_DIRECTORY` is the script's checkout | pointing the working-tree collector at the clean clone's build: refused |
| HIGH | the clean-tree check ignored untracked files, although `shader-tests` discover cases from the source tree at run time | untracked files (outside `.gitignore`) make the tree dirty | an untracked `shader-tests/glsl300/injected.frag`: refused |
| MEDIUM | `kernel_log_checked` was always true | `"kernel_log": "read"` only when journalctl succeeded, otherwise the failure is named | all four new manifests say `read` |

The committed evidence of `2544bad` had in fact been collected correctly: the build directories belonged to the
clean clone, the clones had no untracked files, and journalctl worked. The fixes close the gaps for future runs, and
the evidence was collected again with them at `9050c5a`.
