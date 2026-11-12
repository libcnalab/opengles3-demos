# Final release readiness: opengles3-demos 1.0.0-rc.2

Prepared 2026-11-12 under the final completion directive. Every number below is from a run listed in section B. The
rc.1 version of this report is `git show v1.0.0-rc.1:docs/FINAL_RELEASE_READINESS.md`; its evidence stays in
`docs/evidence/1.0.0-rc.1/`. What changed between the candidates and why: [FINAL_COMPLETION_REPORT.md](FINAL_COMPLETION_REPORT.md).
This is not a Khronos conformance claim.

## A. Release identity

| | |
|---|---|
| proposed version | **1.0.0-rc.2** (rc.1 is tagged locally as `v1.0.0-rc.1`; nothing new is pushed or tagged remotely) |
| tested revision | opengles3-demos **`2ac69b744a4d945834d3a045c9fedd9cf7834cb9`**: every executable in section B was compiled from a clean clone of this commit, and the evidence was collected there |
| release-candidate commit | the commit that adds this report and the rc.2 evidence on top of `2ac69b7`; it changes documentation and evidence files only (no source, shader, CMake file or test registration) |
| meta-gl | **`333d777fadbe53613a9c2eead358ffbfe6a2ef33`** on `develop` (`v0.4.1-9-g333d777`), cloned from <https://github.com/libcna/meta-gl> for the runs. Configure fails without it (`OGLES3_METAGL_REQUIRED_COMMIT`) |
| supported and tested | Linux x86-64 (Debian 13, kernel 6.12); GCC 14.2 and Clang 19.1.7 (C++23, both `-Werror`-clean); CMake 3.31 + Ninja; Mesa 25.0.7: radeonsi (AMD Radeon 780M), llvmpipe, zink on lavapipe, softpipe; EGL 1.5 surfaceless pbuffers; SDL3 3.4.0 windows (X11, tested under Xvfb); ANGLE 2.1.17841 (Android SDK emulator build) on lavapipe |
| expected to work, not tested | other Linux distributions; other conformant ES 3.0–3.2 drivers (deviations would show as FAIL or need a narrowly matched XFAIL); Wayland |
| not supported | desktop OpenGL; ES 2.0-only contexts (ES 2.0 features run on ES 3.x contexts); Windows and macOS (not built or tested; the headless path needs EGL) |

## B. Validation results

### Executed for this candidate (2026-11-12, clean clones of `2ac69b7` and meta-gl `333d777`)

Evidence: [evidence/1.0.0-rc.2/](evidence/README.md) (manifest, per-run results, summary), and the archive in section
E (logs, JUnit reports, every result JSON and capture).

| matrix | compiler, build | registered | PASS | SKIP_UNSUPPORTED | BLOCKED_ENVIRONMENT | FAIL / CRASHED | run |
|---|---|---|---|---|---|---|---|
| standard (`debug`) | GCC 14.2, 0 warnings | 701 | 674 | 26 | 0 | **0** | 19:59–20:05 |
| sanitizer (`sanitize`, ASan + UBSan, full matrix incl. ANGLE) | GCC 14.2, 0 warnings | 701 | 674 | 26 | 0 | **0** | 20:38–20:50 |
| sanitizer, first attempt (kept, see below) | same build | 701 | 673 | 26 | 0 | 1 CRASHED (timeout) | 20:13–20:27 |
| strict (`headless-ci`, `-Werror`) | GCC 14.2, **0 warnings** | 395 | 365 | 26 | 3 | **0** | 20:30–20:32 |
| strict (`headless-ci-clang`, `-Werror`) | Clang 19.1.7, **0 warnings** | 395 | 365 | 26 | 3 | **0** | 20:36–20:37 |

Each registered count includes the CPU-only `unit-deviation-matcher` (PASS, no result JSON). Breakdown of the
standard matrix by driver and mode (from `results.csv`):

| driver | mode | PASS | SKIP_UNSUPPORTED |
|---|---|---|---|
| gpu (radeonsi) | es-max (ES 3.2) / es-min (override) | 83 / 70 | 0 |
| llvmpipe | es-max / es-min / 333x777 | 83 / 70 / 83 | 0 |
| zink (lavapipe) | es-max / es-min | 83 / 70 | 0 |
| softpipe (native ES 3.1) | es-max / es-min | 4 / 4 | 13 (ES 3.2 executables) |
| ANGLE on lavapipe | native (ES 3.1) / native-min (ES 3.0) | 70 / 54 | 13 (ES 3.2 executables) |

- **The sanitizer timeout:** in the first sanitizer run, `es30-deferred-shading:llvmpipe:size-333x777` exceeded
  CTest's 120 s limit while another session's ASan suite kept the load average near 30. All nine deferred-shading
  runs were 6–7 times slower than in rc.1. Run back to back under that load, the rc.1 binary (built at `9050c5a`) and
  the rc.2 binary need the same CPU time (106.6 / 111.5 s and 109.0 / 120.9 s), and both PASS 79/79. The same build
  was then collected again at load 12–20: PASS in 74 s (26 s on the idle machine of rc.1). Both runs are kept in
  `docs/evidence/1.0.0-rc.2/` (`sanitize-attempt1`, `sanitize`); the timeout is a load artifact of a slow ASan +
  llvmpipe run, not a regression, but it shows that this one run has little margin under heavy load.
- **Every SKIP_UNSUPPORTED** is an ES 3.2 executable on an ES 3.1 implementation (softpipe, ANGLE).
- **BLOCKED_ENVIRONMENT** occurs only in the two `headless-ci` matrices, which build without SDL3 by design: there
  `es30-multisample-default-framebuffer` cannot use its window runs and reports the Mesa surfaceless-pbuffer limit
  on llvmpipe (es-max, es-min, 333x777). In the standard matrix those runs pass in an Xvfb window.
- **Known driver deviations:** 650 checks in 202 runs are matched XFAILs (as in rc.1), each listed with its spec
  quote in KNOWN_LIMITATIONS.md and per run in [generated/DEVIATIONS.md](generated/DEVIATIONS.md).
- **Total passed checks** in the standard matrix: 73,700 (rc.1: 73,511). The difference, 189, is exactly the checks
  of the five former BLOCKED runs (39 + 36 + 39 + 39 + 36).
- **Clean build:** both repositories freshly cloned (meta-gl from GitHub); all four presets configured and built
  without a warning; `cmake --build --preset debug --target coverage` on that tree: 83 FULLY_VALIDATED, 700 of 700
  registered GL runs with a fresh result.
- **GPU:** every manifest records the kernel log as read, with 0 amdgpu fault, timeout or reset lines during the
  runs. `OGLES3_RUN_GPU_FAULT_CASES` was not set.

### Additional checks of this completion phase (working tree, 2026-11-12)

- **Clang fixes:** object code of 88 of 92 objects identical (`.text`, `.rodata`, Clang -O0); deferred-shading and
  normal-mapping identical at -O2; the llvmpipe runs of all 30 changed executables give identical outcomes, check
  texts and measured details (83 of 84 runs; the 84th differs only in the deliberately unasserted
  `atomicCompSwap` winner) and byte-identical captures (54 of 54).
- **Multisample default framebuffer in a window:** llvmpipe and zink, ES 3.2 and ES 3.0, 256x256 and 333x777: PASS
  with `SAMPLE_BUFFERS` 1 and `SAMPLES` 4; captures viewed.
- **3D demos and window resizing:** 14 3D executables in SDL3 windows under Xvfb (llvmpipe), resized from 640x400
  to 500x720 while running; validation passes at the new size (after the runtime fix in `b259bf8`), and the captures
  keep their aspect ratio. Window-size sweep (`tools/sweep_sizes.sh`, 333x777, 1000x200, 200x1000, 500x720,
  640x400): every executable PASS or SKIP_UNSUPPORTED (the headless multisample case aside).
- **`--frames N`:** all executables at `--frames 300`: 80 pass; es20-transformations-camera, es30-shadow-mapping
  and es30-sync-objects validate only their documented frame (section D).

## C. Coverage

From [generated/COVERAGE_SUMMARY.md](generated/COVERAGE_SUMMARY.md) (source revision `2ac69b7`, meta-gl `333d777`):

| | total | FULLY_VALIDATED | other |
|---|---|---|---|
| core API entry points (gl.xml, ES 2.0/3.0/3.1/3.2 = 142/104/68/44) | 358 | 358 | — |
| semantic features (ES 2.0/3.0/3.1/3.2 = 87/78/30/31) | 226 | 225 | 1 DOCUMENTED_ONLY (ES20.FBO.FEEDBACK_LOOP: undefined results) |
| GLSL ES items (1.00/3.00/3.10/3.20 = 4/51/23/16) | 94 | 93 | 1 DOCUMENTED_ONLY (GLSL300.UNDEFINED_BEHAVIOR) |
| executables | 83 | 83 | — |

FULLY_VALIDATED means every registered run is a validated PASS or a justified SKIP_UNSUPPORTED, with gpu and
llvmpipe passing at the minimum and maximum version (VALIDATION.md). An API entry point counts only where a manifest
says how its effect is verified, and a lint checks that the executable calls it.

## D. Limitations and open items

| kind | item |
|---|---|
| external driver defect | Mesa 25.0.7's surfaceless EGL gives llvmpipe and zink a 0x0 multisampled pbuffer. Worked around for the one demo that needs it by running it in an X11 window; without SDL3/`xvfb-run` it reports BLOCKED_ENVIRONMENT. Other driver deviations: 650 XFAIL checks, KNOWN_LIMITATIONS.md |
| unavailable environment | every tested implementation is Mesa or ANGLE on Mesa's lavapipe; ANGLE adds an independent front-end and compiler, not an independent rasterizer. ES 3.2 has Mesa evidence only. No other vendor's driver was available; no cross-vendor claim is made |
| unverified hardware | ANGLE on a hardware Vulkan driver is not run by default (GPU fault of 2026-11-11, section F); other GPUs untested |
| project limitation (minor) | es20-transformations-camera, es30-shadow-mapping and es30-sync-objects validate only their documented frame: with an explicit `--frames N --validate` they can FAIL on a correct render. The matrix never sets `--frames` |
| project limitation | Mesa's `es-min` lowers only the reported version; native lower-version evidence is softpipe (ES 3.1) and ANGLE (ES 3.1, ES 3.0) |
| owner decision | meta-gl: the AI-authored G1–G8 commits on `develop` versus meta-gl's AI Contribution Policy, and a meta-gl release tag ([METAGL_MAINTAINER_NOTE.md](METAGL_MAINTAINER_NOTE.md)) |
| owner decision | publishing: push, tag `v1.0.0`, release with the evidence archive |

Undefined behaviour (feedback loops, the GLSL undefined-behaviour row) is documented, never asserted.

## E. Reproducibility and evidence archive

```sh
mkdir -p root/libcna root/libcnalab && cd root
git clone https://github.com/libcna/meta-gl libcna/meta-gl
git -C libcna/meta-gl checkout 333d777fadbe53613a9c2eead358ffbfe6a2ef33
git clone <opengles3-demos> libcnalab/opengles3-demos && git -C libcnalab/opengles3-demos checkout <RC commit>
cd libcnalab/opengles3-demos
for p in debug sanitize headless-ci headless-ci-clang; do
  cmake --preset $p && cmake --build --preset $p
  python3 -I tools/release/collect_evidence.py --build build/$p --label $p --out ../evidence/$p   # or: ctest --preset $p
done
cmake --build --preset debug --target coverage      # docs/generated/: 83 FULLY_VALIDATED
```

Requirements outside the repositories: Mesa with radeonsi (or another hardware driver), llvmpipe, zink and softpipe
and the lavapipe Vulkan ICD; EGL; SDL3 and `xvfb-run` (for the multisample demo's window runs); Python 3; an ANGLE
build for the `angle` runs. On another machine the `gpu` runs exercise that machine's driver.

Archive: `opengles3-demos-1.0.0-rc.2-evidence.tar.xz` (21,298,164 bytes, 4,119 entries, sha256
`3e7a4522079c0d15472a77fc2c49a2f0ee7f17cdbdcbf65978a1f610c6309d6b`), on the reference machine under
`build/release-evidence/` with its `.sha256` file. It holds, per matrix, the manifest, results, summary, CTest log,
JUnit report, configure and build logs and every result JSON and capture; the coverage output of the clean clone; and
the supplementary material of this phase (Clang object comparison and build logs, the multisample window runs, the
resize runs). The rc.1 archive is unchanged (sha256 `2d96c332…cba7`).

## F. GPU safety

Verified in the source and the manifests:
- The known GPU-faulting case (`es31-image-load-store`, non-layered 3D-slice `imageStore` on ANGLE/RADV) is not run
  on ANGLE unless `OGLES3_RUN_GPU_FAULT_CASES=1`; the variable was not set in any run.
- ANGLE's Vulkan driver defaults to lavapipe (`OGLES3_ANGLE_VULKAN_ICD`). The new zink window runs pin Vulkan to
  the same lavapipe ICD; the llvmpipe window runs are software rendering into a virtual X server.
- Driver crash cases run in child processes with timeouts; CTest kills any run after 120 s and the result is
  `CRASHED`.
- Compilation stays at ≤ 3 jobs (presets, the owner's limit since 2026-11-12).

A clean kernel log is evidence for these runs on this machine, not proof of GPU safety in general.

## G. Decision

Pending the independent review (section H).

## H. Independent review

Pending: the focused review ([agent-briefs/RC2_REVIEW_BRIEF.md](agent-briefs/RC2_REVIEW_BRIEF.md)) by a different agent and model runs on this commit.
