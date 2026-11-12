# Final completion report: opengles3-demos 1.0.0-rc.2

Prepared 2026-11-12 under the final completion directive, starting from release candidate 1.0.0-rc.1
(`4e0c16c`, tag `v1.0.0-rc.1`). Evidence and full numbers: [FINAL_RELEASE_READINESS.md](FINAL_RELEASE_READINESS.md).

## Starting state (verified, not assumed)

- Git: `main` at `4e0c16c`, equal to `origin/main`, clean; local annotated tag `v1.0.0-rc.1`; no other branches or
  worktrees.
- 83 executables (82 FULLY_VALIDATED, 1 VISUALLY_VERIFIED), 358/358 API entry points, 226 features, 94 GLSL items:
  confirmed from `docs/generated/` and the rc.1 manifests.
- rc.1 evidence archive `build/release-evidence/opengles3-demos-1.0.0-rc.1-evidence.tar.xz`: present, sha256
  `2d96c3325bbc6c62be8545f135dbe5f01f0e54050f09d7f10045b38b2409cba7` matches the documented value; it contains the
  manifests, results, CTest logs, JUnit reports, build and configure logs of the four rc.1 matrices, 2192 result JSON
  files and 587 captures. It was not modified.
- Clang 19.1.7: the rc.1 numbers were reproduced exactly on a fresh build: 189 warnings (178 `-Wdouble-promotion`,
  7 `-Wunused-const-variable`, 3 `-Wunused-function`, 1 `-Wunused-but-set-variable`).

## Completed work

| commit | change |
|---|---|
| `0149f5d` | Clang 19 warnings fixed; `headless-ci-clang` preset |
| `bf74123` | es30-multisample-default-framebuffer validated on llvmpipe and zink in an Xvfb window (`window_drivers`) |
| `4bdad0b` | `shader-tests/README.md`; ten-step learning path in the catalog; note on audit-report paths |
| `5bacc8d` | compilation limit lowered to 3 cores (owner's instruction during this phase) |
| `b259bf8` | runtime: swap before a validated frame that follows a resize; es30-lighting terminator band and minimum size |
| `2ac69b7` | technical note for the meta-gl maintainer |
| RC commit | rc.2 evidence, regenerated coverage, CHANGELOG, README, STATUS, handoff, readiness and this report |

## Clang

- **Before:** 189 warnings in 30 files; `-Werror` build failed.
- **After:** **0 warnings**; `headless-ci-clang` (`-Werror`) builds from a clean clone and its matrix passes
  (365 PASS, 26 SKIP_UNSUPPORTED, 3 BLOCKED_ENVIRONMENT by design without SDL3, 0 FAIL). GCC `headless-ci`
  (`-Werror`) also builds with 0 warnings.
- **Method:** each `-Wdouble-promotion` site was an intended, lossless float-to-double widening into a
  double-precision CPU reference. The fix inserts `static_cast<double>` over exactly the expression range Clang
  reports, which makes the existing implicit conversion explicit and cannot change a value. No arithmetic moved
  between float and double (for example, `static_cast<double>(b.px - a.px)` keeps the float subtraction). A cast per
  component was replaced by small helpers where it hurt readability (`position_of`, `normal_of`, `d3`, local
  `px/py/pz`). Unused entities: dead helpers and constants removed; the shape ids that document the order in
  es30-interactive-3d-scene kept as `[[maybe_unused]]`. No warning was suppressed globally and no flag was removed.
- **Verification of unchanged semantics:** `.text` and `.rodata` identical for 88 of 92 objects (Clang -O0);
  deferred-shading and normal-mapping identical at -O2; es32-astc-and-copy-image differs by one scheduled instruction
  (the removed dead counter); es32-tessellation-terrain differs only in the register allocation of
  `check_geometry`. The pre-fix revision, built with Clang in a scratch worktree, and the fixed revision give
  identical outcomes, check texts and measured values in 83 of 84 llvmpipe runs of the 30 changed executables (the
  84th differs only in the deliberately unasserted `atomicCompSwap` winner), and byte-identical captures (54/54).

## Multisample default framebuffer

- **Observed:** Mesa 25.0.7's surfaceless EGL platform gives llvmpipe and zink a 0x0 default framebuffer for a
  4-sample pbuffer config; `glReadPixels` raises INVALID_OPERATION; the runtime reports BLOCKED_ENVIRONMENT
  (reproduced). radeonsi and ANGLE on lavapipe work headless.
- **Investigated:** an SDL3 window on X11 (Xvfb). llvmpipe and zink create a correct multisampled window surface
  there (`SAMPLE_BUFFERS` 1, `SAMPLES` 4, sample positions reported). The unchanged demo passes: 39/39 (ES 3.2),
  36/36 (ES 3.0 override), 39/39 at 333x777, on both drivers. The checks that only hold for a multisampled default
  framebuffer pass (resolve with offset, scale or one-sided mirror → INVALID_OPERATION; blit into the window →
  INVALID_OPERATION; per-sample stencil and occlusion counts), and llvmpipe's blit resolve blends edge pixels. This
  confines the defect to Mesa's surfaceless multisampled-pbuffer path.
- **Observed and recorded, not asserted:** llvmpipe's `glReadPixels` of the window does not average samples (its
  edges are stair-stepped in the capture) while its blit resolve does; ES 3.0 §4.1.10 leaves the combination method
  unspecified, and the demo has only printed resolved colors since the earlier audits.
- **Implemented:** the matrix runs this demo (only) on llvmpipe and zink with `--window` under `xvfb-run -a`
  (`"window_drivers"` in `demo.json`), with the same test names and checks, a `window` label, a resource lock, and
  zink pinned to lavapipe. A window is a default framebuffer, so no framebuffer object stands in. Without SDL3 or
  `xvfb-run` the runs fall back to headless and stay BLOCKED_ENVIRONMENT (as in both `headless-ci` matrices).
- **Status:** FULLY_VALIDATED (all 9 registered runs PASS in the standard matrix of the clean clone).

## Other defects found and fixed during regression testing

- **Window resize race (runtime):** when a resize event arrived in the validated frame, Mesa's X11 back buffer still
  had the old size, so validation read pixels outside it (es30-skybox-and-fog, 6 of 6 runs). Validation runs now
  swap once before drawing such a frame; 4 of 4 runs pass afterwards.
- **es30-lighting at 500x720 (all drivers):** a sweep sample fell on a light's terminator, where the shading has a
  step. The sweep now skips sphere pixels with |N.L| < 0.01, which is more than three times the documented 0.003 rad
  normal error; the tolerance is unchanged. Below 200 px the demo now reports SKIP_UNSUPPORTED instead of vacuous
  FAILs (the brief's rule); 200x200 to 1024x768 pass on llvmpipe and radeonsi.

## 3D demonstrations

14 3D executables (transformations-camera, cube-map, lighting, normal-mapping, shadow-mapping, skybox-and-fog,
deferred-shading, post-processing, interactive-3d-scene, compute-particles, geometry-shaders, layered-rendering,
tessellation-terrain, cube-map-array) ran in SDL3 windows under Xvfb (llvmpipe) and were resized from 640x400 to
500x720 while running. Validation passes at the new size, and the captures keep their aspect ratio (viewed). All
of them pass the matrix (perspective and view transformations, shaders, results) and the size sweep. Every
run checks that `shutdown()` raises no GL error, and the sanitizer matrix checks CPU-side memory errors. GPU resource
leaks are not measured: ASan runs with `detect_leaks=0` because driver-internal allocations are outside the suite's
control.

## Test evidence (clean clones of `2ac69b7`)

| matrix | registered | PASS | SKIP_UNSUPPORTED | BLOCKED_ENVIRONMENT | FAIL / CRASHED |
|---|---|---|---|---|---|
| debug (radeonsi, llvmpipe, zink, softpipe, ANGLE native ES 3.1 and ES 3.0, 333x777) | 701 | 674 | 26 | 0 | 0 |
| sanitize (ASan + UBSan) | 701 | 674 | 26 | 0 | 0 |
| sanitize, first attempt (load average ≈ 30) | 701 | 673 | 26 | 0 | 1 timeout |
| headless-ci (GCC, `-Werror`) | 395 | 365 | 26 | 3 | 0 |
| headless-ci-clang (Clang, `-Werror`) | 395 | 365 | 26 | 3 | 0 |

Counts include the CPU unit test. SKIP_UNSUPPORTED: ES 3.2 executables on ES 3.1 implementations (softpipe, ANGLE).
BLOCKED_ENVIRONMENT: the window-less fallback described above. The first sanitizer attempt's timeout is explained in
FINAL_RELEASE_READINESS.md section B (same CPU cost as the rc.1 binary under the same load; PASS in 74 s when
re-collected). Kernel log: no amdgpu fault, timeout or reset in any run. XFAIL: 650 checks in 202 runs, unchanged
from rc.1.

## Coverage

358/358 core API entry points FULLY_VALIDATED; 226 semantic features (225 FULLY_VALIDATED, 1 DOCUMENTED_ONLY:
undefined feedback-loop results); 94 GLSL ES items (93 FULLY_VALIDATED, 1 DOCUMENTED_ONLY: undefined behaviour);
**83 of 83 executables FULLY_VALIDATED**; 137 GLSL compile/link cases.

## Dependencies

meta-gl `333d777fadbe53613a9c2eead358ffbfe6a2ef33` (`develop`; publicly reachable on GitHub; in no tag). No meta-gl
change was made in this phase. Outstanding owner decisions are summarized in
[METAGL_MAINTAINER_NOTE.md](METAGL_MAINTAINER_NOTE.md): keep the AI-authored G1–G8 commits under a recorded policy
exception, or have them re-implemented by a human; and tag a meta-gl release the suite can pin. The local meta-gl
checkout shows `AGENTS.md` and `CLAUDE.md` deleted in its working tree (not done in this phase, left untouched).

## Release candidate

- Version **1.0.0-rc.2**; code revision under test `2ac69b7`; the RC commit (documentation and evidence only) is the
  commit that adds this report. A local tag `v1.0.0-rc.2` marks it; nothing was pushed.
- Evidence: `docs/evidence/1.0.0-rc.2/`; archive `opengles3-demos-1.0.0-rc.2-evidence.tar.xz`, sha256
  `3e7a4522079c0d15472a77fc2c49a2f0ee7f17cdbdcbf65978a1f610c6309d6b` (`build/release-evidence/`).

## Remaining limitations

| class | item |
|---|---|
| real project defects | none open that affect a release claim. Minor: three executables validate only their documented frame (`--frames N --validate` can FAIL on a correct render; the matrix never uses it). One ASan + llvmpipe run has little timeout margin under heavy machine load |
| external driver defects | Mesa surfaceless multisampled pbuffer (0x0) on llvmpipe/zink; the 650 XFAIL checks of KNOWN_LIMITATIONS.md |
| unavailable environments | no non-Mesa ES 3.2 driver; ANGLE only on lavapipe; no Windows/macOS |
| unverified hardware | GPUs other than the AMD Radeon 780M; ANGLE on hardware Vulkan (not run by default for GPU safety) |
| policy decisions for the owner | meta-gl G1–G8 policy status and release tag; publishing 1.0.0 |

## Independent review

Pending: the focused independent review ([agent-briefs/RC2_REVIEW_BRIEF.md](agent-briefs/RC2_REVIEW_BRIEF.md)) runs on this commit.

## Final decision

Pending the independent review.
