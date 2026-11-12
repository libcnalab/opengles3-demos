# Release evidence

Machine-readable evidence of the matrix runs behind a release, written by `tools/release/collect_evidence.py`.
Each run directory has:

| file | content |
|---|---|
| `manifest.json` | opengles3-demos and meta-gl commits (clean working trees), build type and every `OGLES3_*` option, compiler, CMake, kernel and Mesa package versions, the driver strings seen, the exact ctest command, start/end time, CTest status and outcome counts, the ANGLE Vulkan ICD, and the amdgpu fault/timeout/reset lines of the kernel log during the run |
| `results.csv` | one row per registered test: CTest status, outcome from the result JSON (`PASS`, `SKIP_UNSUPPORTED`, `BLOCKED_ENVIRONMENT`, ...), reason, context ES version, GLSL version, renderer, GL_VERSION, checks passed/failed, known-deviation (XFAIL) checks, version override |
| `SUMMARY.md` | a digest of both, including every run that did not PASS |

CTest's status `notrun` is its name for exit code 77 (SKIP_UNSUPPORTED or BLOCKED_*). The CPU-only
`unit-deviation-matcher` writes no result JSON ("N/A ... by design").

## 1.0.0-rc.2

Runs in a clean clone at opengles3-demos `ef7b53d` (executables compiled in that clone at `2ac69b7`; `ef7b53d` changes
only test registration and documentation, and reconfiguring compiled and linked nothing) with meta-gl `333d777`
cloned from GitHub, on the reference machine, 2026-11-12:

| run | compiler | registered | PASS | SKIP_UNSUPPORTED | BLOCKED_ENVIRONMENT | FAIL / CRASHED | unit test |
|---|---|---|---|---|---|---|---|
| [debug](1.0.0-rc.2/debug/SUMMARY.md) | GCC 14.2 | 701 | 674 | 26 | 0 | 0 | PASS |
| [sanitize](1.0.0-rc.2/sanitize/SUMMARY.md) (ASan + UBSan) | GCC 14.2 | 701 | 674 | 26 | 0 | 0 | PASS |
| [headless-ci](1.0.0-rc.2/headless-ci/SUMMARY.md) (`-Werror`, no SDL3) | GCC 14.2 | 395 | 365 | 26 | 3 | 0 | PASS |
| [headless-ci-clang](1.0.0-rc.2/headless-ci-clang/SUMMARY.md) (`-Werror`, no SDL3) | Clang 19.1.7 | 395 | 365 | 26 | 3 | 0 | PASS |

All builds compiled without a warning. The 3 BLOCKED_ENVIRONMENT runs of the `headless-ci` presets are
es30-multisample-default-framebuffer on llvmpipe: those presets build without SDL3, so its window runs fall back to
the surfaceless pbuffer that Mesa creates as 0x0.

[superseded-2ac69b7/](1.0.0-rc.2/superseded-2ac69b7/) keeps the collection made before the independent review, at
`2ac69b7` (same builds and counts; its first sanitizer attempt had one timeout under a load average near 30, see
FINAL_RELEASE_READINESS.md section B). The review's only finding (zink ICD pin) changed test registration, so
everything was collected again at `ef7b53d`.

The full logs, JUnit reports, every result JSON and capture, the coverage output, the superseded collection and the
supplementary material are in `opengles3-demos-1.0.0-rc.2-evidence.tar.xz` (sha256
`7bad7492500359fe79153bb1fecd7e39e981d3000d0530feb4db8aaba0d5141c`), kept on the reference machine under
`build/release-evidence/`. The pre-review archive (`...-rc.2-pre-review-2ac69b7-evidence.tar.xz`, sha256
`3e7a4522079c0d15472a77fc2c49a2f0ee7f17cdbdcbf65978a1f610c6309d6b`) is kept next to it.

## 1.0.0-rc.1

Runs from clean clones (opengles3-demos `9050c5a`, executables compiled at `2544bad` with identical sources;
meta-gl `333d777` cloned from GitHub) on the reference machine, 2026-11-12:

| run | registered | PASS | SKIP_UNSUPPORTED | BLOCKED_ENVIRONMENT | FAIL | unit test |
|---|---|---|---|---|---|---|
| [debug](1.0.0-rc.1/debug/SUMMARY.md) | 701 | 669 | 26 | 5 | 0 | PASS |
| [sanitize](1.0.0-rc.1/sanitize/SUMMARY.md) (ASan + UBSan) | 701 | 669 | 26 | 5 | 0 | PASS |
| [headless-ci](1.0.0-rc.1/headless-ci/SUMMARY.md) (`-Werror`, llvmpipe/softpipe/ANGLE) | 395 | 365 | 26 | 3 | 0 | PASS |
| [clang19](1.0.0-rc.1/clang19/SUMMARY.md) (headless-ci with Clang 19.1.7, **without** `-Werror`) | 395 | 365 | 26 | 3 | 0 | PASS |

In rc.1 the Clang build compiled with 189 warnings (178 `-Wdouble-promotion`, 11 unused entities) in 30 files, so the
strict preset failed with Clang (fixed in rc.2).

The full CTest logs, JUnit reports, every result JSON and the debug captures are not in git (6.5 MB compressed).
They are in `opengles3-demos-1.0.0-rc.1-evidence.tar.xz` (sha256
`2d96c3325bbc6c62be8545f135dbe5f01f0e54050f09d7f10045b38b2409cba7`, with the build and configure logs), kept on the reference machine under
`build/release-evidence/`, and meant to be attached to the release when it is published.
