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

## 1.0.0-rc.1

Runs from clean clones (opengles3-demos `2544bad`, meta-gl `333d777` cloned from GitHub) on the reference machine,
2026-11-12:

| run | registered | PASS | SKIP_UNSUPPORTED | BLOCKED_ENVIRONMENT | FAIL | unit test |
|---|---|---|---|---|---|---|
| [debug](1.0.0-rc.1/debug/SUMMARY.md) | 701 | 669 | 26 | 5 | 0 | PASS |
| [sanitize](1.0.0-rc.1/sanitize/SUMMARY.md) (ASan + UBSan) | 701 | 669 | 26 | 5 | 0 | PASS |
| [headless-ci](1.0.0-rc.1/headless-ci/SUMMARY.md) (`-Werror`, llvmpipe/softpipe/ANGLE) | 395 | 365 | 26 | 3 | 0 | PASS |
| [clang19](1.0.0-rc.1/clang19/SUMMARY.md) (headless-ci with Clang 19.1.7, **without** `-Werror`) | 395 | 365 | 26 | 3 | 0 | PASS |

The Clang build compiles with 189 warnings (178 `-Wdouble-promotion`, 11 unused entities) in 30 files, so the
strict preset fails with Clang; see FINAL_RELEASE_READINESS.md section D.

The full CTest logs, JUnit reports, every result JSON and the debug captures are not in git (6.5 MB compressed).
They are in `opengles3-demos-1.0.0-rc.1-evidence.tar.xz` (sha256
`aad63211ad3b7d1627f6758b29968f2cb222ad75c38ba0182a944638fe102f69`, with the build and configure logs), kept on the reference machine under
`build/release-evidence/`, and meant to be attached to the release when it is published.
