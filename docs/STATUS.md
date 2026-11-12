# Project status

**1.0.0-rc.1 prepared (2026-11-12); functionality frozen.** The authoritative report is
[FINAL_RELEASE_READINESS.md](FINAL_RELEASE_READINESS.md): release identity, the evidence behind every number,
the final audit, reproducibility commands, GPU safety and the release decision. Live coverage numbers:
[generated/COVERAGE_SUMMARY.md](generated/COVERAGE_SUMMARY.md). Release history: [../CHANGELOG.md](../CHANGELOG.md).

In one paragraph: 83 executables (82 FULLY_VALIDATED, 1 VISUALLY_VERIFIED) cover all 358 OpenGL ES 2.0–3.2 core
entry points, 226 semantic features and 94 GLSL ES items. The evidence is recorded and machine-readable
(`docs/evidence/1.0.0-rc.1/`): the full debug matrix (radeonsi, llvmpipe, zink, softpipe, ANGLE on lavapipe at
native ES 3.1 and native ES 3.0, and 333x777), the full sanitizer matrix and the `-Werror` build, all from clean
clones at the same revision. It is not Khronos conformance, and every tested implementation is Mesa or ANGLE on
Mesa's lavapipe.

## Earlier status reports

The evidence-based report of 2026-11-12 02:00 (acceptance criteria of the continuation directive, "700/700",
"480/480", "340/340") is in git history (`git show d75359a:docs/STATUS.md`). Two of its numbers need a reading
today: "700/700" counted CTest's "no failure", which includes 31 skipped runs (26 SKIP_UNSUPPORTED below `min_es`,
5 BLOCKED_ENVIRONMENT); "480/480" was a Mesa-only subset of the sanitizer preset. A plain
`ctest --preset sanitize` failed every ANGLE run until 2544bad.
