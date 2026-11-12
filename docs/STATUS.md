# Project status

**1.0.0-rc.2 prepared (2026-11-12); functionality frozen; awaiting the owner's release decision.** The authoritative
reports are [FINAL_RELEASE_READINESS.md](FINAL_RELEASE_READINESS.md) (release identity, evidence, audit, GPU safety,
decision) and [FINAL_COMPLETION_REPORT.md](FINAL_COMPLETION_REPORT.md) (what the completion phase changed and why).
Live coverage numbers: [generated/COVERAGE_SUMMARY.md](generated/COVERAGE_SUMMARY.md). Release history:
[../CHANGELOG.md](../CHANGELOG.md).

In one paragraph: 83 executables, all FULLY_VALIDATED, cover all 358 OpenGL ES 2.0–3.2 core entry points, 226
semantic features (225 validated, 1 documented only because its results are undefined) and 94 GLSL ES items (93
validated, 1 documented only). GCC 14.2 and Clang 19.1.7 build every strict configuration without warnings. The
evidence is machine-readable (`docs/evidence/1.0.0-rc.2/`) and was collected from clean clones of the candidate
source revision: the standard matrix (radeonsi, llvmpipe, zink, softpipe, ANGLE on lavapipe at native ES 3.1 and
native ES 3.0, and 333x777), the sanitizer matrix and both `-Werror` builds. It is not Khronos conformance, and
every tested implementation is Mesa or ANGLE on Mesa's lavapipe.

## Earlier status reports

rc.1 (82 FULLY_VALIDATED + 1 VISUALLY_VERIFIED, Clang with 189 warnings): `git show v1.0.0-rc.1:docs/STATUS.md` and
the rc.1 evidence in `docs/evidence/1.0.0-rc.1/`. The report of 2026-11-12 02:00 ("700/700", "480/480", "340/340")
is in `git show d75359a:docs/STATUS.md`; "700/700" counted CTest's "no failure", which included 31 skipped runs, and
"480/480" was a Mesa-only subset of the sanitizer preset.
