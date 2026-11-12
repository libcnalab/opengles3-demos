# Changelog

The project follows [Semantic Versioning](https://semver.org/). Version numbers describe the suite as a whole (its
executables, validation rules and documentation); there is no library API.

## 1.0.0-rc.1 — release candidate (2026-11-12, not tagged)

The first release candidate. It freezes functionality; only defects found during the candidate period are fixed
before 1.0.0. Readiness report with all evidence: [docs/FINAL_RELEASE_READINESS.md](docs/FINAL_RELEASE_READINESS.md).

### What is in it

- **83 executables**, each an independent C++23 program that calls OpenGL ES only through meta-gl:
  76 demo programs (ES 2.0 foundations, ES 3.0, ES 3.1, ES 3.2, and `diagnostics-capabilities`), 6 GL tests
  (GLSL ES 3.00/3.10/3.20 built-in functions, GLSL language rules, meta-gl contract death tests, a driver crash
  case) and the data-driven `glsl-compile-tests`. One CPU-only unit test (`unit-deviation-matcher`).
- **Coverage:** all 358 OpenGL ES 2.0–3.2 core entry points of the Khronos registry with validated evidence;
  226 semantic features (224 validated, 1 visually verified, 1 documented only because its results are undefined);
  94 GLSL ES 1.00–3.20 language items (93 validated, 1 documented only).
- **Validation:** every executable checks observable results (CPU-reference pixels, readback, state queries,
  expected errors). The matrix runs every executable on radeonsi, llvmpipe and zink at the maximum and the minimum
  ES version, on softpipe (native ES 3.1) where that boundary matters, on ANGLE 2.1.17841 over lavapipe (native ES
  3.1, and genuine ES 3.0 contexts for every ES 3.0 executable), and once at 333x777.
- **Documentation:** a README per executable (purpose, API calls, spec sections, validation, pitfalls), a learning
  path ([docs/DEMO_CATALOG.md](docs/DEMO_CATALOG.md)), the validation method, every observed driver deviation with
  its spec quote, and the independent review reports.

### Changes in the stabilization phase (since d75359a)

- Runtime: a `demo::Skip` or `BlockedByMetaGl` raised after a failed check is now reported as FAIL instead of
  SKIP.
- Deviation matching is one function, `demo::deviation_matches`, shared by `Check::expect_or_known_deviation`
  and the shader-test `@xfail` annotations. An empty driver name no longer matches every driver. The new CPU test
  `unit-deviation-matcher` pins the rules, including that no Mesa or backend name can excuse ANGLE.
- The coverage generator checks completeness against the CTest registration: a registered run without a fresh
  result prevents FULLY_VALIDATED, and results of tests that are no longer registered are ignored. The summary
  records the meta-gl revision.
- Configure fails with a clear message when the meta-gl checkout lacks the required commit `333d777`.
- `tools/release/collect_evidence.py` records machine-readable evidence of a matrix run (revisions, configuration,
  compiler, drivers, per-run outcomes, kernel-log check). It refuses a build directory from another checkout and
  trees with uncommitted or untracked files (fixes from the independent release review). The release-candidate
  evidence is in `docs/evidence/`.
- The sanitizer preset passes in full: ANGLE runs preload the ASan runtime first (before, every ANGLE run aborted,
  and the earlier "480/480" covered the Mesa runs only).
- Documentation synchronized with the repository (README entry point, VALIDATION, KNOWN_LIMITATIONS,
  METAGL_DEPENDENCY, STATUS, handoff); a short learning path in the catalog. AGENTS.md: agents no longer modify
  meta-gl.

### Requirements

- meta-gl `develop` at `333d777fadbe53613a9c2eead358ffbfe6a2ef33` (tested). meta-gl `main` / `v0.4.1` is too old.
- Linux, CMake ≥ 3.23, Ninja, GCC 14 (C++23), EGL and an OpenGL ES 3.x driver. Optional: SDL3, Python 3, ANGLE.
  Clang 19 builds and passes the headless matrix, with warnings (no `-Werror` build with Clang yet).

### Known limitations

See [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) and the readiness report, section D. In short:
`es30-multisample-default-framebuffer` is VISUALLY_VERIFIED only (Mesa's software EGL path gives a 0x0 multisampled
pbuffer); all tested implementations are Mesa or ANGLE on Mesa's lavapipe; ES 3.2 has Mesa evidence only; one
ANGLE/RADV case is a known GPU fault and runs only on explicit opt-in; meta-gl's AI Contribution Policy and the
AI-authored G1–G8 commits on its `develop` need a decision by its maintainer.

## Development history before the release candidate (2026-11-11 – 2026-11-12)

Not released. About 290 commits in two days, in phases recorded in `docs/AGENT_HANDOFF.md` and `docs/audits/`:
bootstrap (CMake, `common/` runtime, registry-derived inventory); meta-gl audit and gap fixes G1–G8 (on meta-gl,
owner-authorized); the demo backlog; the CTest driver matrix with softpipe and ANGLE; four independent Codex review
rounds (audit, educational review, two diff reviews) with their findings fixed or rejected with reasons; window-size
robustness (every demo fills any window); native ES 3.0 evidence through ANGLE.
