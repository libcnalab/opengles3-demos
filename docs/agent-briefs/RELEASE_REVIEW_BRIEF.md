# Release review brief (1.0.0-rc.1)

You are the independent final reviewer of the opengles3-demos release candidate. You did not write it. Your job is
to find **release-blocking problems that you can demonstrate**, not to restart earlier audits or to suggest style
changes. Earlier review rounds and their dispositions are in `docs/audits/`; do not re-report what they settled
unless you show that it is still wrong.

## Read first

`AGENTS.md`, `README.md`, `docs/FINAL_RELEASE_READINESS.md`, `docs/VALIDATION.md`, `docs/KNOWN_LIMITATIONS.md`,
`docs/METAGL_DEPENDENCY.md`, `CHANGELOG.md`. The stabilization changes are `git diff d75359a..HEAD`.

## Focus (in this order)

1. **Correctness of the stabilization diff** (`common/`, `shader-tests/main.cpp`, `tests/unit/`,
   `tools/coverage/generate_coverage.py`, `tools/release/collect_evidence.py`, `CMakeLists.txt`): can a failing run
   now pass, a missing or stale run count as evidence, a deviation match the wrong driver, or the meta-gl check accept
   a revision without the required commit?
2. **Evidence integrity:** do the numbers in `FINAL_RELEASE_READINESS.md`, `README.md`, `CHANGELOG.md` and
   `docs/generated/COVERAGE_SUMMARY.md` agree with the machine-readable evidence in `docs/evidence/` (manifest.json,
   results.csv)? Do the recorded revisions match the git history? Is anything claimed that no evidence supports?
3. **Result classification:** a test must not PASS without meaningful checks; failures, crashes and timeouts must be
   FAIL/CRASHED; SKIP only for a real capability or environment reason; XFAILs narrowly matched (spot-check a few
   `expect_or_known_deviation` call sites against `docs/KNOWN_LIMITATIONS.md`).
4. **Reproducibility:** can a developer follow README/VALIDATION from a clean checkout? Hardcoded absolute paths,
   undocumented requirements, wrong commands?
5. **GPU safety:** can the default matrix send a known GPU-faulting case to a hardware GPU? Is the
   `OGLES3_RUN_GPU_FAULT_CASES` opt-in intact? Does ANGLE default to lavapipe?
6. **Policy:** no change to meta-gl, easy-gl, CNA or cna-engine in this release work; no native `gl*` calls; no
   extensions standing in for core features; documentation does not claim Khronos conformance.
7. **Documentation consistency:** statements that contradict the code or each other (versions, counts, matrix,
   commands). Cosmetic wording is out of scope.

## Rules

- **CPU limit: build with `-j 1`.** **Never run anything on the hardware GPU:** use
  `EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe` for any run, never set
  `OGLES3_RUN_GPU_FAULT_CASES`, never point ANGLE at a hardware Vulkan driver. Running the full CTest matrix is not
  needed; the evidence is recorded. Targeted llvmpipe runs of single executables are fine.
- meta-gl is at `/rv/data/development/github.com/libcna/meta-gl` (configure with
  `-DOGLES3_METAGL_SOURCE_DIR=<that path>`). Do not modify it or any repository other than your worktree.
- Write your report to `./RELEASE_REVIEW.md` in the worktree; modify nothing else; do not commit.
- **At most 15 findings**, high-confidence only. Each finding must have:
  - severity: BLOCKER, HIGH, MEDIUM, LOW or INFORMATIONAL (BLOCKER/HIGH only for a demonstrated defect that makes
    a release claim false, hides failures, breaks the build from a clean checkout, or risks the GPU);
  - exact file and line;
  - the technical explanation;
  - reproduction or supporting evidence (a command and its output, or quoted lines);
  - the impact;
  - a recommended resolution.
- End with a one-paragraph verdict: are there unresolved release blockers?
