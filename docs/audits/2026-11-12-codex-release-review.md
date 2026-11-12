# Independent release review: 1.0.0-rc.1

Reviewed commit: `4c6d7883f6dff8eb88ab05d4a0407ce5a3514039` (`4c6d788`). Scope followed `docs/agent-briefs/RELEASE_REVIEW_BRIEF.md`, including the stabilization diff, committed evidence, result classification, reproducibility, GPU safety, policy and documentation consistency.

## Findings

### 1. HIGH — Evidence can attribute a different build to the current source revision

**File:** `tools/release/collect_evidence.py:71-86`

`repo` is always derived from the collector script path (`line 71`), while `--build` is accepted independently (`lines 72, 75`). CTest is then run against that supplied build directory (`line 86`). The collector never compares the build cache's `CMAKE_HOME_DIRECTORY` with `repo`. It records `src`, obtained from the script's repository, as `manifest.opengles3_demos` (`line 156`).

**Supporting evidence:** `nl -ba tools/release/collect_evidence.py | sed -n '71,86p;154,162p'` shows that the build's source directory is not checked before CTest runs, while the manifest records the script repository revision. For example, invoking `tools/release/collect_evidence.py --build /path/to/another/checkout/build/debug --label debug --out /tmp/evidence` runs the other checkout's tests but labels them with this checkout's commit.

**Impact:** The evidence tool can produce a clean, successful manifest and CSV whose tested source revision is false. That undermines release evidence provenance when a build path is accidentally stale or points to another checkout.

**Recommended resolution:** Read `CMAKE_HOME_DIRECTORY` from `CMakeCache.txt` and refuse to collect evidence unless it resolves to the same repository as the script (or explicitly record and validate the build source revision).

### 2. HIGH — The clean-tree check ignores untracked inputs

**File:** `tools/release/collect_evidence.py:41-48`

`git_state()` runs `git status --porcelain --untracked-files=no` (`line 47`), so untracked files are treated as a clean tree. This matters for this suite: `shader-tests/main.cpp:91-113` discovers shader cases by iterating the source directories at runtime, so an untracked shader case can change the tested inputs without changing the commit recorded in the manifest.

**Supporting evidence:** `nl -ba tools/release/collect_evidence.py | sed -n '41,48p'` shows the untracked-files exclusion; `nl -ba shader-tests/main.cpp | sed -n '91,113p'` shows runtime discovery of files from the source tree. A newly added untracked case is therefore invisible to the collector's dirty check but visible to the executable.

**Impact:** The collector can label evidence as coming from a clean revision even though uncommitted test inputs affected the run. This makes the “clean clones” and source-revision evidence claims unverifiable for such a run.

**Recommended resolution:** Include untracked files in the dirty check, or explicitly reject untracked source and runtime-input files in both repositories before running the matrix.

### 3. MEDIUM — `kernel_log_checked` is always true, including when `journalctl` fails

**File:** `tools/release/collect_evidence.py:34-38,148-151,176-180`

The `sh()` helper discards subprocess return codes and returns an empty string on `OSError`. The manifest sets `kernel_log_checked` to `bool(klog) or klog == ''`, which is true for every possible string, including the empty string returned when `journalctl` is unavailable or fails without output.

**Supporting evidence:** `nl -ba tools/release/collect_evidence.py | sed -n '34,38p;148,151p;176,180p'`. The boolean expression at line 179 is tautologically true; it does not establish that the kernel log was read successfully.

**Impact:** A manifest can claim the amdgpu kernel log was checked when no log was retrieved, weakening the recorded GPU-safety evidence.

**Recommended resolution:** Preserve and check `journalctl`'s exit status, and record an explicit unavailable/failed state when the query does not succeed. Do not report `kernel_log_checked: true` unless the command completed successfully.

## Checks performed

- Configured the debug preset against `/rv/data/development/github.com/libcna/meta-gl` with `OGLES3_TEST_DRIVERS=llvmpipe`.
- Built `unit-deviation-matcher` with `-j 1`; it reported `RESULT: PASS (0 failure(s))`.
- Built and ran `es30-hello-triangle` with `EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe`; it reported `RESULT: PASS demo=es30-hello-triangle es=3.2 checks=9/9`.
- Reconciled the committed `results.csv` row totals with each evidence manifest and the readiness report: debug/sanitize each have 701 rows (669 PASS, 26 SKIP_UNSUPPORTED, 5 BLOCKED_ENVIRONMENT, 1 CPU-only test); headless-ci/clang19 each have 395 rows (365 PASS, 26 SKIP_UNSUPPORTED, 3 BLOCKED_ENVIRONMENT, 1 CPU-only test).
- Confirmed `2544bad..HEAD` changes no executable or CMake implementation; the committed evidence records `2544bad` as the tested code revision, as described by the readiness report.
- Did not run the hardware GPU or the full CTest matrix. `OGLES3_RUN_GPU_FAULT_CASES` was not set.

## Verdict

There are unresolved release blockers in evidence collection: the collector can associate results from another build with this source revision and can overlook untracked test inputs while claiming a clean tree. Fix those provenance checks before relying on newly collected release evidence. The current committed evidence totals reconcile, and the targeted llvmpipe and CPU-only checks passed; the remaining kernel-log status defect is a separate weakness in the recorded GPU-safety audit.
