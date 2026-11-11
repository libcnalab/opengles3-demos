# Brief: independent review of post-audit changes

After the phase 6 audit, several agents changed demos: precision checks became observations, window-size layouts
and probes were rewritten, and thresholds were re-derived. Each agent tested its own work. You are the independent
reviewer; the code author did not review it.

Review `git diff 5bd59a6..HEAD -- <your directories>` (5bd59a6 is the state the audit saw). For each changed demo,
check:
1. **No check was silently loosened.** A tolerance widened, a threshold lowered, a hard check turned into a print, or
   probes excluded: each needs a spec justification (normative text, or "implementation-dependent/undefined" with a
   quote) written in the code or README. Flag any change without one.
2. **New thresholds and minimum counts are derived** (from predicted geometry, cell sizes, spec bounds), not tuned to
   pass. Look for magic numbers that just clear the measured value.
3. **New layouts keep the checks non-vacuous** at small sizes; the Skip minimums are honest; the README states them.
4. **Negative controls still exist and still bite.** Run the demo and read its output.
5. **README and demo.json match the new behaviour.**

Run demos only on llvmpipe (`EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe`), never on the
GPU. Build with `-j 1`. Write `./DIFF_REVIEW.md`: per demo "ok" with one line, or findings (BUG / WEAK) with file:line
and the concrete issue. Do not modify other files and do not commit.
