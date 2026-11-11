# Brief for an independent audit (phase 6)

You audit a group of existing demos/tests in `opengles3-demos` **without trusting their authors**. Your output
is a findings report; you change code only where the task says so. Correctness matters more than volume.

## Read first
`AGENTS.md`, `docs/VALIDATION.md`, `docs/agent-briefs/DEMO_BRIEF.md` (the rules every demo had to follow),
`docs/KNOWN_LIMITATIONS.md` (accepted driver deviations), `docs/META_GL_GAPS.md`.
Spec text (greppable): `/tmp/claude-1000/-rv-data-development-github-com-libcnalab-opengles3-demos/e091dc40-e8fe-4d2b-81ea-4b4c6931285b/scratchpad/khronos/`
(`es_spec_3.0.txt`, `es_spec_3.1.txt`, `es_spec_3.2.txt`, `GLSL_ES_Specification_{1.00,3.00,3.10,3.20}.txt`).

## For every demo in your group, check
1. **Spec fidelity.** Every asserted rule exists in the normative text, with the cited section. Flag invented
   requirements, misquoted rules, and wrong section numbers. ES 2.0/3.0 features cite ES 3.0 numbering; 3.1/3.2
   features cite ES 3.2 numbering.
2. **Known deviations.** Every `expect_or_known_deviation` must be a real contradiction of normative text. Check
   that no exposed extension amends the rule. Flag deviations that are really allowed behaviour (the check should
   then be relaxed *to the spec*, or printed), and deviations missing from KNOWN_LIMITATIONS.md.
3. **Vacuity.** Every check can fail. Look for comparisons against values computed from the GPU result itself,
   loops that may run zero times, and tolerances so wide nothing fails. Is there a negative control?
4. **Tolerances.** Each is justified by the spec (quantization, filtering) or explicitly marked as a sanity bound.
5. **Manifest honesty.** `demo.json` `api` / `features` / `glsl_features` claim only what `validate()` verifies.
6. **ES purity.** No desktop-only GL, no extensions standing in for core, correct `#version` per stage.
7. **Size independence.** The layout fills any window and validation never FAILs a valid render at other sizes.
   Spot-check one or two sizes, e.g. `--width 333 --height 777`.

8. **Guaranteed vs implementation-dependent.** Classify every assertion as normatively guaranteed,
   implementation-defined, implementation-dependent, undefined, or merely empirical (holds on Mesa). Only the
   first may be a hard check. Flag every hard check of the others. Examples: multisample resolve as an average
   (see VALIDATION.md "Suite-wide assumptions"), filter weights, rasterization of edge pixels, derivative
   precision. A conformant non-Mesa implementation must not FAIL.
9. **Known-deviation scope.** For each `expect_or_known_deviation`, is the matched substring as narrow as the
   evidence? "Mesa" is justified only if the deviation reproduces on radeonsi, llvmpipe AND zink. Report
   over-broad matches.
10. **Version claims.** `es-min` runs use `MESA_GLES_VERSION_OVERRIDE` and prove only API-version detection, not
    ES 3.0/3.1 conformance (Mesa stays an ES 3.2 driver). Flag any README or demo.json wording that claims more.
11. **Manifest vs code.** Every function in `demo.json` `api` is actually called on a validated path, and its
    effect, return value or state change is checked. A call alone is not coverage.
12. **Unsafe negatives.** Calls known to crash a driver must run only in an isolated child process with a
    timeout. Undefined GPU operations must not run at all.

## How
Build once (`cmake --preset debug -DOGLES3_METAGL_SOURCE_DIR=/rv/data/development/github.com/libcna/meta-gl &&
cmake --build --preset debug -j 2`; ALWAYS `-j 2`: CPU limit). Run individual demos with
`./build/debug/bin/<demo> --headless --validate`. If the GPU is not reachable in your sandbox, use
`LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless`.

## Report
For each demo, list findings as: severity (BUG = wrong assertion/claim, WEAK = vacuous or unjustified, NIT),
file:line, the spec quote that decides it, and the concrete fix. Demos with no findings are listed as "clean"
with one line on what you verified. Do not pad the report.
