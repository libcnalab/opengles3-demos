# Brief: triage of ANGLE failures

The `angle` driver (see docs/VALIDATION.md, section "ANGLE") is the first GL ES implementation in the matrix
whose front-end and GLSL compiler are not Mesa's. Its first run failed checks that pass on Mesa. The list is in
`docs/agent-briefs/angle-triage-input.txt`. Every failure is one of these:

- **A. ANGLE deviates from the normative text.** Keep the strict check and add
  `expect_or_known_deviation(..., "ANGLE", "<reason>")`, matching on GL_RENDERER/GL_VERSION. Quote the spec in
  your report. Before claiming a deviation, check ANGLE's exposed extensions, which can amend the rule.
- **B. The test asserts more than the spec guarantees.** It encoded Mesa behaviour or an
  implementation-dependent result. Fix the test so it asserts only the normative rule; print the observation
  instead. This is the most important category: a conformant implementation must not FAIL.
- **C. Our reading of the spec is wrong.** This needs special attention where ANGLE AND Mesa both "deviate" in the
  same way (the check already has a Mesa XFAIL). Re-read the spec and its errata/changelog. If our reading is
  wrong, fix the check for all drivers, remove the Mesa XFAIL, and say so explicitly.
- **D. The ANGLE build is incomplete** (e.g. a missing feature it does not claim). Treat it as unsupported and
  skip with a reason. Do not count it as a pass.

How to run ANGLE: `ctest --preset debug -R '<exe>:angle'`, or
`env -u DISPLAY OGLES3_EGL_PLATFORM=angle LD_LIBRARY_PATH=build/debug/angle-shim:$HOME/Android/Sdk/emulator/lib64/gles_angle LD_PRELOAD=$HOME/Android/Sdk/emulator/lib64/gles_angle/libGLESv2.so ./build/debug/bin/<exe> --headless --validate`.
After every change, re-run the full Mesa matrix of the touched executables as well. Mesa results must not regress.
Spec text: /tmp/claude-1000/-rv-data-development-github-com-libcnalab-opengles3-demos/e091dc40-e8fe-4d2b-81ea-4b4c6931285b/scratchpad/khronos/. CPU limit: build with -j 2.
Report one line per failing check: category, spec quote, what you changed. Also report the KNOWN_LIMITATIONS
rows to add or remove.
