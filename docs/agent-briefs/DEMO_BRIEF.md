# Brief for an agent implementing demos (Claude subagents and Codex alike)

You implement one or two demo executables in `opengles3-demos`, in your own git worktree/branch.
The integrating agent reviews, re-runs the full matrix, views captures, and merges. Correctness
matters more than speed.

## Read first
- `AGENTS.md` (rules), `docs/VALIDATION.md` (what PASS means), `docs/META_GL_GAPS.md`.
- Templates (copy their structure):
  - `demos/es30/hello-triangle/`: minimal structure and a CPU barycentric reference.
  - `demos/es20-foundations/shader-compilation/`: state queries and expected GL errors.
  - `demos/es20-foundations/textures-2d/`: CPU sampler reference, a grid of cells, probe grids.
  - `demos/es20-foundations/uniforms/`: negative control (prove a check can fail).
  - `demos/es20-foundations/blending/`: CPU model of fixed-function math, tolerance justification.
- Runtime API: `common/include/demo/{runtime,check,gl_util,math}.hpp`. Do not modify `common/`.
- meta-gl API, the only way to call GL. Verify every signature and enum there and never guess:
  `/rv/data/development/github.com/libcna/meta-gl/include/metagl/{Functions,Enums}.hpp`.
- Spec text, greppable: `/tmp/claude-1000/-rv-data-development-github-com-libcnalab-opengles3-demos/e091dc40-e8fe-4d2b-81ea-4b4c6931285b/scratchpad/khronos/`
  (`es_spec_3.0.txt`, `es_spec_3.1.txt`, `es_spec_3.2.txt`, `GLSL_ES_Specification_3.{00,10,20}.txt`).
  If that directory is gone, fetch the PDFs listed in `docs/references/README.md` and run `pdftotext -layout`.
  Cite ES 3.0 section numbers for ES 2.0/3.0 features and ES 3.2 numbering for 3.1/3.2 features.
  Verify every rule you test against this text.

## Deliverables per demo (`demos/<group>/<dir>/`, executable `<group-prefix>-<dir>`)
- `CMakeLists.txt` containing just `ogles_add_demo()`.
- `demo.json` in the template schema:
  - `api` lists only entry points whose *effect is verified*, each with how it is verified.
  - `features` and `glsl_features` use only IDs present in `inventory/*.csv`. Report missing IDs; do not edit the inventory.
  - Do not overclaim: if `validate()` does not check a feature, do not list it.
- `main.cpp` implementing `demo::Demo`, with `main()` and `demo::Info{... .source_dir = DEMO_SOURCE_DIR}`.
  GL calls stay visible; repetition is fine. Clean up every object in `shutdown()`.
- `shaders/` with the correct `#version` (300 es for ES 3.0 demos).
- `README.md` in real prose: purpose, minimum version, central calls, spec sections, shader concepts,
  expected image, validation method, pitfalls.

## Validation rules
- Compare observable results with CPU-computed expectations. Justify every tolerance with the spec
  (8-bit quantization ±1, filter weights, ...). Probe away from primitive edges and texel boundaries.
- Include at least one **negative control** or mutation argument showing that a check can fail.
- Use defined error cases (`check.expect_gl_error`) where the spec defines them.
- Never loosen a check to make a driver pass, and never "accept either outcome". If a driver
  contradicts the normative text, keep the strict check and use
  `check.expect_or_known_deviation(ok, what, detail, "<GL_RENDERER substring>", "<reason>")`. Quote the
  spec in your report, and the integrator adds the row to docs/KNOWN_LIMITATIONS.md.
- If meta-gl cannot express something, do NOT work around it with raw casts. Report the gap (exact
  enum or function, spec evidence) and skip that check.

## Build and test (inside your worktree)
```
cmake --preset debug -DOGLES3_METAGL_SOURCE_DIR=/rv/data/development/github.com/libcna/meta-gl
cmake --build --preset debug
ctest --preset debug -R '<demo-a>|<demo-b>' --output-on-failure
```
Every matrix run (gpu, llvmpipe, zink × es-max, es-min) must PASS, and the build must have no warnings
(`cmake --preset headless-ci -DOGLES3_METAGL_SOURCE_DIR=... && cmake --build --preset headless-ci`
uses -Werror). Open `build/debug/test-results/<demo>_gpu_es-max.png` with an image viewer and look at it.
Only then add `"visual_review": {"date": "...", "by": "<agent>", "notes": "<what you saw>"}`.

## Finish
Commit only your demo directories on your branch, with an imperative summary and an explanatory
body, and no AI attribution lines. Do not merge. Report: branch and commit, check counts and matrix
results, spec ambiguities and how you resolved them, suspected meta-gl gaps, and missing feature IDs.
