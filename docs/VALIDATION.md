# Validation methodology

## What a run proves

Each executable, run with `--headless --validate`:

1. creates a real native OpenGL ES context (EGL pbuffer — a genuine default framebuffer
   with color/depth/stencil and optional MSAA; or an SDL3 window with `--window`);
2. calls `metagl::Initialize()` only after the context is current;
3. **runtime checks** (always): meta-gl reports `ApiKind::OpenGLES`; meta-gl's parsed version equals
   `GL_MAJOR_VERSION/GL_MINOR_VERSION`; the context meets the demo's minimum version (otherwise
   `SKIP_UNSUPPORTED`); `glGetError() == GL_NO_ERROR` after `init()`, after the last frame and
   after `shutdown()` (deletion of every object it created);
4. **demo checks** (`validate()`): observable results — pixels against an independent CPU
   reference, buffer/texture readback, state queries, or deliberately provoked GL errors.

A run is `PASS` only if all runtime checks pass, `validate()` recorded **at least one** check,
and none failed. An empty `validate()` is a `FAIL` — never a vacuous success. A skip or block raised after a check
has already failed is reported as `FAIL`, so a late `demo::Skip` cannot hide a failure.

## Outcomes and exit codes

| Outcome | Exit | Meaning |
|---|---|---|
| `PASS` | 0 | all checks passed |
| `FAIL` | 1 | a check failed, a setup step threw `demo::Failure`, or no checks were recorded |
| `SKIP_UNSUPPORTED` | 77 | the context is below `min_es` or lacks an optional capability the demo needs |
| `BLOCKED_ENVIRONMENT` | 77 | no usable EGL/SDL context, or `metagl::Initialize` failed |
| `BLOCKED_META_GL` | 77 | a meta-gl gap prevents a correct implementation (see META_GL_GAPS.md) |

CTest maps 77 to "skipped" (`SKIP_RETURN_CODE`); the precise outcome is in the JSON written by
`--result-json` and is what the coverage generator reads.

## Test matrix (CTest)

Every executable is registered as `<name>:<driver>:<mode>`:

| driver | environment | what it adds |
|---|---|---|
| `gpu` | default EGL vendor (radeonsi on the reference machine) | real hardware path |
| `llvmpipe` | `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe` | deterministic software ES 3.2 |
| `softpipe` | `... GALLIUM_DRIVER=softpipe` | a **native ES 3.1 / GLSL ES 3.10** implementation: ES 3.2 executables must report `SKIP_UNSUPPORTED`, 3.20 shaders must be rejected. Only registered for `min_es` 3.2 executables and those listing it in `extra_drivers` (it miscompiles some complex shaders, see KNOWN_LIMITATIONS.md) |
| `zink` | `... GALLIUM_DRIVER=zink` (on lavapipe) | GL-on-Vulkan: a different backend (NIR→SPIR-V), so agreement is less correlated |

| mode | environment |
|---|---|
| `es-max` | the driver's highest ES version |
| `es-min` | `MESA_GLES_VERSION_OVERRIDE=<min_es>` (skipped for `min_es` 3.2) |

**What `es-min` proves, and what it does not.** The override changes `GL_VERSION`,
`GL_MAJOR/MINOR_VERSION` and `GL_SHADING_LANGUAGE_VERSION`. So it exercises meta-gl's version
detection, the runtime's version gating, and demo code that branches on the version. It does
**not** emulate a real lower-version implementation. On radeonsi and llvmpipe the GLSL compiler
still accepts `#version 310 es`/`320 es` under an ES 3.0 override, and entry points stay loadable.
Version-rejection assertions (e.g. "a 3.20 shader fails on an ES 3.1 context") are therefore made
only on **native** lower-version contexts (softpipe = ES 3.1; ANGLE `native` = ES 3.1 and `native-min` = ES 3.0,
see below). Under the override they are printed as notes.

All runs use a headless EGL pbuffer (`EGL_PLATFORM=surfaceless` for Mesa), a 256×256 surface (333×777 for the
size runs), fixed `--seed 1`, and deterministic animation time (`frame / 60 s`).

**Counting.** CTest reports SKIP_UNSUPPORTED and BLOCKED_* as "Skipped" and counts them as not failed, so its "100%
tests passed" means *no FAIL*, not *all PASS*. Reports of this project give the outcomes separately (e.g. the
release-candidate debug matrix: 701 registered tests = 669 PASS + 26 SKIP_UNSUPPORTED + 5 BLOCKED_ENVIRONMENT + the
CPU unit test). Earlier reports wrote such a run as "700/700".

## ANGLE: an independent implementation (driver `angle`, mode `native`)
Every Mesa driver shares Mesa's GL front-end (state tracker, GLSL compiler, validation). Their agreement is
therefore not independent evidence. The `angle` driver runs the same executables through ANGLE: Google's
GL ES front-end and GLSL translator, here with the Vulkan backend on lavapipe (software).
- **Build used:** the one shipped with the Android SDK emulator (`~/Android/Sdk/emulator/lib64/gles_angle`,
  ANGLE 2.1.17841, git 5fa2e61da947). It is auto-detected, or set with `-DOGLES3_ANGLE_DIR=`.
- **How it is loaded:** only at the context-provider level. A shim makes `libEGL.so.1` resolve to ANGLE's libEGL,
  `libGLESv2.so` is preloaded, and `OGLES3_EGL_PLATFORM=angle` selects `EGL_PLATFORM_ANGLE_ANGLE`
  (Vulkan, headless). Every GL call still goes through meta-gl.
- **Version:** mode `native` takes ANGLE's highest version, **ES 3.1** (ES 3.2 executables are SKIP_UNSUPPORTED).
  Mode **`native-min`** is registered for every min_es 3.0 executable. It requests a context of exactly the
  minimum version via `EGL_ANGLE_create_context_backwards_compatible = FALSE` (`OGLES3_ANGLE_EXACT_VERSION=1`).
  ANGLE then creates a **genuine ES 3.0 context**: GL_VERSION "OpenGL ES 3.0.0", GLSL ES 3.00, and ES 3.1 objects
  rejected (e.g. `glCreateShader(COMPUTE_SHADER)` → INVALID_ENUM). Unlike Mesa's `es-min` override, this is native
  ES 3.0 evidence. All 54 pass on true ES 3.0. One ANGLE ES 3.0-only deviation (output-array capture in transform feedback) is
  recorded.
- **Vulkan driver under ANGLE: lavapipe (software) by default** (`OGLES3_ANGLE_VULKAN_ICD`). On 2026-11-11 one
  ANGLE/RADV case faulted the GPU. Repeated runs caused amdgpu `ring gfx timeout, soft recovered` events (72 kernel
  log lines between 20:23 and 20:38) and visible desktop freezes. The ANGLE matrix therefore never touches the real
  GPU by default. Point `OGLES3_ANGLE_VULKAN_ICD` at the RADV ICD only on purpose. ANGLE's API front-end and
  shader compiler are independent of Mesa; lavapipe, the backend, is the same Vulkan driver that zink uses.
- **GPU-fault safety rule:** a case known to fault or hang a hardware GPU must not run in the default matrix,
  not even in a child process. A child process protects the runner, not the machine. It needs an explicit opt-in
  variable (e.g. `OGLES3_RUN_GPU_FAULT_CASES=1`).

## Window-size runs (driver `llvmpipe`, mode `size-333x777`)
Every executable also runs once at 333x777 on llvmpipe (`OGLES3_TEST_SIZE_VARIANT`, default ON). The size is
non-square, portrait and odd, so fixed-pixel layouts and size-tuned thresholds show up as failures. The coverage
generator counts these runs: a failure makes the executable FAILED. `tools/sweep_sizes.sh` runs more sizes
(1000x200, 200x1000, ...) on demand. The rule is that a valid render PASSes at every size, and a surface too small for
non-vacuous checks SKIPs with the minimum stated in the README.

## Known driver deviations (XFAIL)

When a driver demonstrably violates a normative rule, the test keeps the strict expectation and
names the driver (`// @xfail: Mesa | reason` in shader cases). On a matching driver the result is
reported as XFAIL with the spec citation. Everywhere else the strict check applies. The matching rule is one
function, `demo::deviation_matches` (`common/include/demo/check.hpp`), pinned by the CPU test
`unit-deviation-matcher`: "Mesa" selects every Mesa driver; a Mesa driver name selects that driver ("llvmpipe" also
zink on lavapipe, which shares its rasterizer); "ANGLE" selects ANGLE; ANGLE is matched on GL_VERSION only, so no
Mesa or backend name in its GL_RENDERER can excuse it; an empty name matches nothing. Each deviation
is listed in `docs/KNOWN_LIMITATIONS.md`. Checks are never loosened to make a driver pass.

## Pixel comparison policy

- References are computed on the CPU from the same inputs (barycentric interpolation, blend
  equations, filter weights, ...), not stored golden images, unless a feature is inherently
  image-based.
- Tolerances are stated per check and justified by the spec (8-bit quantization ±1;
  interpolation at `mediump` ±3; filtering allows implementation-dependent precision).
- Probes avoid primitive edges and texel boundaries where the spec permits implementation
  variance (e.g. exact rasterization of edge pixels, LOD selection details).
- Undefined results (feedback loops, out-of-bounds without robustness) are documented, not tested.
  Inventory rows whose verification method is `doc` get the status DOCUMENTED_ONLY. That status does not
  count as passed.

## Suite-wide assumptions beyond the normative text
- **No assumption about the multisample resolve of normalized colors (removed 2026-11-11).** ES 3.0 §4.3.3 says that
  "the sample values for each pixel are resolved in an implementation-dependent manner" for floating-point and
  normalized formats, and §4.1.10 says of the window that "the method of combination is not specified" (an average is
  only recommended). Neither guarantees that a resolved value lies between the samples, nor that a pixel whose samples
  are all equal resolves to that value. es30-framebuffer-blit-msaa, es30-flat-centroid and
  es30-multisample-default-framebuffer used to assert both; they now print those measurements as `note: observed ...`
  lines and assert only what holds for any resolve: the integer and stencil rule ("a single sample's value is
  selected"), the depth rule ("between the minimum and maximum depth values in the pixel") on pixels whose samples were
  set by scissored clears, the filter being ignored, identical-bounds sub-rectangles equal to the whole resolve (a
  resolved pixel is a function of that pixel's samples; Appendix A.1 repeatability), per-sample readback with
  `texelFetch(gsampler2DMS)` in ES 3.1+ contexts, and per-sample stencil counts with occlusion queries.

- **Stretched NEAREST display blits** (es30-framebuffer-blit-msaa, es30-framebuffer-invalidate) assume the
  pixel-centre mapping `(d + 0.5) * src / n` and accept either neighbour within 0.05 pixel of a source pixel edge.
  ES 3.0 §4.3.3 gives no mapping formula. This affects only the presentation into the window, not the blit rules
  under test (independent review, 2026-11-12).

## Evidence traceability and staleness
Every result JSON records:
- the executable path and mtime, and the run time;
- the GL/GLSL version, renderer and vendor strings;
- the EGL platform and meta-gl version;
- `version_override`, which holds the `MESA_GLES_VERSION_OVERRIDE` value for es-min runs;
- every check with its detail.

A run first writes a `CRASHED` result and replaces it when it finishes. A crash, abort or timeout therefore leaves
`CRASHED` behind, and the generator treats that as a failure; the run cannot silently disappear.
The coverage generator ignores a result when its executable was rebuilt after the run, or when compiled sources (demo or `common/`) are newer than the executable, or when run-time inputs
(shaders, data, the CMake test registration) are newer than the run. Those results are listed under
"Stale results" in COVERAGE_SUMMARY.md. The generator also reads the CTest registration of the build directory
(`ctest --show-only=json-v1`): a registered run without a fresh result prevents FULLY_VALIDATED and is listed under
"Registered runs without a fresh result", and a result file of a test that is no longer registered is ignored. The
summary states the source and meta-gl revisions it was generated from.

`tools/release/collect_evidence.py` runs one matrix and writes machine-readable evidence: both revisions (it refuses
dirty working trees), the build configuration, compiler and tool versions, the driver strings seen, the exact ctest
command, start and end time, one row per registered test, and the amdgpu lines of the kernel log during the run.
Only result files written during that run count. The release-candidate evidence is in `docs/evidence/`.
The generator lints every `api` claim: the entry point must be called by the executable's sources, directly or
through a `common/` helper or a meta-gl convenience overload. The lint is name-based, so it is a heuristic; the
actual verification is described per entry in `demo.json`.

## Reproducibility from a clean checkout
Procedure (the release-candidate run, with its revisions and results, is in
[FINAL_RELEASE_READINESS.md](FINAL_RELEASE_READINESS.md) section E):
1. `git clone https://github.com/libcna/meta-gl <root>/libcna/meta-gl` and check out
   `333d777fadbe53613a9c2eead358ffbfe6a2ef33` (on `develop`).
2. Clone opengles3-demos to `<root>/libcnalab/opengles3-demos` and check out the revision under test.
3. `cmake --preset debug && cmake --build --preset debug`: no warnings; 83 executables plus `unit-deviation-matcher`.
   Configure stops with an error when the meta-gl checkout lacks 333d777.
4. `tools/release/collect_evidence.py --build build/debug --label debug --out <dir>` (or plain
   `ctest --preset debug`), then the same for the `sanitize` and `headless-ci` presets.
5. `cmake --build --preset debug --target coverage`: 82 FULLY_VALIDATED + 1 VISUALLY_VERIFIED, every registered run
   with a fresh result.

Requirements not in the repositories: Mesa (radeonsi, llvmpipe, zink and the lavapipe ICD), EGL, optionally SDL3,
and for the `angle` driver an ANGLE build (`OGLES3_ANGLE_DIR`, auto-detected from the Android SDK emulator).
Without a hardware GPU the `gpu` runs use whatever EGL selects, and the `angle` runs are absent without ANGLE.

History: the first clean-clone check (2026-11-11 23:40, opengles3-demos `d100bfc`, before the size and ANGLE
`native-min` runs existed) gave 563 registered runs without a failure.

## Coverage status definitions

See `tools/coverage/generate_coverage.py` (single source of truth):
`NOT_STARTED`, `IMPLEMENTED_UNVERIFIED`, `BUILD_PASSED`, `FAILED`, `BLOCKED_BY_PLATFORM`,
`BLOCKED_BY_META_GL`, `UNSUPPORTED_BY_CONTEXT`, `RUNTIME_PASSED` (≥1 validated PASS),
`VISUALLY_VERIFIED` (+ the capture was actually inspected and recorded in `demo.json`),
`FULLY_VALIDATED` (every registered matrix run has a fresh result, each a validated PASS or a justified
SKIP_UNSUPPORTED; gpu and llvmpipe pass in es-min and es-max).

A function's status is the best status among the executables whose `demo.json` `api` map
claims it **with a description of how it is verified**. Use of a function without
verification of its effect must not be listed in `api`.

## Limits of this evidence

- Passing these tests is **not** Khronos conformance. Two Mesa drivers share the GLSL compiler
  front end and much of the state tracker; agreement between them is weaker evidence than
  agreement between independent vendors.
- CPU sanitizers (`sanitize` preset) check our C++ only, not GPU memory safety.
- `es-min` relies on Mesa's version override, which changes only the reported API version (see above). Native
  lower-version evidence comes from softpipe (ES 3.1) and ANGLE (`native` ES 3.1, `native-min` ES 3.0).
- The four Mesa drivers share Mesa's GL front-end and GLSL compiler, which is why the "Mesa" XFAILs appear on all of
  them. ANGLE is an independent front-end and compiler, but in this matrix it runs on lavapipe, the same Vulkan
  driver as zink, so it is not an independent rasterizer. No driver from another vendor was available; ES 3.2 has
  Mesa evidence only (ANGLE 2.1.17841 stops at ES 3.1).
- Results describe Mesa 25.0.7 and ANGLE 2.1.17841 on one machine (AMD Radeon 780M). Another driver version can
  pass or fail differently; a new deviation needs a spec reading, a KNOWN_LIMITATIONS row and a narrowly matched
  XFAIL, never a looser check.
