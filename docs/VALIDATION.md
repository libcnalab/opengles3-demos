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
and none failed. An empty `validate()` is a `FAIL` — never a vacuous success.

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
only on **native** lower-version drivers (softpipe = ES 3.1). Under the override they are printed
as notes. There is currently no native ES 3.0-only driver in the matrix (KNOWN_LIMITATIONS.md).

All runs use `EGL_PLATFORM=surfaceless`, a fixed 256×256 surface, fixed `--seed 1`, and
deterministic animation time (`frame / 60 s`).

## ANGLE: an independent implementation (driver `angle`, mode `native`)
Every Mesa driver shares Mesa's GL front-end (state tracker, GLSL compiler, validation). Their agreement is
therefore not independent evidence. The `angle` driver runs the same executables through ANGLE: Google's
GL ES front-end and GLSL translator, here with the Vulkan backend on RADV.
- **Build used:** the one shipped with the Android SDK emulator (`~/Android/Sdk/emulator/lib64/gles_angle`,
  ANGLE 2.1.17841, git 5fa2e61da947). It is auto-detected, or set with `-DOGLES3_ANGLE_DIR=`.
- **How it is loaded:** only at the context-provider level. A shim makes `libEGL.so.1` resolve to ANGLE's libEGL,
  `libGLESv2.so` is preloaded, and `OGLES3_EGL_PLATFORM=angle` selects `EGL_PLATFORM_ANGLE_ANGLE`
  (Vulkan, headless). Every GL call still goes through meta-gl.
- **Version:** the build exposes **ES 3.1 natively**. ES 3.2 executables are SKIP_UNSUPPORTED. The context is not
  produced by a version override, so ANGLE gives the only native lower-version (3.1) evidence in the matrix.
  ES 3.0 requests also return a 3.1 context, as EGL allows.
- **Shared code:** the backend still runs on Mesa's RADV Vulkan driver. Rasterization and texture hardware are
  shared with radeonsi; the API front-end and the shader compiler are not.

## Known driver deviations (XFAIL)

When a driver demonstrably violates a normative rule, the test keeps the strict expectation and
names the driver (`// @xfail: Mesa | reason` in shader cases). On a matching driver the result is
reported as XFAIL with the spec citation. Everywhere else the strict check applies. Each deviation
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
"Stale results" in COVERAGE_SUMMARY.md. The summary also states the source revision it was generated from.
The generator lints every `api` claim: the entry point must be called by the executable's sources, directly or
through a `common/` helper or a meta-gl convenience overload. The lint is name-based, so it is a heuristic; the
actual verification is described per entry in `demo.json`.

## Coverage status definitions

See `tools/coverage/generate_coverage.py` (single source of truth):
`NOT_STARTED`, `IMPLEMENTED_UNVERIFIED`, `BUILD_PASSED`, `FAILED`, `BLOCKED_BY_PLATFORM`,
`BLOCKED_BY_META_GL`, `UNSUPPORTED_BY_CONTEXT`, `RUNTIME_PASSED` (≥1 validated PASS),
`VISUALLY_VERIFIED` (+ the capture was actually inspected and recorded in `demo.json`),
`FULLY_VALIDATED` (every matrix run is a validated PASS or a justified SKIP_UNSUPPORTED; gpu and llvmpipe pass in es-min and es-max).

A function's status is the best status among the executables whose `demo.json` `api` map
claims it **with a description of how it is verified**. Use of a function without
verification of its effect must not be listed in `api`.

## Limits of this evidence

- Passing these tests is **not** Khronos conformance. Two Mesa drivers share the GLSL compiler
  front end and much of the state tracker; agreement between them is weaker evidence than
  agreement between independent vendors.
- CPU sanitizers (`sanitize` preset) check our C++ only, not GPU memory safety.
- `es-min` relies on Mesa's version override, which changes only the reported API version (see above).
- All four drivers are Mesa: they share the GLSL front end (which is why the two XFAILs appear on all of them).
