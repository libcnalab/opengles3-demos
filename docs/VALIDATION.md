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

## Suite-wide assumptions beyond the normative text
- **Multisample resolve is a convex combination of the samples.** ES 3.0 §4.3.3 says only that the samples are
  "converted to a single sample" in an implementation-dependent manner. Three demos rely on the resolve being a
  weighted average of the samples: es30-framebuffer-blit-msaa, es30-flat-centroid (resolved bands) and
  es30-multisample-default-framebuffer (bounds, seam sums, coverage area). The checks that depend on it are:
  - resolved values lie between the sample colours;
  - a pixel whose samples are all equal resolves to that value.
  Both READMEs state the assumption. A conformant implementation that resolves differently would fail these
  checks without violating the spec.

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
