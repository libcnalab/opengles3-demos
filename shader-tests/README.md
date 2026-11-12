# glsl-compile-tests: GLSL ES compile and link cases

`shader-tests/` is one executable, `glsl-compile-tests`, that runs data-driven compile and link tests for GLSL ES
1.00, 3.00, 3.10 and 3.20. The demos show what a shader *does*; these cases pin down what the compiler and linker must
*accept or reject*: a missing default precision, a `const` initialized from a uniform, mismatched interface blocks, a
3.10 feature in a `#version 300 es` shader, a geometry stage without a vertex stage, and so on.

Minimum OpenGL ES version: 3.0 (cases in `glsl310/` and `glsl320/` need ES 3.1 and 3.2).

## The cases

A case is one or more shader files that share a name (stem) in one directory. The extension gives the stage:
`.vert`, `.frag`, `.comp`, `.geom`, `.tesc`, `.tese`.

| directory | language | cases | of which negative (must fail) |
|---|---|---|---|
| `glsl100/` | GLSL ES 1.00 (and 1.00 mixed with 3.00) | 21 | 13 |
| `glsl300/` | GLSL ES 3.00 | 98 | 77 |
| `glsl310/` | GLSL ES 3.10 | 14 | 7 |
| `glsl320/` | GLSL ES 3.20 | 4 | 1 |

Each file states its expectation in comments after the `#version` line:

```glsl
#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.2 "Initializers for const declarations must be constant expressions" - a uniform is not one (§4.3.3)
// @feature: GLSL300.QUAL.CONST
precision mediump float;
out vec4 o;
uniform float u;
void main() { const float c = u; o = vec4(c); }
```

| annotation | meaning |
|---|---|
| `@expect: compile-pass` | every file compiles |
| `@expect: compile-fail` | at least one file fails to compile |
| `@expect: link-pass` | every file compiles and the program links |
| `@expect: link-fail` | every file compiles, but linking fails |
| `@expect: build-fail` | compiling or linking fails (the spec allows either) |
| `@min-es: 3.1` | minimum context version; the default comes from the directory (`glsl100`/`glsl300` 3.0, `glsl310` 3.1, `glsl320` 3.2) |
| `@spec: ...` | the rule being tested, with its section |
| `@feature: ID` | a GLSL inventory ID (`inventory/glsl_features.csv`); read by the coverage generator |
| `@xfail: <driver> \| <why>` | a known driver deviation, matched to the narrowest driver name (see below) |

## How it is validated

For every case the executable compiles each file with `glCompileShader`, links when the expectation needs it, and
checks `COMPILE_STATUS` / `LINK_STATUS` against the expectation: one check per evaluated case, plus one that at least one
case applies to the context. Each check shows the first line of the info log; a positive case that fails shows the
whole log.

- **Version gating.** On a context that is *natively* below a case's `@min-es`, a case that would pass must be
  rejected instead (the implementation does not know that `#version` or stage). softpipe (native ES 3.1) and ANGLE
  (native ES 3.1 and ES 3.0) assert this. Under `MESA_GLES_VERSION_OVERRIDE` only the reported version changes, so
  the outcome is printed and not asserted (docs/VALIDATION.md, "What `es-min` proves").
- **Known deviations.** `@xfail: Mesa | ...` turns a failed expectation into XFAIL on a driver matching that name
  (`demo::deviation_matches`, the same rule the demos use); every other driver gets the strict check, and a driver
  that starts to conform reports XPASS. Each is listed with its spec quote in docs/KNOWN_LIMITATIONS.md.
- Cases are discovered from the source tree at run time, so a new file is picked up without a rebuild.

## How to run

```sh
./build/debug/bin/glsl-compile-tests --headless --validate         # every case on the default driver
ctest --preset debug -R glsl-compile-tests                          # every driver and version mode
EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=softpipe \
    ./build/debug/bin/glsl-compile-tests --headless --validate      # native ES 3.1: 3.20 cases must be rejected
```

## Adding a case

Put the shader file(s) in the directory of their language version, start each with the `#version` line, then the
annotations. A negative case should break exactly one rule, cited in `@spec`, so that it cannot fail for an
unrelated reason. Add the `@feature` ID when the case is evidence for an inventory row.

## Related

- Runtime semantics of the built-in functions: `tests/glsl/glsl300-builtins`, `glsl310-builtins`,
  `glsl320-builtins`; language rules that need execution (evaluation order, out parameters): `tests/glsl/glsl-language`.
- Inventory methods: [docs/GLSL_COVERAGE.md](../docs/GLSL_COVERAGE.md); per-item status:
  [docs/generated/](../docs/generated/COVERAGE_SUMMARY.md).
