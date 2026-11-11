# Independent audit report — tests and shader tests

Audit scope: `tests/glsl/glsl300-builtins`, `tests/glsl/glsl310-builtins`,
`tests/glsl/glsl320-builtins`, `tests/glsl/glsl-language`,
`tests/negative/driver-crash-readpixels-none`, `tests/negative/metagl-contracts`, and
`shader-tests`.

## Findings

### BUG — hard check uses an undefined integer division result

- **Location:** `tests/glsl/glsl300-builtins/main.cpp:314-320`
- **Spec:** GLSL ES 3.00 §12.33: “The rounding mode is undefined for this version of the specification.”
- **Finding:** The integer case asserts the exact result of signed `-17 / 5` (`ia.x / ib.x`, with
  `ia.x = -17` and `ib.x = 5`). This quotient is not exact, so a conforming implementation may
  choose a different rounding result. The test passes on llvmpipe, but that does not make the hard
  check portable. The README's claim that undefined rounding cases are avoided is also contradicted.
- **Fix:** Use exactly divisible signed operands for this case (for example `-15 / 5`) and update
  the CPU reference, or omit the result from the hard comparison.

### BUG — meta-gl manifest reverses the ES 3.0 matrix transpose rule

- **Location:** `tests/negative/metagl-contracts/demo.json:9`
- **Spec:** ES 3.0 §2.12.6: “If transpose is FALSE, the matrix is specified in column major order,
  otherwise in row major order.”
- **Finding:** The manifest says “UniformMatrix transpose must be FALSE.” That is not the ES 3.0
  rule; it was the ES 2.0 restriction. The test correctly exercises `GL_TRUE` and verifies the
  transposed uniform value.
- **Fix:** Replace the parenthetical with the ES 3.0 rule, or remove it from the manifest entry.

### BUG — manifest claims coverage of undefined behavior that is explicitly not tested

- **Location:** `tests/glsl/glsl-language/demo.json:62`
- **Spec:** GLSL ES 3.00 §5.9, §12.33, and the listed built-in domain rules define several tested
  edge areas as undefined; the test README says, “Nothing in this list is asserted.”
- **Finding:** `GLSL300.UNDEFINED_BEHAVIOR` is listed as a validated GLSL feature even though the
  test deliberately avoids those results and only documents them. This overstates coverage.
- **Fix:** Remove this ID from `glsl_features`, or represent it in documentation-only coverage
  outside the validated feature list.

### WEAK — rasterizer-discard state is claimed without checking its effect

- **Location:** `tests/glsl/glsl300-builtins/demo.json:25-26`
- **Spec:** ES 3.0 §4.1.4: “When enabled, primitives are discarded immediately before the
  rasterization stage.”
- **Finding:** The manifest claims `glEnable` and `glDisable` coverage for
  `RASTERIZER_DISCARD`, but transform-feedback results are captured whether rasterization is
  enabled or discarded. No framebuffer result or enabled-state query proves either state change.
- **Fix:** Query and assert `glIsEnabled(RASTERIZER_DISCARD)` immediately after each toggle, or
  otherwise validate the rasterization effect before retaining these API/feature claims.

### WEAK — API manifest labels diagnostic output as verified API behavior

- **Location:** `shader-tests/demo.json:27-28`
- **Spec:** GLSL ES §1.3 requires compile/link errors to be reported at the latest by link time;
  it does not require non-empty diagnostic text for each failure.
- **Finding:** `glGetShaderInfoLog` and `glGetProgramInfoLog` are listed as API coverage because
  logs are copied into check details, but the harness never asserts their return values or
  contents. These calls provide diagnostics for other checks, not a validated behavior claim.
- **Fix:** Remove these two entries from `api`, or add a separately justified assertion for the
  info-log behavior.

### WEAK — indexed API query is listed despite being explicitly unasserted

- **Location:** `tests/glsl/glsl-language/demo.json:38`
- **Spec:** GLSL ES 3.10 §7.2 gives minimum values for the built-in constants; the README and
  manifest state that equality between the other built-in constants and API limits is not
  asserted.
- **Finding:** `glGetIntegeri_v` values are only printed alongside compute built-in constants.
  The manifest explicitly says “not asserted,” so this is not verified API coverage.
- **Fix:** Remove this entry from `api`, or add an assertion grounded in a normative relation.

## Clean directories

- `tests/glsl/glsl310-builtins` — checked compute readback and array-of-arrays expectations,
  defined integer/float edge cases, negative controls, version checks, and manifest claims. No
  additional findings.
- `tests/glsl/glsl320-builtins` — checked atomic serializability, image formats and access paths,
  fma/precise bounds, compile controls, and Mesa deviation scope. No additional findings.
- `tests/negative/driver-crash-readpixels-none` — checked the defined error cases, known Mesa
  crash scope, and child-process timeout/marker handling. No additional findings.
- `tests/negative/metagl-contracts` — death tests avoid driver calls in children; row-major
  transpose and context restoration are checked. Apart from the manifest citation above, no
  additional findings.
- `shader-tests` — checked all 137 discovered cases, shared-stem annotation consistency, version
  gating, and documented XFAIL rules. Apart from the diagnostic API manifest entries above, no
  additional findings.

## Validation performed

- Configured with the supplied read-only meta-gl path and built the full project with `-j 1`:
  177/177 build steps completed.
- Ran the assigned executables with `--headless --validate` on llvmpipe. The suite reported PASS;
  `test-glsl-language` also passed at 333×777.
- Re-ran `glsl-compile-tests` (137 cases, 138 checks) and `test-glsl320-builtins` (83 checks):
  both PASS. `test-glsl310-builtins` passed 77 checks, `test-glsl300-builtins` passed 42, and
  `test-metagl-contracts` passed 19. The crash-isolation test passed 30 checks; its three
  documented Mesa cases were observed to segfault inside the isolated child call.
