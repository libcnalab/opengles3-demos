# test-glsl-language

Run-time checks of GLSL ES *language* rules: the parts of the shading language that are not
built-in functions but still produce observable values. These are operators, parameter passing,
constant expressions, scoping, comments, built-in constants and uniforms, invariance, and the
GLSL ES 1.00 language that every ES 3.x context must still accept. The compile-only half of the
same rules (shaders that must be rejected) lives in `shader-tests/` (see the end of this file).

- **Minimum version:** OpenGL ES 3.0. Shaders use GLSL ES 1.00 and 3.00. The GLSL ES 3.10 parts
  (version mixing with `310 es` and the §7.2 constants) run only when the context reports ES 3.1
  or newer. On the `es-min` runs they print a note and are skipped.
- **Executable:** `test-glsl-language` (tests/glsl/glsl-language).

## How the values are observed

Each language case is a block of GLSL that writes four `uvec4` values `r[0..3]`. Floats are stored
as their bit patterns with `floatBitsToUint`. The inputs come from uniform arrays
`uf[4]`/`ui[4]`/`uu[4]`, so the compiler cannot fold the run-time expressions. Every case runs
**twice**, once in each shader language:

| language | how the 16 words come back |
|---|---|
| vertex | outputs `flat out uvec4 o0..o3`, one point drawn with `RASTERIZER_DISCARD` and captured with **transform feedback** (`glTransformFeedbackVaryings`, `glBeginTransformFeedback(POINTS)`, `glMapBufferRange`) |
| fragment | `layout(location = 0..3) out uvec4`, a full-viewport triangle (positions from `gl_VertexID`) into a 1x1 framebuffer with four **RGBA32UI** colour attachments (`glDrawBuffers`); each attachment is read with `glReadBuffer` + `glReadPixels(RGBA_INTEGER, UNSIGNED_INT)` after a `glClearBufferuiv` sentinel |

Both paths are exact: no conversion or quantization touches the values. RGBA32UI is
colour-renderable in ES 3.0 (Table 3.13), and `MAX_DRAW_BUFFERS` is at least 4.

### Comparison classes (GLSL ES 3.00 §4.5.1, highp)

| class | used for | rule |
|---|---|---|
| bits | integers, booleans (0/1), bit casts, packing | identical 32-bit pattern |
| float | exactly representable float results | equal; `+0` and `-0` are interchangeable ("subject to allowing positive and negative zeros to be interchanged") |
| rounded | `a + b`, `a - b`, `a * b` with inexact results, incl. overflow of `1e30 * 1e30` | §4.5.1 says "Correctly rounded" but "The rounding mode is not defined". The result must be one of the two floats that bracket the exact value (FLT_MAX or +Inf for the overflow) |
| ULP | `a / b`, `1.0 / b` | 2.5 ULP for b in [2^-126, 2^126] |
| quality | `sin`, `exp`, `pow`, `normalize` inside a constant expression | precision undefined; relative sanity bound 1e-3 only. The §4.6.3 invariance of the same expression is compared bit-exactly |

A negative control feeds a value one step outside each class into the comparator and checks that
it is rejected.

## What is checked

### GLSL ES 1.00 on an ES 3.x context (picture)

The default framebuffer is split into 3x2 cells (all positions relative to the surface size):

| cell | shader | expected |
|---|---|---|
| bottom left | `#version 100`, `attribute` position + colour, `varying`, `gl_FragColor` | affine colour (u, v, 1-u): 25 probes, tolerance 2 (RGBA8 ±1 plus mediump interpolation, GLSL ES 1.00 §4.5.2 relative 2^-10 ≈ 0.25 step) |
| bottom middle | **no `#version` line at all** (`#if __VERSION__ != 100 #error`), `texture2D`, `gl_FragData[0]` | the 4x4 NEAREST texture, probed at the 16 texel centres, ±1 |
| bottom right | `texture2DProj` with `vec3` (left) and `vec4` (right; 3rd component 5.0 must be ignored), q = 2.5 | same texels, ±1 |
| top left | `textureCube` with six directions | the six face colours |
| top middle | **vertex-shader** `texture2DLod` (lod 0, 1, 3), `texture2DProjLod` vec3 (lod 2), vec4 (lod 1), `textureCubeLod` | the colour of the selected mipmap level of an 8x8 texture whose four levels are red/green/blue/yellow, or the -Z face |
| top right | fragment-only `texture2D(s, uv, bias)` | the texture coordinates step 1/16 per pixel, so ρ = 0.5 and λ_base = -1 exactly (axis-aligned, so the ρ approximations of ES 3.0 §3.8.10 are exact). Bias 0 gives magnification (level 0, red). Bias 2 (≤ `MAX_TEXTURE_LOD_BIAS`, which is at least 2) gives λ = 1, level 1 (green) with `NEAREST_MIPMAP_NEAREST` |

A negative control checks that the two bias halves differ. Specs: GLSL ES 1.00 §3.4, §4.3.3,
§4.3.5, §7.2, §8.7; GLSL ES 3.00 §1.5 (ES 3.0 runs ES 2.0 shaders; only `gl_FragData[0]`).

### Version mixing (GLSL100.MIX_VERSIONS, GLSL310.VERSION)

`LINK_STATUS` must be FALSE for 100 VS + 300 es FS and for 300 es VS + 100 FS (GLSL ES 3.00 §3.4
"Shaders declaring version 3.00 ... cannot be linked with shaders declaring version 1.00", §1.5).
On ES ≥ 3.1 it must also be FALSE for 310 es + 300 es (both orders) and 310 es + 100 (GLSL ES
3.10 §3.3). Same-version pairs, including a 310 es VS that checks `__VERSION__ == 310`, are
positive controls that must link.

### Language cases (vertex and fragment)

| rows | case | spec |
|---|---|---|
| COMMENTS_TOKENS | `3/**/-/**/1` is 2 (a comment is one space). A `//` comment ending in `\` swallows the next line, because phase 6 (line continuation) runs before phase 7 (comments). `in\`+newline+`t` is the keyword `int`. `my__value` is legal | GLSL ES 3.00 §3.1, §3.6, §3.8 |
| OPERATORS | `+ - *` correctly rounded, unary `-`/`+`, division in 2.5 ULP | §5.9, §4.5.1 |
| OPERATORS | `< <= > >=` (incl. unsigned compare of 0xFFFFFFFFu), `== !=` on scalars, vectors, structures and arrays | §5.9, §5.7 |
| OPERATORS | `&& \|\| ^^ !` truth tables; short-circuit of `&&`/`\|\|`, `^^` evaluates both, `?:` evaluates one branch (side-effect counters) | §5.9 |
| OPERATORS | sequence operator (left to right, value/type of the right-most operand, in a `for` header) | §5.9 |
| OPERATORS | `++`/`--` prefix/postfix on int, uint (0u-- wraps), float, vec2, mat2 | §5.9 |
| OPERATORS | precedence/associativity: `2+3*4`, `1<<2+1`, `6^3&5`, `10-4-3`, `64/4/2`, `t\|\|f&&f`, `!f==t`, `a<b==t`, nested `?:`, `p = q = 3`, `2*3%4` | §5.1 table |
| OPERATORS | all eleven assignment operators on int, `*=`/`+=` on vec3, `*=` on mat2 | §5.8 |
| OPERATORS | integer wrap-around (`INT_MAX+1`, `INT_MIN-1`, `46341²`, `-INT_MIN`, `0xFFFFFFFFu+1u`, `0u-1u`, `0xFFFFFFFFu²`), exactly divisible `/` with both signs, `%` with non-negative operands | §4.1.3, §5.9 |
| OPERATORS | scalar ⊗ vector and component-wise vector arithmetic | §5.9, §5.10 |
| OPERATORS | `x[i++] = i` stores into `x[0]` (the left side is evaluated first) | §5.8 |
| PARAM_QUALIFIERS | `in` copies are local, `out` is written back, `inout` both, for scalars, vectors, arrays and structures | §6.1.1 |
| PARAM_QUALIFIERS | aliasing: `f(x, x)` with `(in a, out b)` returns the call-time copy of x and x becomes the out value. `(inout a, inout b)` sees two copies. The final x of the latter is **not** asserted ("The order in which output parameters are copied back to the caller is undefined") | §6.1.1 |
| PARAM_QUALIFIERS | `digits(i++, i++, i++)` is 123 and three `next_call()` arguments give 123: "evaluated at call time, exactly once, in order, from left to right". `add_ten(arr[j++])` modifies `arr[0]` and increments `j` once | §6.1.1 |
| QUAL.CONST | constant folding equals run-time evaluation for exactly defined operations (wrap-around, `-12/4`, `7%3`, `1.5*2.25`, `-16>>2`, bit operations, `floor/abs/max/clamp/fract`). Built-in calls, constructors, a matrix product, a struct field, an array element and `length()` of a non-constant array are constant expressions | §4.3.2, §4.3.3 |
| INVARIANT / QUAL.CONST | the same constant expression (`sin(1.0)*exp(0.5)+sqrt(2.0)/3.0`, `normalize(...)*pow(...)`) is bit-identical twice in one shader and in the vertex vs the fragment language | §4.6.3 |
| SCOPE | `int x = 2, y = x` (y = 2), `int x = x + 10` reads the outer x, the outer x is unchanged after the block, the do-while condition sees the outer `i` and not the body's `i`, the for-init `k` hides the outer `k`, `S S = S(3); S.x`, a local `vec4 uf` hides the uniform array, a block inside a function hides its parameter | §4.2.2, §4.2.3 |

### Built-in constants (GLSL300.BUILTIN_CONSTANTS, GLSL310.BUILTIN_CONSTANTS)

GLSL ES 3.00 §7.3 and 3.10 §7.2 say only: "The actual values used are implementation dependent,
but must be at least the value shown." The test therefore asserts:

- **Minimums, in both languages.** All 11 constants of GLSL ES 3.00 §7.3 are written by a vertex
  shader (transform feedback) and a fragment shader (RGBA32UI) and must be at least the value in
  the table (`gl_MinProgramTexelOffset` at most -8). On ES ≥ 3.1 the same is done in `#version 310 es`
  shaders for the 20 new constants (24 values, counting the `ivec3` work-group count and size) of
  GLSL ES 3.10 §7.2: image, compute and atomic-counter limits.
- **Equality with the API only where the spec ties them.** GLSL ES 3.00 §1.5: "The number of
  uniforms specified by gl_MaxVertexUniformVectors and returned by the corresponding API query is
  the same for GLSL ES versions 1.00 and 3.00 when used as part of OpenGL ES 3.0." So
  `gl_MaxVertexUniformVectors` must equal `MAX_VERTEX_UNIFORM_VECTORS` in the 3.00 vertex and
  fragment shaders and in a `#version 100` vertex shader (captured as a float, which is exact for
  these values). The 1.00 value must also be at least 128 (GLSL ES 1.00 §7.4).
- **Everything else is only reported.** For every other constant the output says whether it equals
  the API limit of the same name (for example `MAX_VERTEX_OUTPUT_COMPONENTS/4` for
  `gl_MaxVertexOutputVectors`, or `glGetIntegeri_v(MAX_COMPUTE_WORK_GROUP_COUNT, i)`). Nothing is
  asserted about it. On Mesa 25.0 every constant equals its API limit.

### gl_DepthRange (GLSL300.BUILTIN_UNIFORMS)

For `glDepthRangef(0,1)`, `(0.25,0.75)`, `(0.75,0.125)` and `(1.5,-0.5)`, `gl_DepthRange.near`,
`.far` and `.diff` must equal the values clamped to [0, 1] (ES 3.0 §2.13.1: "The parameters n and
f are clamped to the range [0, 1] when specified") and `f - n`. The differences are exact in
float. They are checked in both languages, and `glGetFloatv(DEPTH_RANGE)` must report the
clamped pair.

### Invariance (GLSL300.INVARIANT)

Two programs compute `gl_Position` with the same expression (sin, exp, inversesqrt, division, pow,
log2, dot) over 64 attribute vertices, both with `invariant gl_Position;`. The second program
also computes other outputs from the same sub-expressions, which invites different optimisation.
The captured `gl_Position` values must be **bit-identical** (§4.6.1: same inputs, same operations,
all data flow in one compilation unit). The same is checked for a user output with
`#pragma STDGL invariant(all)` as the first line. A negative control (the same program with
`+ 1e-3` on x) shows that the bitwise comparison detects a difference.

## Known Mesa deviations (XFAIL on Mesa, strict elsewhere)

| case | rule | Mesa 25.0 behaviour |
|---|---|---|
| `y[j] = ++j` | GLSL ES 3.00 §5.8 "Expressions on the left of an assignment are evaluated before expressions on the right of the assignment." `y[0]` must become 1 | the index of `y[j]` is read after `++j`, so it stores into `y[1]` (all drivers, both languages). `x[i++] = i` is correct |
| `store(arr[j], ++j)` with `out int dst` | §6.1.1 "All arguments are evaluated at call time, exactly once, in order, from left to right ... Evaluation of an out parameter results in an l-value that is used to copy out a value when the function returns." `arr[0]` must become 1 | the out l-value's index is evaluated at copy-out, after `++j`, so it writes `arr[1]` |

Compile-level deviations of the companion shader-tests: Mesa accepts `texture2D()` in
`#version 300 es` shaders and `(c ? a : b) = 1.0` (§5.8 "The ternary operator (?:) is also not
allowed as an l-value").

## Undefined and implementation-dependent behaviour (GLSL300.UNDEFINED_BEHAVIOR)

Nothing in this list is asserted. Where a case comes near one of these rules, the inputs are
chosen to stay on the defined side. Quotes are from GLSL ES 3.00.

| topic | spec text | how the test avoids it |
|---|---|---|
| division by zero | §5.9 "Dividing by zero does not cause an exception but does result in an unspecified value." (§4.5.1 adds for highp floats that "dividing a non-zero by 0 results in the appropriately signed IEEE Inf".) | no zero divisors |
| integer division rounding | §12.33 "RESOLUTION: The rounding mode is undefined for this version of the specification." | only exactly divisible operands (`12/4`, `-12/4`, `12/-4`, `0xFFFFFFFFu/5u`) |
| `%` | §5.9 "The resulting value is undefined for any component computed with a second operand that is zero ... Results are undefined if one or both operands are negative." | non-negative operands only |
| `INT_MIN / -1` | §4.1.3 "it is allowed to return either the minimum representable value or the maximum representable value" | not used |
| shifts | §5.9 "The result is undefined if the right operand is negative, or greater than or equal to the number of bits in the left expression's base type." | shift counts between 1 and 5 |
| NaN, subnormals, rounding mode | §4.5.1 "NaNs are not required to be generated ... Operations and built-in functions that operate on a NaN are not required to return a NaN as the result." "Any subnormal (denormalized) value ... can be flushed to 0." "The rounding mode cannot be set and is undefined." | no NaN or subnormal inputs or results; inexact results accept either bracketing float |
| float → uint | §5.4.1 "It is undefined to convert a negative floating point value to an uint." | not used |
| uninitialized variables | §5.8 "Reading a variable before writing (or initializing) it is legal, however the value is undefined."; §4.3 globals "will enter main() with undefined values"; §4.6.4 undefined values cannot be made invariant | every variable is initialized; `r[0..3]` are zeroed first |
| out parameters | §6.1.1 "The order in which output parameters are copied back to the caller is undefined." An out parameter that the function never writes copies back an undefined value | the aliased `inout_pair(y, y)` result `y` is not checked |
| missing return | §12.34 "if the function returns without executing a return statement the behavior is undefined" | every path returns |
| out-of-range indexing | §4.1.9 "Undefined behavior results from indexing an array with a non-constant expression that's greater than or equal to the array's size or less than 0" (§5.5 the same for vectors, §5.6 for matrices) | indices stay in range |
| built-in domains | §8: `asin`/`acos` "Results are undefined if ∣x∣ > 1", `atan(y, x)` "if x and y are both 0", `pow` "if x < 0 ... if x = 0 and y <= 0", `log`/`log2` "if x <= 0", `sqrt` "if x < 0", `inversesqrt` "if x <= 0", `clamp` "if minVal > maxVal", `smoothstep` "if edge0 >= edge1", `inverse` "if m is singular" | the invariance expressions use `abs(...) + c` to stay inside the domains |
| precision of trigonometric functions | §4.5.1 "Built-in functions not listed above and not defined as equations of the above have undefined precision." | sanity bound only |
| late invariance pragma | §4.6.1 "If this pragma is used after the declaration of any variables or functions, then the set of outputs that behave as invariant is undefined." | the pragma is the first line after `#version` |
| derivatives | §8.9 derivatives are undefined in non-uniform control flow | not used at run time |

## Companion compile/link cases (`shader-tests/`)

These rows also have negative and positive compile cases with `@feature` annotations in
`shader-tests/glsl100`, `glsl300` and `glsl310`. Examples: comments that cannot nest, reserved and
keyword identifiers, the 1024-character identifier limit, `gl_`/`GL_` prefixes, no implicit
conversions, redeclarations and scoping errors, the const rules (including `dFdx` of a constant
being 0 in an initializer), parameter qualifier errors, invariant errors (undeclared, input,
fragment pragma, inside a function, after use, qualifier order), operator type errors, the GLSL
ES 1.00 storage and texture-function rules, and the built-in constants as constant expressions.

## Running

```
./build/debug/bin/test-glsl-language --headless --validate --capture out.png
ctest --preset debug -R test-glsl-language
```

The surface must be at least 144x96 pixels (smaller sizes report SKIP_UNSUPPORTED).

## Pitfalls

- Values must not be constant-folded when the run-time behaviour is the subject. All run-time
  inputs come from uniforms, and the constant-folding case compares a `const` expression with the
  same expression on uniforms.
- RGBA8 would quantize results, and ES 3.0 does not require float colour buffers. Integer
  attachments and transform feedback give exact words in both languages.
- `texture2DProj(sampler2D, vec4)` divides by `.w`, not `.z`. The test puts 5.0 in `.z` to show
  that it is ignored.
- A shader without any `#version` line is GLSL ES 1.00. On ES 3.x the first line of such a
  shader can still be a comment.
- `MESA_GLES_VERSION_OVERRIDE=3.0` only lowers the reported API version. A `310 es` + `100` link
  attempt crashes Mesa 25.0 in that mode, so this test does not link 310 es shaders below ES 3.1.
