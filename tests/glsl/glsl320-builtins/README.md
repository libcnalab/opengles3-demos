# test-glsl320-builtins

Checks three GLSL ES 3.20 additions:

- **Image atomics** (GLSL ES 3.20 §8.12): `imageAtomicAdd`, `imageAtomicMin`, `imageAtomicMax`,
  `imageAtomicAnd`, `imageAtomicOr`, `imageAtomicXor`, `imageAtomicExchange` and
  `imageAtomicCompSwap` on `r32ui` and `r32i` images, and `imageAtomicExchange` on an `r32f` image.
  In ES 3.1 these needed `OES_shader_image_atomic`. ES 3.2 makes them core.
- **`fma()`** (§8.3), with the precision rules of §4.7.1.
- **The `precise` qualifier** (§4.9).

Minimum version: OpenGL ES 3.2, `#version 320 es`. A few negative compile checks use
`#version 310 es`. softpipe is a native ES 3.1 driver, so the test reports SKIP_UNSUPPORTED there.

## Method

Everything runs in compute shaders. Results are written to shader storage buffers and read with
`glMapBufferRange` after `glMemoryBarrier`. Output buffers start filled with a sentinel, so a
dispatch that writes nothing cannot pass. The test draws nothing, and the window stays black.

### Image atomics (`shaders/image_atomics.comp`)

Three immutable 4×4 textures (`R32UI`, `R32I`, `R32F`) are uploaded with `glTexSubImage2D` and
bound to image units 0–2 with `glBindImageTexture(..., READ_WRITE, R32UI/R32I/R32F)`. One dispatch
of 1024 invocations (16 groups × 64) runs. **Every invocation applies one atomic of each kind to the
same texel**, so up to 1024 invocations contend for each texel. Each invocation writes the value its
atomic returned (the texel's previous value) to the `Olds` buffer. A second dispatch,
`shaders/image_dump.comp`, runs after `glMemoryBarrier(SHADER_IMAGE_ACCESS_BARRIER_BIT)` and copies
all 48 texels into a buffer with `imageLoad`.

| texel | operation | data per invocation *i* | what is checked |
|---|---|---|---|
| 0 | add | `i % 7 + 1` (r32i: `-(i % 5) - 1`) | final = initial + Σ data; serial order |
| 1 | add | 32-bit random (the sum wraps) | final mod 2³²; serial order |
| 2, 3 | min, max | 32-bit random (r32i: half negative) | final = fold; serial order |
| 4, 5, 6 | and, or, xor | bit masks / random | final = fold; serial order |
| 7 | exchange | distinct values | serial order; returned values plus final value form a permutation of the initial value plus the data |
| 8 | compSwap, one attempt | compare `i % 16`, data `i % 16 + 1` | serial order |
| 9 | compSwap counter | retry loop | every value base … base+1023 is obtained exactly once; final = base + 1024 |
| 10 | invocation 0 only | fixed sequence | every returned value exact |
| 11–15 | never addressed | — | unchanged |

The `r32f` image uses texel 0 for a contended exchange of 1024 distinct floats, texel 1 for a
single-invocation exchange chain 1.5 → −2.75 → FLT_MAX → FLT_MIN → 0.1, and the rest stays untouched.

**What is deterministic.** §8.12: an atomic "read[s] a value from the selected texel, compute[s] a
new value …, write[s] the new value to the selected texel, and return[s] the original value read",
and the texel is "guaranteed not to be modified by any other image store or atomic function between
the time the original value is read and the time the new value is written". Two things follow:

1. For add, min, max, and, or and xor the final value is independent of the order. It is compared
   exactly with the CPU fold of all 1024 data values.
2. For every operation, some serial order of the 1024 atomics must reproduce every returned value
   and the final texel. The values returned by exchange and compSwap depend on the order, so for
   those this is the *only* assertion. The test checks it exactly. Each atomic is an edge
   `old_i → op(old_i, data_i)` in a graph whose vertices are texel values. A serial order exists
   if and only if these edges form one trail from the initial value to the final value that uses
   every edge (an Euler trail). Adding the virtual edge `final → initial` reduces this to the
   Euler-circuit condition: every vertex has equal in- and out-degree, and all edges are connected.
   The check is sound and complete. It does not accept anything a real serial order could not
   produce, and it never rejects a valid one.

The compSwap counter is the classic lock-free increment
(`old = imageAtomicCompSwap(img, P, expected, expected + 1)`, retried with `expected = old`). It
must hand out the values `base … base + 1023` exactly once each. The attempt count is printed as
evidence of real contention (radeonsi: up to ~570 attempts, llvmpipe ~290, zink ~280). The loop is
capped at 8192 attempts so that a broken implementation cannot hang the test.

**Exact sequence (invocation 0).** On `r32ui`: add with wrap-around (`0x0F0F0F0F + 0xF0F0F0F2 = 1`),
add, min/max in *unsigned* order (`0x80000000` vs `0x7FFFFFFF`), and, or, xor, exchange, compSwap
with a different compare (no store, returns the old value), and compSwap with an equal compare
(store). On `r32i`: the same sequence with *signed* order and a signed wrap-around
(`-15 + INT_MIN`; GLSL ES 3.20 §4.1.3 defines overflow as the low-order 32 bits). Every returned
value and the final texel are compared bit for bit.

**Invalid atomics** (ES 3.2 §8.23): "Invalid image atomics will not update any texture bound to the
image unit and will return zero". An access is invalid when "the selected texel … doesn't exist".
Invocation 0 runs add/min/max/exchange/compSwap at `(4,0)`, `(0,4)`, `(-1,1)` and `(4,4)` on all
three images. Each call must return 0 (±0.0 for the float exchange), and texels 11–15 (2–15 for the
float image) must keep their uploaded values.

**Other image types.** §8.12 defines the atomics for every `IMAGE_PARAMS` form. In separate
dispatches, 1024 invocations each run `imageAtomicAdd(img, P, 1u)` on one texel of an `R32UI`
`uimage2DArray` (`ivec3(1,0,2)`), `uimage3D` (`ivec3(0,1,1)`), `uimageCube` (`ivec3(1,1,4)`,
face +Z), `uimageCubeArray` (`ivec3(0,1,9)`: cube 1, face −Y) and `uimageBuffer` (`5`). These are
bound with `layered = TRUE` (the buffer texture with `FALSE`). The coordinate maps to a texel as in
ES 3.2 Table 8.25. A second program reads every texel back with `imageLoad`. The returned values
must be `init … init+1023` once each. The addressed texel ends at `init + 1024`, and every other
texel keeps its uploaded value, which also verifies the coordinate-to-layer/face mapping.

### fma() and precise

Each case is a generated compute shader. One invocation reads one row of four `uvec4` (the a, b, c,
d lanes of four independent inputs as float bit patterns) and writes four `uvec4` of results as bit
patterns.

**The precision rules, read exactly.** §4.7.1 (highp):

- `a + b, a - b, a * b`: "Correctly rounded."
- `a*b+c`: "Correctly rounded single operation or sequence of two correctly rounded operations."
- `fma()`: "Inherited from a * b + c."
- "The rounding mode cannot be set and is undefined but must not affect the result by more than
  1 ULP."

So a "correctly rounded" result is one of the two floats that bracket the exact value. The test
does not assume round-to-nearest-even. For each input the CPU computes the exact value in double
precision (the product of two floats is exact in double, and every sum is verified exact with
TwoSum), then builds the two allowed sets:

- *fused* = {RD, RU}(a·b + c)
- *twice* = {RD, RU}(p + c) for p ∈ {RD, RU}(a·b)

§8.3 on `fma(a, b, c)`: "In uses where the return value is eventually consumed by a variable
declared as precise: fma() is considered a single operation, whereas the expression a * b + c
consumed by a variable declared precise is considered two operations. The precision of fma() can
differ from the precision of the expression a * b + c. fma() will be computed with the same
precision as any other fma() consumed by a precise variable, giving invariant results for the same
input values of a, b, and c."

The spec does **not** require `fma()` to round once; its precision is inherited from `a*b+c`. The
test therefore asserts:

| case | assertion |
|---|---|
| 256 inputs where a·b+c is exactly representable | `precise` vec4 fma, plain fma, `precise` scalar fma and plain a*b+c all return the exact value |
| 512 inputs where *fused* and *twice* are disjoint | every fma result (`precise` vec4, plain vec4, `precise` scalar) and plain `a*b+c` is in *fused* ∪ *twice* |
| same inputs | **invariance**: `precise` vec4 fma and four `precise` scalar fmas give identical bits; a second program (lanes reversed, other surrounding code) gives identical bits; an fma assigned to a `precise` local inside a helper function (the §4.9 `func2` pattern) gives identical bits |

The inputs are chosen so that single and double rounding give different values under every rounding
mode. If one `precise` fma were fused and another were not, the invariance checks would fail. How
the implementation rounds is printed, not asserted. radeonsi rounds `fma()` once (and also contracts
a plain `a*b+c`). llvmpipe and zink (on lavapipe) round twice. Both are conformant.

§4.9 states what `precise` forbids: "a * b + c is not allowed to become a single operation
fma(a, b, c)", "a + (b + c) is not allowed to become (a + b) + c", "a * (b * c) is not allowed to
become (a * b) * c". It also lists identities that must hold: "a * b + c * d = b * a + c* d = d * c +
b * a = <any other mathematically valid combination>". The test asserts:

- **No contraction**: `a * b + c` lies in *twice*, which is disjoint from *fused* for these inputs.
  This is checked for six forms: `precise vec4 v = a*b + c`; `vec4 m = a*b; precise vec4 w = m + c`
  (the §4.9 example marks values "used to compute" a precise variable as precise); a `precise`
  structure variable (`precise PS ps; ps.v = …`); a global redeclared `precise gv;`; four `precise`
  scalars; and a member of a `precise` buffer block. All forms must also return identical bits
  ("operator consistency").
- **Identities**: for 512 inputs, `precise` a·b + c·d, d·c + b·a and b·a + c·d (program A) and the
  scalar c·d + a·b (program B) are bit-identical and lie in the set
  {RD, RU}(P₁ + P₂), P₁ ∈ {RD, RU}(a·b), P₂ ∈ {RD, RU}(c·d). The inputs make c·d ≈ −a·b, so
  contracting either product into an fma lands outside that set.
- **No reassociation**: `precise a + (b + c)` with (2²⁴, 1, 1) must be 2²⁴ + 2, not 2²⁴, and
  `precise a * (b * c)` with (2¹⁰⁰, 2¹⁰⁰, 2⁻¹⁰⁰) must be 2¹⁰⁰, not ∞. Two more lanes of each kind
  are included. The inner and outer operations are exact, so no rounding mode matters.
  (§5.12 allows exactly these regroupings when `precise` is absent.)

A caveat: §4.7 also lets implementations "calculate and/or store any results at higher precisions".
A hypothetical implementation that kept the product of a `precise a*b + c` in higher precision would
still produce a two-operation sequence, but its value would equal the fused one. No such
implementation was found, and the strict reading of §4.9 together with §4.7.1 ("a * b: correctly
rounded" for highp) is kept.

### Compile checks

Each negative check has a positive control that differs only in the tested construct:

- `#version 310 es`: `imageAtomicAdd`, `fma` and `precise` are rejected. GLSL ES 3.10 has neither
  built-in, and §3.7 lists `precise` as a reserved keyword.
- `#version 320 es`: `imageAtomicAdd(uimage2D, ivec2, 1)` with an `int` literal has no overload,
  because GLSL ES has no implicit int → uint conversion. `imageAtomicAdd` on an `r32f` `image2D`
  with float data does not exist, because §8.12 allows float only for exchange.
- `#version 320 es`: `precise float f(...)` is an error by §6.1, "Only a precision qualifier is
  allowed on the return type of a function". The control uses `highp float f(...)`.

## Negative controls

- The serial-order checker rejects a **lost update**: the GPU's imageAtomicAdd results are modified
  so that two invocations returned the same value, as a non-atomic read-modify-write would.
- It rejects an **exchange that returns the new value** instead of the old one.
- The counter check rejects a **duplicated** compSwap result.
- `r32i` min/max folded with **unsigned** order disagrees with the GPU's final texels.
- Sequence models with compSwap **compare/data swapped**, or with **unsigned** min/max on `r32i`,
  disagree with the GPU.
- For every discriminating fma input, the single-rounding value lies outside the `precise a*b+c`
  set. For every a·b+c·d input, `fma(a, b, c·d)` lies outside the allowed set. A contracting
  compiler would therefore fail. (The inputs come from 853 and 1287 seeded candidates.)
- For the reassociation inputs, `(a + b) + c` and `(a * b) * c` evaluated in float on the CPU differ
  from the expected values.

## Expected result

The window is cleared to black; this test produces no image. All 83 checks pass (`--validate`).

## Run

```
./build/debug/bin/test-glsl320-builtins --headless --validate
ctest --preset debug -R test-glsl320-builtins
```

## Observations (Mesa 25.0.7)

radeonsi, llvmpipe and zink pass every check, with one documented deviation. **Mesa's GLSL parser
rejects `precise` as an interface-block qualifier.**
`layout(std430, binding = 2) precise writeonly buffer PreciseOut { vec4 pv[]; };` (and the order
`precise layout(...) buffer`) fails with "syntax error, unexpected NEW_IDENTIFIER" at the block
name. GLSL ES 3.20 §4.3.9 says the block's type-qualifier "may optionally include layout qualifiers,
the auxiliary storage qualifier patch, and the precise qualifier", and §4.9 says "When applied to a
block …, precise applies to each contained member". The check stays strict and is recorded with
`expect_or_known_deviation(..., "Mesa", ...)`. Its value checks run only when the block compiles.
`precise` on a block *member* is accepted by Mesa, but §4.3.9 does not list `precise` among the
member qualifiers, so the test does not use it as a substitute.

## Pitfalls

- An `r32ui`/`r32i`/`r32f` image used for atomics must not be `readonly` or `writeonly`. Only these
  three formats may omit the memory qualifier (§4.10).
- `imageAtomicAdd(uimg, P, 1)` does not compile; write `1u`.
- `glBindImageTexture` requires an immutable texture (`glTexStorage*`) or a buffer texture.
  Array, 3D and cube textures need `layered = TRUE` to be addressed with a 3-component coordinate.
- The image writes of one dispatch are visible to `imageLoad` in the next dispatch only after
  `glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)`.
- Without `precise`, `a*b+c` may be contracted into an fma (radeonsi does this), and
  `a + (b + c)` may be reassociated (§5.12). Use `precise` where consistency matters.
- `precise` binds to the variable that *consumes* the value. Values computed in a helper function
  and returned are not precise unless the helper itself assigns them to a precise variable (§4.9
  `func` vs `func2`).
- A spin loop around `imageAtomicCompSwap` is lock-free: every round, some invocation succeeds.
  Cap it anyway.
