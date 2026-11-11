# test-glsl310-builtins

Checks the built-in functions that GLSL ES 3.10 added on top of 3.00, and arrays of arrays:

- **Integer functions** (GLSL ES 3.10 §8.8): `uaddCarry`, `usubBorrow`, `umulExtended`,
  `imulExtended`, `bitfieldExtract` (signed and unsigned), `bitfieldInsert`, `bitfieldReverse`,
  `bitCount`, `findLSB`, `findMSB`.
- **frexp / ldexp** (§8.3).
- **4x8 packing** (§8.4): `packUnorm4x8`, `packSnorm4x8`, `unpackUnorm4x8`, `unpackSnorm4x8`.
- **Arrays of arrays** (§4.1.9).

Minimum version: OpenGL ES 3.1, `#version 310 es` (one negative compile check uses `#version 300 es`).
Besides the usual matrix, the test also runs on softpipe, which is a native ES 3.1 driver.

## Method

ES 3.1 has compute shaders and shader storage buffers, so this test does not need the transform
feedback set-up that `test-glsl300-builtins` uses. Each case is a small compute shader generated in
`main.cpp`:

```glsl
layout(std430, binding = 0) readonly buffer Inputs { uvec4 inp[]; };
layout(std430, binding = 1) writeonly buffer Outputs { uvec4 outp[]; };
// one invocation per row: u0..u3 = inp[4*row .. 4*row+3]; body writes r0..r3
```

Every invocation reads one row of four `uvec4` and writes four `uvec4`. Floats are passed as bit
patterns (`uintBitsToFloat` / `floatBitsToUint`), so values reach the shader and come back
unquantized. The host calls `glDispatchCompute`, then `glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT)`
(ES 3.2 §7.11.2: needed before `glMapBufferRange` sees the shader writes), maps the output buffer
and compares each word with a CPU reference. Output buffers start filled with a sentinel, so a
dispatch that writes nothing cannot pass.

Inputs are a deterministic list of edge cases plus a random set from `std::mt19937` with the
fixed seed `0x310B17`. The raw generator output is used directly, so the set is the same on every
platform.

## What is exact and why

| functions | comparison | spec basis |
|---|---|---|
| all integer functions | bit-exact | integers are exact two's complement (§4.7.1); every result is defined by the §8.8 text |
| integer functions with literal arguments | bit-exact | §4.3.3: built-in calls with constant arguments (and no `out` parameter) are constant expressions; they are used as `const` initializers, so the compiler's constant folding is tested separately from run-time evaluation |
| `frexp`, `ldexp`, `ldexp(frexp(x))` | bit-exact, +0 and −0 interchangeable | §8.3 defines both exactly for zero and normal values; §4.7.1 lets zeros swap sign |
| `packUnorm4x8`, `packSnorm4x8` | bit-exact | §8.4 formula `round(clamp(c, 0, 1) * 255.0)` / `round(clamp(c, -1, 1) * 127.0)`; inputs keep away from .5 ties (see below) |
| `unpackUnorm4x8`, `unpackSnorm4x8` | relative 2⁻¹⁰ | they return `mediump vec4` and contain a division (`f / 255.0`): the result does not have to be exact, and 2⁻¹⁰ is the minimum mediump precision of §4.7.1 |
| `packXnorm4x8(unpackXnorm4x8(x))` | bit-exact | the unpack error is at most 0.25 of a step, so packing again must give back the byte. Snorm `0x80` gives `0x81`, because −128/127 clamps to −1 |

**Defined edge cases that are tested:**

- `bitfieldExtract` / `bitfieldInsert` with `offset + bits == 32`, with `bits == 0` (result 0, or
  `base` for insert), and with `offset == 32, bits == 0`. Signed extracts copy bit
  `offset + bits − 1` into all higher bits. The spec text says `offset+base-1`, which is a typo.
- `findLSB(0) == -1`, `findMSB(0) == findMSB(-1) == -1`. For negative values `findMSB` returns the
  most significant *zero* bit, so `findMSB(INT_MIN) == 30`.
- `uaddCarry` / `usubBorrow` over every ordered pair of 16 edge values (0, 1, 2^31 − 1, 2^31,
  0xFFFFFFFF, …) and 256 random pairs.
- `frexp` of ±0 (significand and exponent 0), `FLT_MIN`, `FLT_MAX`, `nextafter(1, 0)` and 1024
  random normal values of both signs. The significand lies in [0.5, 1).
- `ldexp` with exponent 128 (`ldexp(0.75, 128) = 1.5 · 2^127`) and −126, and with random pairs
  whose result is a normal float.

**Undefined cases that are avoided:** negative `offset`/`bits` and `offset + bits > 32`; `frexp` of
infinity or NaN; subnormal inputs, which §4.7.1 allows to be flushed; `ldexp` exponents above 128
(undefined) or below −126 (may be flushed to zero).

**Pack inputs and rounding.** `round()` may break a .5 tie in either direction (§8.3), and the
pack functions take `mediump vec4`. The inputs are therefore multiples of 2⁻¹⁰, which are exact in
any mediump format inside [−1, 1]. An input is kept only if
`|p − round(p)| + 2⁻¹⁰·|p| < 0.5` holds for both `p = clamp(v,0,1)·255` and `p = clamp(v,−1,1)·127`.
Then no tie can be reached even if the product is evaluated at mediump precision, so the packed byte
is unique. Values such as 0.5 (127.5 and 63.5) are excluded, and a check confirms that the filter
excludes them.

## Arrays of arrays (`shaders/arrays_of_arrays.comp`)

One invocation writes one `int` per case into `res[]`. `main.cpp` holds the label and the
expected value of each index. Covered:

- the three spellings `float a[2][3]`, `float[3] b[2]`, `float[2][3] c` (all "2 arrays of 3"),
  `vec4[2] w[3]` and the three-level `int d[2][3][4]`. Every dimension's `length()` is checked, and
  `b[1][2]` / `c[1][2]` are read, which is only in range with the [2][3] orientation;
- `length()` as a constant expression: `int q[d[0][0].length() * 2]` has length 8;
- initializers `int[2][3](int[3](…), …)`, outer size taken from the initializer (`int m[][3]`), and
  the fully unsized constructor `int[][](int[](…), …)`;
- whole-array assignment and `==` / `!=` (§5.7), a constructor indexed directly;
- functions: an array of arrays passed by value, as an `inout int[2][3]` parameter, and as a
  return type (`int[3][2] transpose23(int x[2][3])`);
- a structure containing an array, used in an array of arrays (`Cell cells[2][2]`);
- run-time indices (from the SSBO) for reads and an l-value write;
- shader storage block members: `int src[2][3][4]` written by the CPU and read by the shader, and
  `uint grid[3][4]`, `uvec2 pairs[2][2]`, `uint rt[][3]` written by the shader and read back by the
  CPU in std430 layout with the inner index fastest. `rt` is sized at run time:
  `rt.length() == 5` for a 140-byte buffer (80 bytes of fixed members + 5 × 12).

Negative compile checks, each with a one-dimensional positive control: `float a[2][3]` in a
`#version 300 es` shader (GLSL ES 3.00 §4.1.9: "It is an error to declare arrays of arrays"), and a
fragment output `vec4 o[2][2]` and vertex output `float v[2][2]` in `#version 310 es` (§4.3.6:
compile-time errors).

## Negative controls

Each control runs a plausible wrong CPU model against the GPU output of a passing case and
requires that they disagree somewhere. This shows that the exact comparison would detect the
mistake:

- `packSnorm4x8` / `packUnorm4x8` with truncation instead of `round()`;
- `packUnorm4x8` scaled by 256, `packSnorm4x8` scaled by 128;
- signed `bitfieldExtract` without sign extension;
- `findMSB(int)` that treats negative values as unsigned;
- `frexp` with a significand in [1, 2);
- `unpackUnorm4x8` modelled as `byte / 256` falls outside the 2⁻¹⁰ tolerance for all 255 nonzero
  bytes, so the tolerance is tight enough to reject a wrong formula;
- reading the shader-written `grid[3][4]` with the outer index fastest.

## Expected result

The window is cleared to black; this test produces no image. All checks pass (`--validate`).

## Run

```
./build/debug/bin/test-glsl310-builtins --headless --validate
ctest --preset debug -R test-glsl310-builtins
```

## Observations (Mesa 25.0.7)

radeonsi, llvmpipe, zink and softpipe all pass every check, including the folded constant
expressions and the `offset == 32, bits == 0` edge case. The `unpack*4x8` results are within
about 1 ULP of the exact quotient (measured maximum relative error ≈ 1e-7), far inside the mediump
tolerance.

## Pitfalls

- `float[3] b[2]` is *2* arrays of 3, not 3 arrays of 2: the type-first part names the inner size.
- `bitfieldExtract(x, 32, 0)` is defined (result 0). Implementations that compute
  `x >> offset` without special-casing `bits == 0` can get it wrong.
- `findMSB(-1)` is −1, not 31, and `findMSB(INT_MIN)` is 30.
- `packSnorm4x8(-1.0)` is 0x81 (−127), never 0x80. `unpackSnorm4x8(0x80)` is −1.0 only because of
  the clamp.
- `bitCount`, `findLSB` and `findMSB` return `lowp int`. The test assigns them to `highp` variables
  before converting to `uint`, so a 9-bit lowp cannot affect the bit pattern.
- Shader writes reach `glMapBufferRange` only after `glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT)`.
