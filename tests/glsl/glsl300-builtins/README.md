# test-glsl300-builtins

Numerical test of the GLSL ES 3.00 built-in functions (§8.1–§8.7: trigonometry, exponential,
common, packing, geometric, matrix and vector-relational) and of the integer operators.

## Method

ES 3.0 does not require float render targets, and an RGBA8 framebuffer would quantize results.
So each case is a vertex-shader expression whose outputs (four `vec4`/`ivec4`/`uvec4`) are
**captured with transform feedback** as exact 32-bit values. The draw runs with
`RASTERIZER_DISCARD`, the values are read back with `glMapBufferRange` and compared with a CPU
reference. Each case is its own program, linked after `glTransformFeedbackVaryings`.

## Tolerances (GLSL ES 3.00 §4.5.1, highp)

| class | used for | bound |
|---|---|---|
| Exact | abs, sign, floor/trunc/ceil/fract, roundEven, round (no .5 ties), min/max/clamp, modf, step, bit casts, packing, vector relational, dot/cross/outerProduct/matrixCompMult/transpose on exact inputs, all integer cases | bit-exact |
| ULP | radians/degrees (2), inversesqrt (2), sqrt (5 = 1/inversesqrt), exp/exp2 (3+2·abs(x)), log/log2 (3 outside [0.5,2], abs 2⁻²¹ inside), mix (2), unpack*Norm (3) | from the precision table |
| Abs/Rel | mod (division inside), smoothstep, pow, length/normalize/distance, reflect/refract | derived from the operations they are defined by |
| Quality | sin…atanh, determinant, inverse | precision is **undefined** by the spec. Only a sanity bound of 1e-3 is asserted (it catches a wrong function); the measured error is printed |

Undefined cases (`asin(2)`, `log(0)`, `%` with negative operands, `round(x.5)`, shifts by ≥ 32) are not used.

## Observations (Mesa 25.0.7)

All drivers produce the same results. `asin`/`acos` have absolute errors up to 3.5×10⁻⁴ (a
polynomial approximation in NIR), allowed because the spec leaves their precision undefined.
Everything else is within a few ULP or exact.
