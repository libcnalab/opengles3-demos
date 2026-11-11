Backlog item 62: `tests/glsl/glsl320-builtins/`, executable `test-glsl320-builtins`, min_es 3.2.
Model on `tests/glsl/glsl300-builtins/` (and `tests/glsl/glsl310-builtins/` if it exists on main),
using a compute shader writing to an SSBO. Features: GLSL320.FN.IMAGE_ATOMIC, GLSL320.FN.FMA,
GLSL320.PRECISE (check inventory/glsl_features.csv).
- imageAtomicAdd/Min/Max/And/Or/Xor/Exchange/CompSwap on r32i/r32ui images, plus imageAtomicExchange on
  r32f (GLSL ES 3.20 §8.12) — exact results with many invocations contending (final value and the set of
  returned old values must be consistent with some serialization; check what is deterministic).
- fma(a,b,c) (§8.3): GLSL ES 3.20 says fma is computed as a single operation with one rounding when
  `precise`/used for precise — verify the exact precision rule (§4.7.1) and test with inputs where
  a*b+c with two roundings differs from the single-rounding result; assert only what the spec requires.
- `precise` qualifier (§4.9): verify invariance-related guarantees that can be observed (e.g. the same
  precise expression in two shaders yields bit-identical results); do not invent requirements.
- Negative controls.
