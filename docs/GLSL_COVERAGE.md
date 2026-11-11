# GLSL ES coverage strategy

Live status: **[generated/GLSL_STATUS.md](generated/GLSL_STATUS.md)**. Inventory:
`inventory/glsl_features.csv` (separate rows for GLSL ES 1.00, 3.00, 3.10, 3.20).

## Strategies (column `strategy`)

| strategy | how it is verified |
|---|---|
| `compile+render` / `render` | shader used in a validated draw; output checked against a CPU reference |
| `compute-check` | the expression is evaluated on the GPU and compared numerically with a CPU evaluation |
| `compile` | positive compile/link test in `shader-tests/` (no runtime semantics to observe) |
| `negative` | the shader must **fail** to compile/link; the test passes only when failure is observed |
| `doc` | undefined / implementation-dependent results; documented, never asserted |

### How "compute-check" works per version

- **GLSL ES 3.00 (ES 3.0 contexts):** float render targets are not required to be renderable in
  ES 3.0, and RGBA8 would quantize results. Built-in results are therefore computed in the
  **vertex shader and captured with transform feedback** (exact 32-bit floats/ints), then compared
  on the CPU. Fragment-only built-ins (derivatives, `gl_FragCoord`, texture lookups with implicit
  LOD) are checked through rendering with appropriate tolerances.
- **GLSL ES 3.10 / 3.20:** compute shaders write results to an SSBO, read back with
  `glMapBufferRange`.

### Overload strategy for built-in functions

Full overload enumeration (genType = float, vec2, vec3, vec4; genIType; genUType; genBType)
would multiply test count without adding information about the implementation — overloads are
component-wise. The selected strategy is:

1. every function is tested with **scalar and vec4** forms (vec4 covers component-wise lanes);
2. functions whose semantics differ per type family (`abs`, `sign`, `min`, `max`, `clamp`,
   `mix` with `bvec`, integer `/`, `%`, shifts) are tested for **each** type family
   (float / int / uint / bool where defined);
3. edge cases from the spec are tested explicitly where the result is defined (`clamp` with
   min == max, `step` at the edge, `mod` with negative operands, `roundEven` at .5, packing
   saturation and rounding, `findMSB` of 0 and -1, `bitfieldExtract` with bits = 0, ...);
4. **undefined** cases (e.g. `asin(2.0)`, `pow(-1.0, 0.5)`, `log(0.0)`, out-of-range shifts) are
   listed in the test source comments and *not* asserted;
5. tolerances follow the spec precision tables (GLSL ES 3.00 §4.5.1 "Range and Precision"):
   results are checked in `highp` with relative error bounds per function family.

### Section numbering

GLSL ES 3.00 and 3.10/3.20 number some chapters differently (3.10 moved layout qualifiers to §4.4).
Each inventory row cites the numbering of its own language version.

## Shader version rules enforced by review

- `#version 300 es` must be the first line; no `attribute`/`varying`/`gl_FragColor`.
- Features introduced in 3.10 (compute, SSBO, images, atomics, `layout(binding)`, uniform
  locations, arrays of arrays, gather) never appear in a 300 es shader; 3.20 features (geometry,
  tessellation, sample variables, `interpolateAt*`, image atomics, buffer/cube-array samplers)
  never appear in 310 es shaders.
- The `es-min` CTest runs give these rules teeth: on a context forced to ES 3.0 the driver reports
  GLSL ES 3.00 and rejects newer shaders.
