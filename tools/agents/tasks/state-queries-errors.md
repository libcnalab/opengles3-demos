Backlog item 18: `demos/es20-foundations/state-queries-errors/`, executable `es20-state-queries-errors`,
min_es 3.0 (ES 2.0 feature set on an ES 3.x context), `#version 300 es` shaders if any.
Features: ES20.STATE.QUERIES, ES20.ERRORS.* (GETERROR, INVALID_ENUM, INVALID_VALUE, INVALID_OPERATION,
INVALID_FRAMEBUFFER_OPERATION, OUT_OF_MEMORY is NOT testable — do not claim it), ES20.HINT, ES20.DITHER,
ES30.STATE.INDEXED (check exact IDs in inventory/features.csv).
- Initial-state tables: verify the initial values of the ES 3.0 state tables (Tables 6.x, chapter 6)
  for a representative, broad set of state via glGetBooleanv/Integerv/Integer64v/Floatv, glIsEnabled,
  including the type-conversion rules of ES 3.0 §6.1.2 (boolean<->int<->float conversions, color and
  depth values scaled when queried as integers — quote the rule).
- Indexed state (glGetIntegeri_v / glGetInteger64i_v) for UNIFORM_BUFFER_BINDING/START/SIZE and
  TRANSFORM_FEEDBACK_BUFFER_* with INVALID_VALUE for an out-of-range index.
- glHint(GENERATE_MIPMAP_HINT, FRAGMENT_SHADER_DERIVATIVE_HINT) round trip + INVALID_ENUM.
- DITHER initial TRUE, toggling.
- Error semantics (ES 3.0 §2.5): the error flag is sticky until glGetError; a command generating an error
  has no effect (verify state unchanged for several cases); INVALID_FRAMEBUFFER_OPERATION for drawing to
  an incomplete FBO.
- A picture that fills the window at any size (e.g. a state "dashboard" grid where each cell is green/red
  drawn from the check results — but the checks themselves must be the GL queries, not the picture).
