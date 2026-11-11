# diagnostics-capabilities

**Minimum:** OpenGL ES 3.0 · **Shaders:** none · **Run:** `diagnostics-capabilities --headless --validate`

## Purpose

Prints what the current OpenGL ES implementation offers (version strings, every implementation
limit, extensions, compressed formats) and checks that the report is self-consistent and meets
the specification. It is also an integration test of meta-gl's loader and capability detection.

## What is checked

1. **Implementation limits.** About 100 `glGet*` limits are compared with the minimum (or, for
   alignments and `MIN_*` offsets, the maximum) required **for the context's version**. The values
   come from ES 3.0 Tables 6.28–6.34 and the ES 3.1 / 3.2 "Implementation Dependent" state tables
   in the last chapter of each specification. Several minimums rise between versions, e.g.
   `MAX_COMBINED_TEXTURE_IMAGE_UNITS` 32 → 48 → 96, `MAX_UNIFORM_BUFFER_BINDINGS` 24 → 36 → 72,
   and `MAX_FRAGMENT_SHADER_STORAGE_BLOCKS` 0 → 0 → 4. A derived rule is also checked:
   `MAX_COMBINED_*_UNIFORM_COMPONENTS ≥ blocks × MAX_UNIFORM_BLOCK_SIZE/4 + components`. On ES 3.1+,
   so are the indexed compute limits read with `glGetIntegeri_v`.
2. **Extensions.** `GL_NUM_EXTENSIONS` must match meta-gl's `Capabilities::extensions`, each
   `glGetStringi(GL_EXTENSIONS, i)` must match entry *i* and satisfy `HasExtension()`, and index
   `NUM_EXTENSIONS` must return `NULL` with `GL_INVALID_VALUE` (ES 3.0 §6.1.6). ES 3.x still defines
   the legacy space-separated `glGetString(GL_EXTENSIONS)`; it must list the same set. meta-gl itself
   uses the indexed form, as recommended for ES 3.x.
3. **Compressed formats.** All 10 ETC2/EAC formats (required since ES 3.0) and, on ES 3.2, all 28
   ASTC LDR formats must appear in `GL_COMPRESSED_TEXTURE_FORMATS`.
4. **meta-gl.** For every command in `inventory/es_core_commands.csv` introduced at or below the
   context's version, `metagl::IsFunctionAvailable()` must be true. On ES 3.2,
   `AllFunctionsLoaded()` must also be true. The `Capabilities` version flags and strings must
   agree with the driver.

## Expected image

Four bars, for ES 2.0, 3.0, 3.1 and 3.2. Each is green when supported and gray when not. On
softpipe (native ES 3.1) the last bar is gray.

## Findings on the reference machine

softpipe advertises ES 3.1 but reports `MAX_SAMPLES = 1` and `MAX_FRAMEBUFFER_SAMPLES = 1`, which is
below the required 4. This is reported as XFAIL for that renderer only (see `docs/KNOWN_LIMITATIONS.md`).
On a context limited by `MESA_GLES_VERSION_OVERRIDE`, Mesa still exposes every ES 3.2 entry point.
The report prints that this is allowed (not required) rather than treating it as an error.

## Pitfalls

- The limits are **minimums**. Portable code must query them, not assume the reference machine's values.
- `UNIFORM_BUFFER_OFFSET_ALIGNMENT` and similar values are *maximum allowed* alignments. An
  implementation may require a smaller one, never a larger one.
- Some limits are 64-bit (`MAX_ELEMENT_INDEX`, `MAX_UNIFORM_BLOCK_SIZE`, `MAX_SHADER_STORAGE_BLOCK_SIZE`).
  Query them with `glGetInteger64v`.
