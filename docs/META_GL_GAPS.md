# meta-gl compatibility report

meta-gl revision audited: `develop` @ `795bcdc` (v0.4.1). Fixes live on the local meta-gl branch
`opengles3-demos/enum-gaps` (commits `146e171`, G4/G5 fix commit after it; not pushed). The owner authorized meta-gl changes on 2026-11-11.

## Audit method

1. **Entry points** — `tools/registry/extract_es_inventory.py` derives the ES core command list
   from Khronos `gl.xml` (`<feature api="gles2">` GL_ES_VERSION_2_0..3_2); the coverage
   generator compares it with every `METAGL_API ... gl*(` declaration in `Functions.hpp`.
2. **Enum domains** — each meta-gl `enum class` was compared with the matching `gl.xml` group
   (GetPName, BufferTargetARB, EnableCap, ...), restricted to ES core enums; every candidate
   gap was then checked against the normative spec text (registry groups also contain desktop
   and ES 1.x values, so they are only a lead).
3. **Runtime** — every demo exercises the wrapper → loader → driver path on radeonsi and
   llvmpipe, at the demo's minimum ES version and at ES 3.2.

## Result summary

| Check | Result |
|---|---|
| Core commands in gl.xml (2.0/3.0/3.1/3.2) | 142 / 104 / 68 / 44 = 358 |
| Wrapped by meta-gl | 358 / 358 — no missing, no non-core extras |
| Core enums not referenced anywhere in Enums.hpp | 16, of which 2 were real gaps (G3); the rest are result values (`GL_TRUE`, `GL_SIGNALED`, `GL_INVALID_INDEX`, ...) used as raw constants |
| Enum-domain gaps confirmed by the spec | G1–G4 below (fixed) |

## Confirmed gaps

### G1 — `GetParameter` lacked pixel-store pnames — FIXED (meta-gl `146e171`)
- **Affected:** `metagl::glGetIntegerv(GetParameter, GLint*)` and friends.
- **Expected:** `PACK_ALIGNMENT`, `UNPACK_ALIGNMENT` (ES 2.0) and `PACK_ROW_LENGTH`,
  `PACK_SKIP_ROWS`, `PACK_SKIP_PIXELS`, `UNPACK_ROW_LENGTH`, `UNPACK_IMAGE_HEIGHT`,
  `UNPACK_SKIP_ROWS`, `UNPACK_SKIP_PIXELS`, `UNPACK_SKIP_IMAGES` (ES 3.0) are queryable with
  `GetIntegerv` (ES 3.0 §3.7.1 + state table "Pixels": *UNPACK_ALIGNMENT Z+ GetIntegerv 4*).
- **Actual:** no enumerator existed; the state could be set (`PixelStoreParam`) but not read.
- **Location:** meta-gl enum domain (not driver, not demo).
- **Fix:** added `GetParameter::PackAlignment ... UnpackSkipImages`.

### G2 — `GetParameter` lacked hint and enable-flag pnames — FIXED (`146e171`)
- **Expected:** `GENERATE_MIPMAP_HINT`, `FRAGMENT_SHADER_DERIVATIVE_HINT` are `GetIntegerv`
  state (ES 3.0 §5.3, state table "Hints"); every enable flag (`BLEND`, `CULL_FACE`, `DEPTH_TEST`,
  `DITHER`, `POLYGON_OFFSET_FILL`, `SCISSOR_TEST`, `STENCIL_TEST`, `SAMPLE_ALPHA_TO_COVERAGE`,
  `SAMPLE_COVERAGE`, `RASTERIZER_DISCARD`) may also be queried with `GetBooleanv` (ES 3.0 §6.1.1).
- **Actual:** hint values could be set but never read back; enable flags only via `glIsEnabled`.
- **Fix:** added `GetParameter::GenerateMipmapHint`, `FragmentShaderDerivativeHint`, `Blend`, ...

### G3 — ES 3.2 texture-buffer state missing — FIXED (`146e171`)
- **Expected:** `TEXTURE_BUFFER` is a buffer binding target (ES 3.2 Table 6.1 "Texture data
  buffer, §8.9"); `TEXTURE_BUFFER_BINDING` and `TEXTURE_BUFFER_OFFSET_ALIGNMENT` are `GetIntegerv`
  state; `glTexBufferRange` requires `offset` to be a multiple of the latter (ES 3.2 §8.9).
- **Actual:** `BufferTarget` had no `TextureBuffer`, so the canonical
  `glBindBuffer(GL_TEXTURE_BUFFER, b); glBufferData(GL_TEXTURE_BUFFER, ...)` sequence was not
  expressible, and the alignment needed for a correct `glTexBufferRange` could not be queried.
- **Fix:** added `BufferTarget::TextureBuffer`, `GetParameter::TextureBufferBinding`,
  `GetParameter::TextureBufferOffsetAlignment`.

### G4 — default-framebuffer attachments could not be queried — FIXED
- **Affected:** `metagl::glGetFramebufferAttachmentParameteriv`.
- **Expected:** "If the default framebuffer is bound to target, then attachment must be BACK, identifying
  the color buffer; DEPTH ...; or STENCIL" (ES 3.0 §6.1.13; ES 3.2 §9.2.3).
- **Actual:** `FramebufferAttachment` holds only `*_ATTACHMENT` points (+ColorAttachment conversions);
  `DefaultFramebufferAttachment` holds `GL_COLOR` (invalidation only). `GL_BACK` was not expressible.
- **Found by:** the es20-blending work (it wanted the pbuffer's color encoding).
- **Fix:** `enum class DefaultFramebufferQueryAttachment { Back, Depth, Stencil }` + an overload of
  `glGetFramebufferAttachmentParameteriv`. Runtime evidence: `diagnostics-capabilities` queries
  OBJECT_TYPE (= FRAMEBUFFER_DEFAULT), sizes and COLOR_ENCODING of the pbuffer on all drivers.

### G5 — `to_string(DefaultFramebufferAttachment)` missing — FIXED
- meta-gl's README promises `to_string()` for all enum classes; this one had none. Added (same commit).

### G6 — multisample level queries missing from `TextureLevelParameter` — FIXED (meta-gl `0a7766c`)
- **Expected:** `TEXTURE_SAMPLES` and `TEXTURE_FIXED_SAMPLE_LOCATIONS` are `glGetTexLevelParameter*` pnames for
  multisample textures (ES 3.1/3.2 §8.11.3, Table 21.11).
- **Actual:** they existed only in `TextureParameterQuery` (for `glGetTexParameter*`, where the spec does not
  allow them), so the correct query was not expressible. Found by the es31-multisample-textures work.
- **Fix:** added `TextureLevelParameter::Samples` / `FixedSampleLocations` (additive).

### G7 — buffer-texture level queries missing from `TextureLevelParameter` — FIXED (meta-gl `a451376`)
- **Expected:** `TEXTURE_BUFFER_DATA_STORE_BINDING`, `TEXTURE_BUFFER_OFFSET`, `TEXTURE_BUFFER_SIZE` are
  `glGetTexLevelParameteriv` state (ES 3.2 Table 21.11).
- **Actual:** only in `TextureParameterQuery`/`TextureParameter` (the `glGetTexParameter*` domain, where §8.11.2
  does not allow them). Found by the es32-texture-buffer work.
- **Fix:** added to `TextureLevelParameter`; `es32-texture-buffer` now asserts all three.
- **Follow-up (not changed, would be breaking):** `TextureParameterQuery` still contains these and the
  multisample pnames, which ES does not accept for `glGetTexParameter*`.

### Note — arbitrary sample masks
`SampleMaskValue` (for `glSampleMaski`) has only `None` and `All`. Since it is a registered 32-bit bitfield
(`all_bits = 0xFFFFFFFF`), a chosen mask is formed with `static_cast<SampleMaskValue>(bits)`. Every value is
valid there, so the demo isolates that one cast in a helper. A `metagl::sample_mask(GLbitfield)` helper would
be a small additive improvement.

## Observations (not defects; documented for users of the demos)

- **O1 — invalid enum values are unreachable by design.** Typed enums make `GL_INVALID_ENUM`
  impossible to trigger without `static_cast`ing a raw value. Negative tests that need it do so
  explicitly and are confined to `tests/negative/`.
- **O2 — contract violations terminate.** Every wrapper calls `std::terminate()` when meta-gl is
  uninitialized or the entry point was not loaded, and several helpers terminate on invalid
  sizes (e.g. `glUniformMatrix` with `transpose != GL_FALSE`). Tests that probe these contracts
  must run in a separate process and expect abnormal termination (`WILL_FAIL`/death tests).
- **O3 — per-call error checking.** `METAGL_ENABLE_DEBUG_LOGGING` inserts `glGetError` after
  every call and would consume error flags; this suite forces it OFF.
- **O4 — `SamplerParameter::MaxAnisotropy`** is an `EXT_texture_filter_anisotropic` value inside a
  core enum. Core demos do not use it (extension policy).
- **O5 — `glGetActiveAttrib(ProgramId, AttribLocation, ...)` convenience overload** converts an
  attribute *location* into an active-attribute *index*. These are different namespaces in ES
  (index ∈ [0, ACTIVE_ATTRIBUTES), location is assigned by the linker). Demos use the
  `ActiveAttribIndex` overload. Candidate for deprecation in meta-gl.
- **O6 — Mesa context versions.** Mesa returns its highest ES version (3.2) even when 3.0 is
  requested; meta-gl correctly reports what the driver says. Minimum-version testing uses
  `MESA_GLES_VERSION_OVERRIDE` (driver behavior, not a meta-gl issue).

- **O6b — ergonomics noted by demo authors (not defects):** `glReadPixels(..., void* pixels)` has no
  buffer-offset overload. With a PIXEL_PACK_BUFFER bound, the pointer is a byte offset, so demos cast
  an integer to `void*`. There are no typed enums for *returned* values such as
  `FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE`, `COMPONENT_TYPE` or `COLOR_ENCODING`, or for the queried
  `IMPLEMENTATION_COLOR_READ_FORMAT/TYPE`, so demos compare them with `GL_*` macros. Candidates for
  small additive helpers in meta-gl.

- **O7 — debug-output ergonomics (from es32-debug-output, not defects):** the `GLDEBUGPROC` callback receives raw
  `GLenum`s (demos `static_cast` to `DebugSource/Type/Severity`); `glGetPointerv(DEBUG_CALLBACK_FUNCTION)` returns
  `void*` (copied into a `GLDEBUGPROC` with `memcpy`); no `std::string_view` overloads for
  `glDebugMessageInsert`/`glPushDebugGroup`/`glObjectLabel`. Candidate additive helpers.

## Open questions

- None yet. New findings are appended here with the same fields:
  affected function/enum, expected, actual, reproduction, spec evidence, location
  (meta-gl / driver / platform / demo), fix.
