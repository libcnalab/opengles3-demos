# es31-multisample-textures

**API introduced in:** OpenGL ES 3.1 · **Minimum context:** ES 3.1 · **Shaders:** GLSL ES 3.10

## Purpose

A multisample texture keeps several samples per texel, can be rendered into like a multisample renderbuffer, and,
unlike a renderbuffer, can be read by a shader sample by sample with `texelFetch(sampler2DMS, texel, sample)`. This demo
allocates multisample textures with `glTexStorage2DMultisample`, gives every sample a known content, reads every sample
of every texel back and compares it with a CPU model. Around that it checks the sample mask (`glSampleMaski` with
`SAMPLE_MASK`), `glGetMultisamplefv`, `textureSize` on a multisample sampler, the limits and internal-format queries
that describe multisampling, and the errors the specification defines.

The positions of the samples and the choice of samples that a triangle edge covers are implementation-defined, so the
demo asserts nothing about them. Instead it gives each sample its own content with the sample mask (below), which
works whatever the positions are.

## Expected image

The 256 x 256 window has one 128 x 128 cell per multisample texture, from the top left: RGBA8 allocated with
`fixedsamplelocations = TRUE`, RGBA8 with `FALSE`, RGBA8UI (`usampler2DMS`) and RGBA8I (`isampler2DMS`). Each cell is a
grid of sub-cells, one per sample (a 2 x 2 grid for 4 samples, 3 x 3 for 8), and each sub-cell shows the 16 x 12 texels
of that sample stretched to fill it. Red grows to the right and green upwards inside a sub-cell, blue and the position of
the diagonal seam (where an 8-bit channel wraps around) change from one sample to the next, so the four sub-cells of a
cell are clearly different. The four cells look alike because all four textures receive the same pattern numbers; the
integer cells are converted back to colors by the display shader.

## Key ideas

- **Multisample textures** are 2D (`TEXTURE_2D_MULTISAMPLE`), have exactly one level, and are immutable: the only way to
  give them storage is `glTexStorage2DMultisample(target, samples, internalformat, width, height,
  fixedsamplelocations)`. `samples` is a minimum: the texture gets a count that is at least the request and at most the
  next larger count the implementation supports (the Mesa drivers of the test machine return the request when it is
  one of the reported counts). Sampler state (`TEXTURE_MIN_FILTER`, wrap modes, ...) does not exist for these targets;
  setting it is `INVALID_ENUM`. `glTexImage2D`, `glTexStorage2D` and `glGenerateMipmap` do not accept the target.
- **Rendering into one.** `glFramebufferTexture2D(..., GL_TEXTURE_2D_MULTISAMPLE, texture, 0)` attaches it (level must be
  0). A framebuffer with such an attachment is multisampled: `SAMPLE_BUFFERS` is 1 and `SAMPLES` is the texture's sample
  count. `glReadPixels` cannot read a multisample framebuffer, so the demo reads samples through a shader instead.
- **Reading a sample.** `texelFetch(sampler2DMS s, ivec2 texel, int sample)` returns one sample, unfiltered; `isampler2DMS` and
  `usampler2DMS` return `ivec4` and `uvec4`. A sample number that does not exist is undefined, so the demo reads only
  samples `0 .. SAMPLES-1`. `textureSize(sampler2DMS)` takes no level argument. GLSL ES 3.10 gives these samplers no default
  precision, so the shaders declare them `highp`.
- **The fetch shader** writes sample `x / W` of texel `(x % W, y)` to pixel `(x, y)` of a single-sample target, so the samples
  of the whole texture end up side by side as strips that `glReadPixels` can read.
- **A known content per sample.** Pattern number *k* is a color that depends on the pixel position and on *k*
  (`pattern.frag`, with the same arithmetic in C++). Drawing pattern *k* over the whole texture while the sample mask
  lets only sample *k* through writes sample *k* of every texel and nothing else. Doing this for every *k* gives each
  sample a different known value, independent of where the samples lie inside the pixel. Reading sample *k* back must give
  pattern *k* at every texel, exactly, for RGBA8 (8-bit unorm), RGBA8UI and RGBA8I.
- **The sample mask** (section 13.8.3). `glSampleMaski(word, mask)` sets one word of `SAMPLE_MASK_VALUE`; bit B of word M
  is sample 32 M + B. When `SAMPLE_MASK` is enabled the fragment's coverage is ANDed with the mask; when it is disabled
  the mask is ignored. Initially it is disabled and every word is all ones. The step is skipped when the framebuffer
  is not multisampled. `MAX_SAMPLE_MASK_WORDS` is at least 1.
- **Sample positions.** `glGetMultisamplefv(SAMPLE_POSITION, i)` returns the *shading* sample location of sample *i* of the draw
  framebuffer, in [0, 1] x [0, 1], (0.5, 0.5) being the pixel centre. The spec says explicitly that the points used for
  coverage need not be these, and that without fixed locations the values describe only some pixels. The demo therefore
  asserts the range, the `INVALID_VALUE` for `index >= SAMPLES`, and one statement that section 8.8 does make: with
  `fixedsamplelocations = TRUE` the locations do not depend on the size or format, so two such textures of different
  size and format with the same sample count have identical positions.
- **`fixedsamplelocations` and the sample count are observable** through framebuffer completeness (section 9.4.2):
  all attached images must have the same sample count; a texture without fixed locations cannot share a framebuffer with a
  renderbuffer, nor with a texture that has fixed locations. The demo uses this because the count itself is not
  reachable (see "meta-gl gaps").
- **Limits.** `MAX_COLOR_TEXTURE_SAMPLES`, `MAX_DEPTH_TEXTURE_SAMPLES` and `MAX_INTEGER_SAMPLES` bound the counts per format
  class, and `glGetInternalformativ(TEXTURE_2D_MULTISAMPLE, format, SAMPLES, ...)` lists the counts actually supported
  (descending, positive; the largest is at least the limit for the format's class, section 20.3.1). The test machine's
  radeonsi offers 8, 4 and 2 samples, llvmpipe and zink 4 and 2.

## Central calls

`glTexStorage2DMultisample`, `glFramebufferTexture2D` (with `TEXTURE_2D_MULTISAMPLE`), `glSampleMaski`,
`glEnable(GL_SAMPLE_MASK)`, `glGetMultisamplefv`, `glGetInternalformativ`, `glGetTexLevelParameteriv`, `glGetIntegerv`
(`SAMPLES`, `SAMPLE_BUFFERS`, `MAX_SAMPLE_MASK_WORDS`, `MAX_*_SAMPLES`), `glGetIntegeri_v(SAMPLE_MASK_VALUE)`,
`glCheckFramebufferStatus`; `texelFetch` and `textureSize` on `sampler2DMS`, `isampler2DMS`, `usampler2DMS`.

## Specification

- ES 3.2 section 8.8 (multisample textures, errors), 8.10 (texture parameters), 8.11.3 (level queries), 8.18 (immutable
  storage), 9.2.3.1 (`SAMPLES`, `SAMPLE_BUFFERS`), 9.2.4 and 9.2.6 (renderbuffers), 9.2.8 (attaching textures), 9.4.2
  (completeness), 11.1.3.3 (multisample texel fetches), 11.1.3.4 (`textureSize`), 13.4 (multisampling, `GetMultisamplefv`),
  13.8.3 (sample mask), 20.3 (internal format queries), appendix A (invariance), chapter 21 (limits).
- GLSL ES 3.10 sections 8.9.1 and 8.9.2 (`textureSize`, `texelFetch`), 4.7.4 (no default sampler precision).
- Nothing of ES 3.2 or GLSL ES 3.20 is used: no `gl_SampleMask`, `gl_SampleID`, sample shading or multisample array textures.

## Validation

107 checks on radeonsi (105 on llvmpipe and zink, which offer one sample count fewer), all exact except where noted:

- **Window:** every pixel of the four cells (65 536) against the same arithmetic as `show.frag`.
- **Limits and queries:** the four limits and `MAX_SAMPLES`; `glGetInternalformativ` for twelve formats (RGBA8, RGBA8UI,
  RGBA8I, RGB10_A2, RG8, R8, SRGB8_ALPHA8, three depth formats, two depth-stencil formats), `count` 1 and 0.
- **Allocation:** every format with every reported count: no error, framebuffer complete, `SAMPLES` between the request
  and the next larger count (printed).
- **Completeness rules** (six cases): texture plus renderbuffer, fixed and non-fixed; two textures differing in
  `fixedsamplelocations`; a multisample and a single-sample texture; each expected status is computed from the measured counts.
- **Sample positions:** range, identity for fixed locations across sizes and formats, three `INVALID_VALUE` cases.
- **Every sample:** the four window textures; RGBA8 with every reported count; a full-coverage draw without mask writes
  all samples.
- **Sample mask:** initial state, the value read back, `glIsEnabled`; masks 0, all ones, "disabled with a stored 0",
  bits beyond `SAMPLES`, 0x55555555, 0xAAAAAAAA, all but one sample (twice); and a single-sample framebuffer that ignores
  a mask of 0.
- **A triangle:** every sample holds the clear or the triangle pattern; pixels 2 or more pixels inside have all samples
  covered, 2 or more outside none (a sanity margin, since the spec allows sample points outside the pixel); at least one
  pixel has partial coverage; then one draw per sample with only that sample in the mask must write exactly that
  sample's coverage of the unmasked draw and leave every other sample untouched (this relies on repeatable
  rasterization, appendix A, rule 1, and on the mask only ANDing the coverage).
- **`textureSize`** for the three sampler types and sizes 5 x 3, 37 x 1 and 16 x 12; `glGetTexLevelParameteriv` of the 37 x 1
  texture; `TEXTURE_IMMUTABLE_FORMAT`.
- **Depth samples** (the only 8-bit-tolerant check, +-1): a DEPTH_COMPONENT24 multisample texture gets depth (k+1)/(N+1) in
  sample k by the mask and the depth test, and `sampler2DMS` reads it back as (d, 0, 0, 1).
- **Errors:** 27 defined `INVALID_ENUM` / `INVALID_VALUE` / `INVALID_OPERATION` cases (see `demo.json`), and three calls that
  must be accepted (a texture that earlier failed calls left untouched gets its storage, `TEXTURE_BASE_LEVEL` 0,
  the last mask word).
- **Negative controls:** a model that ignores the sample mask must disagree with the observed "mask 0x55555555" image
  (half of the values differ), and a model with the sample numbers reversed must disagree with the per-sample patterns.

Every fetch target is cleared to a sentinel first, so a draw that wrote nothing cannot match. 8-bit results are exact
because the pattern values pass through the unorm conversions unchanged (dithering is disabled). During development,
shifting the pattern number by one in the shader made 23 checks fail.

## meta-gl gaps found

- `glGetTexLevelParameteriv(TEXTURE_2D_MULTISAMPLE, 0, GL_TEXTURE_SAMPLES / GL_TEXTURE_FIXED_SAMPLE_LOCATIONS)` is specified
  (section 8.11.3, table 21.11) but `TextureLevelParameter` has no `Samples` or `FixedSampleLocations` enumerator. They exist
  only in `TextureParameterQuery`, i.e. for `glGetTexParameter*`, which the specification does not allow them for. The demo
  therefore reads the sample count through `SAMPLES` of a framebuffer that has the texture attached and the fixed-location
  flag through completeness, and does not claim a direct query.
- `SampleMaskValue` has only `None` and `All`. A mask with some bits set is made with one `static_cast` in `mask_value()`
  (the type is a registered 32-bit bitfield, so any value is valid); an enumerator or a helper for single sample bits would
  remove it.

## Pitfalls

- `glReadPixels` on a multisample framebuffer is `INVALID_OPERATION`; resolve by blit or read samples with `texelFetch`.
- A texture without fixed sample locations cannot be attached together with a renderbuffer; a mixed framebuffer needs equal
  sample counts and `fixedsamplelocations = TRUE`.
- `RENDERBUFFER_SAMPLES` and `TEXTURE_SAMPLES` may be larger than the request, so compare measured counts, not requested ones.
- Integer formats are limited by `MAX_INTEGER_SAMPLES`, depth formats by `MAX_DEPTH_TEXTURE_SAMPLES`; take the counts from
  `glGetInternalformativ` instead of assuming one.
- Sample positions from `glGetMultisamplefv` are shading locations and do not say which samples a triangle edge covers.
- The sample mask does nothing in a single-sample framebuffer. The demo disables it before every fill, so that a mask left
  over from an earlier draw cannot decide what a fill writes.

## Known driver deviations (XFAIL)

- **ANGLE** accepts `glGenerateMipmap(TEXTURE_2D_MULTISAMPLE)` without an error. ES 3.2 8.14.4: *"An INVALID_ENUM error is
  generated if target is not TEXTURE_2D, TEXTURE_3D, TEXTURE_2D_ARRAY, TEXTURE_CUBE_MAP or TEXTURE_CUBE_MAP_ARRAY."*
- **ANGLE** accepts `glGetTexLevelParameteriv(TEXTURE_2D_MULTISAMPLE, 1, ...)`. ES 3.2 8.11.3 makes the maximum level of
  the multisample targets zero, so level 1 is `INVALID_VALUE`.
