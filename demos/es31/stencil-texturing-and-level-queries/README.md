# es31-stencil-texturing-and-level-queries

**API introduced in:** OpenGL ES 3.1 · **Minimum context:** ES 3.1 · **Shaders:** GLSL ES 3.10 (`#version 310 es`) ·
**Run:** `es31-stencil-texturing-and-level-queries [--headless --validate]`

## Purpose

Two small ES 3.1 texture features that are easy to use and hard to trust:

* **`DEPTH_STENCIL_TEXTURE_MODE`.** A `DEPTH24_STENCIL8` or `DEPTH32F_STENCIL8` texture has two components in one image.
  ES 3.0 could sample only the depth. In ES 3.1 the texture parameter `DEPTH_STENCIL_TEXTURE_MODE` selects which one a
  shader sees: `DEPTH_COMPONENT` (the initial value) gives the depth through an ordinary `sampler2D`; `STENCIL_INDEX` gives
  the 8-bit stencil value through an **unsigned integer** sampler, `usampler2D`. The stencil view is an integer texture, so
  only `NEAREST` filtering is allowed (ES 3.2 section 8.16). The parameter is ignored for textures that are not
  depth/stencil textures.
* **`glGetTexLevelParameteriv` / `glGetTexLevelParameterfv`.** The first way to ask the GL what one *image* of a texture
  looks like: its size, its internal format, the actual resolution and the data type of each component, whether it is
  compressed. ES 3.0 had only the texture-wide `glGetTexParameter`.

The demo proves the first by rendering known stencil values into a depth/stencil texture and **reading every texel back inside a
shader**; it proves the second by comparing the reported state with what was allocated, for every kind of texture the
query accepts.

## What the image shows

A 2 x 2 contact sheet of 128 x 128 panels; every panel shows a 32 x 32 texture, one texel per 4 x 4 pixels, through a fragment shader.

```
 DEPTH32F_STENCIL8  depth (gray)  |  DEPTH32F_STENCIL8  stencil (colors)
 DEPTH24_STENCIL8   depth (gray)  |  DEPTH24_STENCIL8   stencil (colors)
```

The scene in the textures is a 4 x 4 grid of 8 x 8-texel blocks. Block 0 (lower left) is the untouched clear value (depth 1, stencil
0); the other fifteen have their own depth (`(2b + 3) / 64`) and their own stencil value. A large rectangle (texels 4..28, depth 0.25)
then won the depth test only where the stored depth was larger (blocks 7 to 15), so there the gray panel is flattened to 0.25, while its stencil was
incremented where it *lost* (blocks 1 to 6); a smaller rectangle with a stencil test (depth 0.125) leaves a darker patch where it passed and inverted or
cleared the upper nibble of the stencil values it touched. The two formats must look **identical**, and they do.

## How the textures are filled

The textures are immutable (`glTexStorage2D`). A framebuffer object has **only** the texture as its `DEPTH_STENCIL_ATTACHMENT`: no
color attachment, `glDrawBuffers(NONE)`, `glReadBuffer(NONE)`. A framebuffer like this is complete (ES 3.2 section 9.4.2 asks for one attached image
of any kind), and a fragment shader with no output variable is legal. Three passes draw axis-aligned rectangles whose corners lie on texel
boundaries and whose depths are multiples of 1/64 (exact in `float`):

| Pass | Draws | Depth test | Stencil test and ops |
|---|---|---|---|
| 1 | 15 blocks of 8 x 8 texels | `LESS`, depth `(2b + 3) / 64` | `ALWAYS`, `REPLACE` with a per-block reference (255, 1, 2, 128, 127, 85, 170, 3, 5, 8, 13, 21, 34, 55, 89) |
| 2a | rectangle (4,4)-(28,28), depth 0.25 | `LESS` | `ALWAYS`; ops `KEEP`, `INCR`, `KEEP` (depth fail increments, 255 saturates at 255) |
| 2b | rectangle (12,12)-(20,24), depth 0.125 | `ALWAYS` | `LESS` 100; ops `ZERO`, `KEEP`, `INVERT` with **write mask 0xF0** |

A CPU model applies the same state to a depth-1 / stencil-0 buffer, test by test (sections 15.1.2 and 15.1.3). It predicts 18 distinct stencil values
(all eight bits are used) and 16 distinct depths, and passes 2a and 2b change 544 of the 1024 texels, so a missing pass cannot hide.
The depths are chosen so that no depth comparison is a tie: the CPU and the GPU agree on every test whatever the precision of the depth buffer.

## How they are read back

A compute shader reads the texture twice per texel, with `texelFetch(u_tex, xy, 0)` and with
`textureLod(u_tex, (xy + 0.5) / size, 0.0)` at the texel centre (NEAREST), and stores the raw bits in a shader storage buffer
(`floatBitsToUint` for depth). Two programs differ only in the sampler: `sampler2D` while the mode is `DEPTH_COMPONENT`, `usampler2D` while it is
`STENCIL_INDEX`. The buffer is initialised with `0xDEADBEEF`, so a texel the shader did not write cannot go unnoticed. The application sets the mode with
`glTexParameteri`, dispatches, calls `glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT)` (the mapped read is a buffer access by the commands of section 6),
and maps the buffer. The contact sheet additionally reads the same textures in fragment shaders (`texture()` on both sampler types).

## Central calls

| Call | Role | Spec (ES 3.2) |
|---|---|---|
| `glTexParameteri(TEXTURE_2D, DEPTH_STENCIL_TEXTURE_MODE, DEPTH_COMPONENT / STENCIL_INDEX)` | what a shader samples | 8.16, 8.10 |
| `glGetTexParameteriv(..., DEPTH_STENCIL_TEXTURE_MODE)` | initial value and state | 8.11.2, 8.19 |
| `glGetTexLevelParameteriv(target, level, pname, &v)` / `...fv` | one image's state | 8.11.3 |
| `glTexStorage2D/3D`, `glTexStorage2DMultisample`, `glTexImage2D/3D` | what is allocated | 8.5, 8.8, 8.18 |
| `glFramebufferTexture2D(..., DEPTH_STENCIL_ATTACHMENT, ...)` | render the stencil values | 9.2.8, 9.4.2 |
| `glDispatchCompute`, `glMemoryBarrier`, `glMapBufferRange` | read the texture in a shader | 17, 7.11.2 |

## The level queries

Every query is compared with the allocation; the component sizes are only checked to be *at least* the minimum of the format tables, because
ES 3.2 section 8.11.3 says they "return the actual resolutions of the stored image components, not the resolutions specified" and tables 8.10 and 8.11 list minimum "bits/component".

| Texture | What is checked |
|---|---|
| 25 sized formats, 8 x 4, `glTexStorage2D` | `WIDTH`, `HEIGHT`, `INTERNAL_FORMAT`, `COMPRESSED` = FALSE; `RED/GREEN/BLUE/ALPHA/DEPTH/STENCIL_SIZE` >= table value; `RED/GREEN/BLUE/ALPHA/DEPTH_TYPE` (`UNSIGNED_NORMALIZED`, `SIGNED_NORMALIZED`, `FLOAT`, `INT`, `UNSIGNED_INT`, `NONE` for a missing component); `SHARED_SIZE` >= 5 for `RGB9_E5`; `TEXTURE_SAMPLES` = 0 and `TEXTURE_FIXED_SAMPLE_LOCATIONS` = TRUE (the defaults the spec promises for non-multisample images); the float query equals the integer query for all 19 pnames |
| 2D 16 x 8, 3 levels | 16 x 8, 8 x 4, 4 x 2; level 3 was never allocated: `WIDTH` = `HEIGHT` = 0 |
| 3D 8 x 4 x 6, 3 levels | 8x4x6, 4x2x3, 2x1x1 (the depth halves with the level) |
| 2D array 8 x 8 x 5, 4 levels | 8x8x5, 4x4x5, 2x2x5, 1x1x5 (the layer count does **not** halve) |
| cube map 8 x 8, 4 levels | all six faces (`TEXTURE_CUBE_MAP_POSITIVE_X` ...) at every level; `TEXTURE_CUBE_MAP` itself is `INVALID_ENUM` |
| mutable `glTexImage2D/3D` | levels specified one by one; respecified with another size and format; an unspecified level reports 0 |
| `COMPRESSED_RGB8_ETC2` | `COMPRESSED` = TRUE, size and format as specified, color types `UNSIGNED_NORMALIZED` (the reported resolutions are "rough approximations" and are printed, not asserted) |
| multisample RGBA8 (fixed locations TRUE) and DEPTH24_STENCIL8 (FALSE) | size, format, resolutions, types; `TEXTURE_SAMPLES` between the request and the next larger count `glGetInternalformativ` lists (section 8.8), `TEXTURE_FIXED_SAMPLE_LOCATIONS` as requested; with the texture attached to a framebuffer `SAMPLES` equals `TEXTURE_SAMPLES` and `SAMPLE_BUFFERS` is 1; level 1 is `INVALID_VALUE` (maximum level is zero) |

Errors: level -1 and `log2(MAX_TEXTURE_SIZE) + 1` are `INVALID_VALUE`; `log2(MAX_TEXTURE_SIZE)` itself is a valid level (and reports 0
for an unspecified image). The `DEPTH` query of a 2D, cube or multisample image is not asserted (the spec only defines it for 3D images and arrays); it is
asserted for 3D textures and arrays.

## Validation

* **State.** The mode is `DEPTH_COMPONENT` in a new texture (section 8.19), `STENCIL_INDEX` after being set, an invalid value (`RED`) is `INVALID_ENUM` and changes
  nothing (section 8.10), and switching back and forth loses nothing.
* **Depth.** All 1024 depth texels, from `texelFetch` and from `textureLod`, in both formats: |read - model| <= 2 / (2^24 - 1) for `DEPTH24_STENCIL8`
  (a fixed-point depth buffer stores k / (2^24 - 1); how a window z is rounded to k is not specified, so one step, plus under half a step for the
  float conversion) and <= 1e-6 for `DEPTH32F_STENCIL8` (the single-precision viewport transform). The measured error is printed; on all drivers it is 0.
* **Stencil.** All 1024 values from both lookups, in both formats, **exactly** equal to the model.
* **Contact sheet.** Every one of the four 1024-texel panels is read back with `glReadPixels` and compared with the model's gray / color at the texel
  centre, tolerance 1 (8-bit quantization of the shader output).
* **Mode ignored.** An RGBA8 texture and a `DEPTH_COMPONENT32F` texture set to `STENCIL_INDEX` are still sampled as before (section 8.16).
* **Level queries.** As in the table above: one check per level or per group, with the full list of mismatches in the failure text.
* **Negative controls (the checks can fail).** (1) A depth/stencil texture that was only cleared, run through the identical dump and comparison,
  mismatches in 1952 reads for depth and 1920 for stencil (of 2048 each). (2) The stencil dump compared with the model of pass 1 only mismatches in 416
  reads. (3) A level judged against the wrong size or the wrong format is rejected (`WIDTH`, `HEIGHT`, `INTERNAL_FORMAT`, `GREEN_TYPE`).
  (4) The multisample level judged as fixed-sample-location (or as a single-sample image) is rejected. (5) The scene itself is checked to be rich enough (>= 16 stencil values using all 8 bits, >= 10 depths, > 400 texels changed by the second pass).

## Pitfalls

* **The default minification filter is `NEAREST_MIPMAP_LINEAR`**, and a sized depth or depth/stencil format with `TEXTURE_COMPARE_MODE = NONE` and any
  non-NEAREST filter makes the texture **incomplete** (section 8.17); so does a `STENCIL_INDEX` view with `LINEAR`. An incomplete texture samples as
  (0, 0, 0, 1), and for an integer sampler the result is undefined. Set both filters to `NEAREST`.
* **Use an unsigned integer sampler for the stencil.** `sampler2D` or `isampler2D` on a `STENCIL_INDEX` view is undefined (section 11.1.3.5).
* **The mode is a property of the texture object**, not of the sampler object and not of the program. Reading both views in one dispatch needs two textures
  (ES 3.1 has no texture views), so the demo runs two compute programs.
* **A framebuffer with a depth/stencil texture needs both attachments to be the same image.** Use `DEPTH_STENCIL_ATTACHMENT`, and detach it before attaching
  something else to `DEPTH_ATTACHMENT` of the same framebuffer object, or the framebuffer becomes `FRAMEBUFFER_UNSUPPORTED`.
* **`TEXTURE_SAMPLES` is what the driver allocated, not what you asked for.** `glTexStorage2DMultisample(samples)` is a request for a *minimum*; the
  query returns the actual count, which the spec bounds by the request and the next larger count the implementation supports (section 8.8). For a
  non-multisample image the same pnames return the table 21.11 defaults, 0 and TRUE.
* The sizes of components that a format does not have are **not** asserted to be 0: the specification identifies a missing component by type `NONE`, and the
  demo prints any non-zero size it sees (none on the tested drivers).

## meta-gl notes

* `metagl::TextureLevelParameter` originally lacked `GL_TEXTURE_SAMPLES` and `GL_TEXTURE_FIXED_SAMPLE_LOCATIONS` (both are legal for
  `glGetTexLevelParameter{if}v` in ES 3.1 / 3.2, section 8.11.3 and table 21.11). meta-gl's `opengles3-demos/enum-gaps` branch (commit 0a7766c) added
  `TextureLevelParameter::Samples` and `::FixedSampleLocations`, which this demo uses; it needs that meta-gl revision.
* `glTexParameteri(..., DepthStencilTextureMode, GLint)` takes the value as a plain `GLint`; there is no typed enum for its legal values
  (`DEPTH_COMPONENT`, `STENCIL_INDEX`), so the demo passes the `GL_*` macros.
* `TextureParameterQuery::Samples` / `FixedSampleLocations` exist for `glGetTexParameter*`, but section 8.11.2 does not list those pnames for that command.
