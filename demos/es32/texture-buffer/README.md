# es32-texture-buffer

**API introduced in:** OpenGL ES 3.2 · **Minimum context:** ES 3.2 · **Shaders:** GLSL ES 3.20 ·
**Run:** `es32-texture-buffer [--headless --validate] [--window]` (window keys: `a`/`d` orbit, `w`/`s` pitch, `q`/`e` zoom)

Spec numbering follows ES 3.2 (§8.9 Buffer Textures, Table 8.18) and GLSL ES 3.20 (§4.1.7, §8.9).

## Purpose

A uniform array is small (a few hundred vectors) and a texture is filtered, 2D and limited by `MAX_TEXTURE_SIZE`. A **buffer texture** sits between
them: an ordinary buffer object is *attached* to a texture of target `TEXTURE_BUFFER`, and the shader sees the bytes of the buffer as a
one-dimensional array of texels, addressed by an integer, with no filtering, no wrapping, no mip levels and no sampler state. The limit is
`MAX_TEXTURE_BUFFER_SIZE` texels (at least 65536; 268435455 on radeonsi), so it is the tool for per-instance data, lookup tables and anything else that is "an
array that is too big for a uniform block and too regular for a vertex attribute".

This demo exercises the whole feature: every internal format the specification allows (Table 8.18, 30 of them), the offset and size rules of
`glTexBufferRange`, the new `TEXTURE_BUFFER` buffer binding point, state queries, `imageLoad`/`imageStore` on `imageBuffer`, and the compile-time
rules of the GLSL ES 3.20 buffer samplers. It then uses the feature the way an application would: a 3D scene of 64 cubes drawn with a single
`glDrawElementsInstanced`, whose transforms, colours and materials are all buffer textures.

## What the image shows

A 256 × 256 frame (frame 19 of the animation, time = frame/60 s):

- **Top 224 rows: 64 cubes in an 8 × 8 grid**, seen from above at 38° pitch with a perspective camera, depth testing and a fixed light. They bob up and down, rotate about the vertical axis and
  change size; every cube has its own hue and one of three materials (diffuse, unlit, bright-top/dim-sides). All of that is one draw call. The vertex shader picks what it needs with `gl_InstanceID`:
  - `u_transform` (`samplerBuffer`, **RGBA32F**): texel `2i` is `(x, y, z, scale)`, texel `2i+1` is `(angle, 0, 0, 0)`. Rewritten every frame with `glBufferSubData` on the `TEXTURE_BUFFER` target.
  - `u_color` (`samplerBuffer`, **RGBA8**, normalized): one texel per instance, `texelFetch` returns `vec4` in [0, 1].
  - `u_material` (`usamplerBuffer`, **R16UI**): one texel per instance, `texelFetch(...).x` is the material number.
- **Bottom 32 rows: a strip with one 4-pixel cell per instance**, drawn by a fragment shader that `texelFetch`es the same two textures. Upper half: the colour texels (a rainbow); lower half: the material texels
  (grey 0.25 / 0.5 / 0.75; the diagonal stripes of the grid show as a repeating pattern).

**Other window sizes.** The strip stays 32 rows high and its 64 cells share the width; the scene fills the rest. The camera's
45° field of view spans the *shorter* side of the scene area, so a tall window shows the whole grid instead of cropping it at
the sides. The minimum window is **128 × 96** (strip cells of 2 pixels, a scene 64 pixels high); smaller windows are skipped.
When the scene is too small to give the 3000 compared probes at every third pixel, every pixel is probed instead.

The three textures are three **ranges of one buffer object** (`glTexBufferRange`): the transforms at offset 0, the colours and the materials at offsets rounded up to
`TEXTURE_BUFFER_OFFSET_ALIGNMENT`, each preceded by a guard gap of sentinel bytes. A wrong offset reads garbage and shows up in the picture.

## Central calls

`glGenBuffers` / `glBindBuffer(TEXTURE_BUFFER, b)` / `glBufferData` / `glBufferSubData`, **`glTexBuffer(TEXTURE_BUFFER, internalformat, buffer)`**,
**`glTexBufferRange(TEXTURE_BUFFER, internalformat, buffer, offset, size)`**, `glBindTexture(TEXTURE_BUFFER, t)`, `glGetIntegerv(TEXTURE_BUFFER_BINDING |
TEXTURE_BINDING_BUFFER | MAX_TEXTURE_BUFFER_SIZE | TEXTURE_BUFFER_OFFSET_ALIGNMENT)`, `glBindImageTexture`, `glDispatchCompute`, `glMemoryBarrier`, `glDrawElementsInstanced`.
In GLSL: `samplerBuffer` / `isamplerBuffer` / `usamplerBuffer`, `texelFetch(gsamplerBuffer, int)`, `textureSize(gsamplerBuffer)`, `imageBuffer` + `imageLoad` / `imageStore` / `imageSize`.

## Key ideas

- **Attaching.** `glTexBufferRange` attaches `size` bytes starting at `offset` of a buffer object to the buffer texture bound on the active unit. `internalformat` must be one of the 30 sized formats of
  Table 8.18 (`R8`, `R16F`, ..., `RGB32UI`, ..., `RGBA32UI`; note that there is no `RGB8`, no `RGB16*`, no snorm, no sRGB and no unsized format).
  `glTexBuffer` is exactly `glTexBufferRange(..., 0, BUFFER_SIZE)`; passing `buffer = 0` detaches (and ignores `offset` and `size`).
- **Offsets.** `offset` must be a multiple of `TEXTURE_BUFFER_OFFSET_ALIGNMENT`, otherwise `INVALID_VALUE`. The value is implementation dependent and at most 256 (4 on radeonsi, 16 on llvmpipe and zink here), so
  a portable program rounds up to the queried value, never to a constant.
- **Texel count.** The image has `floor(size / (components × sizeof(element)))` texels, clamped to `MAX_TEXTURE_BUFFER_SIZE`. A range of 203 bytes of `RGBA8` is 50 texels, a range of 100 bytes of `RGB32F` is 8 texels
  (the 12-byte texel does not divide 100) and trailing bytes are simply not part of the texture. `textureSize(sampler)` returns that count.
- **Element decoding (Table 8.18).** Element `m` of texel `n` is element `n × components + m` of the store. Unsigned normalized formats (`R8`, `RG8`, `RGBA8`) become `c / 255` (equation 2.1), half floats widen
  to float, signed integers are sign-extended, and the **missing components are `0, 0, 1`** (`R8` fetches `(r, 0, 0, 1)`, `RGB32F` fetches `(r, g, b, 1)`; the 1 is the integer 1 for the integer formats).
  The sampler type must match the format family (`samplerBuffer` for normalized and float formats, `isamplerBuffer` for the signed and `usamplerBuffer` for the unsigned integer formats), otherwise the result is undefined.
- **A buffer bound to the `TEXTURE_BUFFER` target is only a handle for editing.** "The buffer object bound to TEXTURE_BUFFER has no effect on rendering" (§8.9): `glBindBuffer(TEXTURE_BUFFER, b)` followed by
  `glBufferData` / `glBufferSubData` is the canonical way to write data that a buffer texture reads, but the texture reads the buffer it was *attached* to, not the one currently bound to the target.
- **Changes are visible.** "When the contents of a buffer object's data store are modified, those changes are reflected in the contents of any buffer texture to which the buffer object is attached." The
  animation of the scene depends on it.
- **Images.** A buffer texture can also be bound to an image unit (`glBindImageTexture` accepts "a buffer texture" even though it is not an immutable texture) and read and written with `imageLoad` / `imageStore`
  through `imageBuffer`, whose coordinate is a plain `int` (Table 8.25: `i = x`). Only the 12 formats that Table 8.18 and Table 8.26 share can be used with exact format match; `RGBA8_SNORM` is an image format but
  not a buffer texture format. After an `imageStore`, `glMemoryBarrier(TEXTURE_FETCH_BARRIER_BIT)` makes the data visible to `texelFetch` and `BUFFER_UPDATE_BARRIER_BIT` to buffer reads.

## Shader concepts

- `uniform highp samplerBuffer u_tex;`: **a precision qualifier is mandatory.** GLSL ES 3.20 §4.7.3 gives a default precision to `int`, `float`, `sampler2D`, `samplerCube` and `atomic_uint` only ("there is no default
  precision qualifier in any of the languages for any type not listed above"), so `uniform samplerBuffer u;` does not compile, nor do `isamplerBuffer`, `imageBuffer`, `samplerCubeArray` and the other new types.
- `texelFetch(gsamplerBuffer, int P)` is the only lookup for buffer samplers. There is no `lod` argument (a buffer texture has no levels), no `texture()`, no `textureLod`, no offset form.
  `textureSize(gsamplerBuffer)` has no `lod` argument either.
- `samplerBuffer` and `imageBuffer` are new in GLSL ES 3.20: a `#version 310 es` shader that uses them does not compile (without an extension, which this project never uses).
- `gl_InstanceID` indexes per-instance data directly, with no divisor and no instanced vertex attribute: `texelFetch(u_color, gl_InstanceID)`.
- `fetch.comp` writes the fetched words with `floatBitsToUint` (float family), `uvec4(v)` (signed integers, a bit-preserving conversion) or the value itself (unsigned), so every component is compared bit for bit on the CPU.

## Validation (`--validate`)

All references are computed on the CPU from the bytes that were uploaded; no stored golden data. 130 checks:

- **Limits (2).** `MAX_TEXTURE_BUFFER_SIZE >= 65536`; `TEXTURE_BUFFER_OFFSET_ALIGNMENT` between 1 and 256 (Table 21.40; the footnote says the value is the maximum allowed). The values are printed.
- **Table 8.18 (30 + 1).** For each of the 30 formats: 64 pseudo-random texels (hash generated: full range for integers, normal-range halves and floats so there are no NaNs or denormals), attached with `glTexBufferRange`, fetched
  in a compute shader through the matching sampler type; every component of every texel must equal the CPU decode of Table 8.18 (bit for bit; the three normalized formats within 1e-6 of `c/255`, the measured worst case
  is 3e-8 on radeonsi and 7e-8 on llvmpipe, which is float rounding of equation 2.1), and `textureSize` must be 64. A last check attaches 100 `RGBA16UI` texels with `glTexBuffer` and checks `BUFFER_SIZE` (800 bytes).
- **Offsets and sizes (9).** Seven ranges with offsets of 0 to 5 × alignment and sizes that are not multiples of the texel size (`textureSize == floor(size / texel size)`, the first texel is the one at the offset,
  all texels match); an `RGBA32F` range at one alignment unit; re-attachment with another format and size.
- **Errors (18).** Misaligned offset (two values), negative offset, size 0, negative size, `offset + size > BUFFER_SIZE` (two forms), a name that is not a buffer (`INVALID_OPERATION`), target `TEXTURE_2D` for both commands
  and three formats outside Table 8.18 (`INVALID_ENUM`); after the misaligned, the oversize and the whole group of failed calls the previous attachment must still be in effect (§2.3.1: an erroring command has no effect);
  `buffer = 0` with offset −1 and size −1 raises no error.
- **State and level queries (13).** `TEXTURE_BUFFER_BINDING` and `TEXTURE_BINDING_BUFFER` are 0 initially (read at the start of `init()`), follow bind / rebind / unbind, and do not disturb `ARRAY_BUFFER_BINDING` or
  `SHADER_STORAGE_BUFFER_BINDING`; `TEXTURE_BINDING_BUFFER` is per texture unit (three units, three textures) and independent of the `TEXTURE_2D` binding. `glGetTexLevelParameteriv(TEXTURE_BUFFER, level 0, ...)` is
  valid, levels 1 and −1 raise `INVALID_VALUE` (§8.11.3).
- **Updates (5).** Initial contents; a decoy buffer bound to the `TEXTURE_BUFFER` target does not change what the attached texture reads; `glBufferSubData` through the target changes one texel, then one component, and the other
  texels stay; **negative control**: a model that missed the update fails the comparison.
- **Images (37).** For each of the 12 shared formats a compute shader loads 100 texels from a `readonly` image and stores them mirrored (`n − 1 − i`, with the four components reversed for the 4-component formats) to a
  `writeonly` image over a sentinel-filled buffer; `imageSize` is 100 for both; the buffer contents read back with `glMapBufferRange` must equal the model (exactly, except `RGBA8`, which goes through floats, where equation 2.3 allows one of the two
  closest integers, so ±1 per byte; the measured difference is 0), and a second compute pass re-reads the image-written texture through `texelFetch` after a `TEXTURE_FETCH` barrier. `IMAGE_BINDING_NAME/FORMAT/ACCESS` of the destination unit are read back for
  every format (12 checks). That exactly 12 formats were tested is a check of its own.
- **Largest texture (1).** 65536 `R8UI` texels, all fetched, last one included, `textureSize == 65536`.
- **Compile rules (7).** One control shader with the three samplers and an `imageBuffer` compiles; six shaders do not: no precision on `samplerBuffer`; `texelFetch` with a `lod`; `texelFetch` with an `ivec2`;
  `textureSize` with a `lod`; `texture()` on a `samplerBuffer`; `samplerBuffer` in `#version 310 es`.
- **Scene (6).** *Strip:* all 64 colour texels (±1 for the 8-bit output) and 64 material texels (±1 for the 127.5 tie) read in a fragment shader; negative control: a strip model shifted by one instance fails everywhere.
  *Cubes:* a ray caster written for this check (rotated boxes, nearest hit, the same light and materials) predicts the colour of every third pixel of the scene (every pixel in a scene too small for 3000 probes that way) (about 6100 probes, 1600 of them on cubes, all 64 instances
  visible, top and side faces). A probe is used only if the hit (instance and face) is the same when every cube is grown and shrunk by 0.01 world units (a fifth of a pixel); this excludes silhouettes and cube edges, where the rasterizer may
  go either way. The worst difference is 1 (8-bit rounding), and the tolerance is 1. Negative control: the same top-face probes against a model whose colours come from the neighbouring instance disagree.
  *Update:* `glBufferSubData(TEXTURE_BUFFER)` of one colour texel changes that instance's pixel to the new colour in the next frame, and writing the old colour back restores it.
- **Final (1).** `glGetError() == NO_ERROR` after all experiments.

**Not asserted.** `texelFetch` of a coordinate outside the texel range: §8.9 says "the results of a texel fetch are undefined if the specified texel coordinate is negative, or greater than or equal to the clamped number of
texels", and §11.1.3.2 lists the same under "undefined". A common assumption is that out-of-range fetches return zero; ES 3.2 does not say so for buffer textures (its only robust-access
text, §10.3.5, is about element-array indices that reference vertex data outside a vertex buffer) and the demo does not rely on it. The demo only prints what this driver returns for four texels *inside the buffer but behind the range* (zeros on radeonsi, llvmpipe and zink); it never reads outside the buffer.
Likewise `TEXTURE_WIDTH` and `TEXTURE_INTERNAL_FORMAT` of a buffer texture level are printed, not asserted, because ES 3.2 does not define them for buffer textures.

## meta-gl gaps found

- `TextureLevelParameter` has no `BufferDataStoreBinding`, `BufferOffset` or `BufferSize`. ES 3.2 Table 21.11 lists `TEXTURE_BUFFER_DATA_STORE_BINDING`, `TEXTURE_BUFFER_OFFSET` and `TEXTURE_BUFFER_SIZE` as
  `GetTexLevelParameteriv` state of the active image unit's buffer texture (§8.9), so `glGetTexLevelParameteriv(TEXTURE_BUFFER, 0, TEXTURE_BUFFER_OFFSET, ...)` should be expressible. The three enumerators exist only in
  `TextureParameterQuery`, which is the domain of `glGetTexParameter*`, where the specification does not allow them (§8.11.2 lists a fixed set of pnames; any other gives `INVALID_ENUM`). Those queries were therefore skipped,
  not cast. Consequently the attached buffer name, offset and size are verified through their effects only (`textureSize`, the fetched bytes, the `INVALID_VALUE` rules).
- Not a gap but worth knowing: `glTexBuffer` takes `InternalFormat` (which includes formats Table 8.18 rejects, so the INVALID_ENUM cases are testable without casts).

## Spec ambiguities and how they were resolved

- **`buffer = 0` and the error list.** §8.9 says that with `buffer == 0` "the values offset and size are ignored", while the `Errors` list lower on the page lists `INVALID_VALUE` for a negative offset and a non-positive size without
  that condition. The demo follows the explicit semantic sentence (no error); all four drivers agree.
- **Out-of-range fetches** are undefined (see above), the task description assumed zero.
- **`TEXTURE_BUFFER_OFFSET_ALIGNMENT`** is a "maximum allowed" limit (Table 21.40 footnote †), so a smaller value is legal and the misaligned-offset case does not exist for an alignment of 1 (the demo notes this and skips it).

## Pitfalls

- `texelFetch` on a buffer texture whose buffer has not been attached, or whose sampler type does not match the format family, is undefined; the demo never does either.
- A texel index outside `[0, textureSize)` is undefined, even if the address happens to be inside the buffer. Clamp or guard in the shader.
- The offset alignment is queried, not assumed: 4, 16 and 256 are all legal. Per-range guard gaps in the scene buffer exist so a hard-coded offset that works on one driver fails on another.
- `glTexBuffer` takes `size` from `BUFFER_SIZE` when the command is issued (§8.9: it is `glTexBufferRange(..., 0, size)`); if you reallocate the buffer with another size, attach it again instead of relying on what the old attachment does.
- `RGB8` (and `RGB16F`, unsized `RGBA`) are not buffer texture formats: `INVALID_ENUM`, and with `texelFetch` it is easy to forget that a 12-byte `RGB32F` texel is not a multiple of any alignment you may have picked.
- Reading the image-written texture through a sampler needs `glMemoryBarrier(TEXTURE_FETCH_BARRIER_BIT)`; reading the buffer with `glMapBufferRange` needs `BUFFER_UPDATE_BARRIER_BIT`.
