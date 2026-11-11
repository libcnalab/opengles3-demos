# es30-immutable-storage

**API introduced in:** OpenGL ES 3.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 ·
**Run:** `es30-immutable-storage [--headless --validate]`

## Purpose

`glTexImage2D` specifies **one level of one texture**, and a texture is only usable once every level it needs has been
specified in a consistent way (texture completeness). Each level may be respecified at any time with another size or format, so
the driver cannot know in advance how much memory or which layout the texture will end up with.

ES 3.0 adds **immutable-format textures**. `glTexStorage2D(target, levels, internalformat, width, height)` and
`glTexStorage3D(...)` specify *all* levels at once: level `i` gets `max(1, floor(size / 2^i))` texels in every dimension, all
with the same sized internal format. From that call on "the format and dimensions of all levels become immutable. The contents of the images and the
parameters can still be modified" (ES 3.0 §3.8.4). The texture can never be incomplete because of a missing level, and any
command that would change format or size is refused with `INVALID_OPERATION`.

The second half of the demo is about the *contents*: ES 3.0 added many sized internal formats (Table 3.13) and a larger set of `(format, type)` pairs
(Table 3.2) that say how client memory is read. Thirteen representative sized formats are created with `glTexStorage2D`, filled with
every valid pair with `glTexSubImage2D`, and compared with values decoded on the CPU; invalid pairs are shown to be errors.

## What the image shows

A contact sheet of 8 x 8 panels of 32 x 32 pixels (panel 0 at the lower left, numbers grow to the right and then upwards; the last
three rows of the sheet are unused background). Panels 0 to 22 are 4 x 4 textures with one texel per 8 x 8 pixels, so each texel can be seen.

| Panels | What is shown |
|---|---|
| 0-3 | `R8`, `RG8`, `RGB8`, `RGBA8` from `RED / RG / RGB / RGBA` with `UNSIGNED_BYTE`. Missing components read as 0 (G, B) and 1 (A) |
| 4 | `R8_SNORM` from `RED / BYTE`; the shader shows `0.5 * value + 0.5`, so -1 is black and +1 is full red |
| 5, 6 | `RGB565` from `UNSIGNED_SHORT_5_6_5` (exact bit fields) and from `RGB / UNSIGNED_BYTE` (8-bit data reduced to 5/6/5 bits) |
| 7, 8 | `RGBA4` from `UNSIGNED_SHORT_4_4_4_4` and from `RGBA / UNSIGNED_BYTE` |
| 9-11 | `RGB5_A1` from `UNSIGNED_SHORT_5_5_5_1`, from `UNSIGNED_BYTE` and from `UNSIGNED_INT_2_10_10_10_REV` |
| 12 | `RGB10_A2` from `UNSIGNED_INT_2_10_10_10_REV` (alpha takes 0, 1/3, 2/3, 1) |
| 13-16 | `R16F` and `RGBA16F` from `HALF_FLOAT` and from `FLOAT` (values up to 4, shown with a scale of 0.25) |
| 17-19 | `R11F_G11F_B10F` from `UNSIGNED_INT_10F_11F_11F_REV`, `HALF_FLOAT` and `FLOAT` |
| 20-22 | `RGB9_E5` from `UNSIGNED_INT_5_9_9_9_REV`, `HALF_FLOAT` and `FLOAT` |
| 23-26 | one immutable 8x8 texture with **4 levels** (8, 4, 2, 1 texels), sampled with `textureLod` at level 0, 1, 2, 3: red, green, blue, yellow. Level 1 has a magenta 2x2 box written with `glTexSubImage2D` |
| 27 | a **mutable** texture with only level 0, the same filter state and `textureLod(1.0)`: incomplete, so it samples black |
| 28 | `TEXTURE_BASE_LEVEL = 100` on the immutable texture: clamped to the last level (equation 3.14), so it shows level 3 (yellow) |
| 29 | `TEXTURE_BASE_LEVEL = 100` on a mutable texture with the same four levels: incomplete, black |
| 30-32 | `textureSize` of the three levels of an immutable 7x5 texture: (7,5), (3,2), (1,1) |
| 33-36 | `textureSize` of the four levels of an immutable 4x2x8 3D texture: (4,2,8), (2,1,4), (1,1,2), (1,1,1) |
| 37-39 | `textureSize` of the three levels of an immutable 6x3 array texture with 5 layers: layers stay 5 |

The `textureSize` panels show the size as a color (`20 x size` per channel).

## Central calls

`glTexStorage2D`, `glTexStorage3D` (the `levels` argument and the size give the level sizes), `glTexSubImage2D` and `glTexSubImage3D` (the only
ways to give texels afterwards), `glGetTexParameteriv(TEXTURE_IMMUTABLE_FORMAT / TEXTURE_IMMUTABLE_LEVELS)`, `glTexImage2D` (as the
command that must now fail), `glCopyTexSubImage2D` (which is still allowed).

## Key ideas

- **One call allocates every level.** After `glTexStorage2D(TEXTURE_2D, 4, RGBA8, 8, 8)` the texture is complete with the default
  `NEAREST_MIPMAP_LINEAR` filter; a texture with one level made by `glTexImage2D` and the same filter is not (panels 27, 23-26).
- **The number of levels has a limit.** `levels > floor(log2(max(width, height))) + 1` is `INVALID_OPERATION`. For 3D textures the *depth*
  counts too, for array textures the *layer count does not* (4x4 with 64 layers has at most 3 levels). The refused call changes nothing:
  the texture is still mutable and the same name accepts a valid call afterwards.
- **What stays mutable.** Texel values (`glTexSubImage*`, `glCopyTexSubImage*`) and texture parameters (`glTexParameter*`, here the filter of
  every panel, `BASE_LEVEL` and `WRAP_S`). Everything that would change format or size (`glTexImage*`, `glCompressedTexImage*`,
  `glCopyTexImage*`, another `glTexStorage*`) is `INVALID_OPERATION`, **even with the very same parameters**.
- **`TEXTURE_BASE_LEVEL` and `TEXTURE_MAX_LEVEL` are clamped** for immutable textures to the last level (equations 3.14 and 3.15), so an out-of-range
  value cannot make the texture incomplete (panel 28), unlike a mutable one (panel 29).
- **Sized internal formats have defined component types and minimum resolutions** (Table 3.13). A sized format is the *only* thing that can name `R8_SNORM`,
  `RGB10_A2`, `R11F_G11F_B10F` or `RGB9_E5` for a texture; `glTexStorage*` accepts only sized formats.
- **`(format, type)` pairs are not free.** Table 3.2 says which pairs may fill which sized format (for example `RGBA16F` accepts `HALF_FLOAT` *and* `FLOAT`, `RGB5_A1`
  accepts `UNSIGNED_BYTE`, `UNSIGNED_SHORT_5_5_5_1` and `UNSIGNED_INT_2_10_10_10_REV`); everything else is `INVALID_OPERATION`. Integer formats need the `*_INTEGER`
  formats, packed types need the format with the matching number of components.
- **Lossy formats.** 8-bit data stored in `RGB565`, `RGBA4` or `RGB5_A1` is converted with equation 2.3, which "returns one of the two ... values which are closest",
  so the result is within one step of the 8-bit input (and an implementation may keep more bits than the table minimum).
- **Shared exponent.** `RGB9_E5` stores three 9-bit mantissas and one 5-bit exponent: the smaller components lose precision when the largest
  is much bigger. Values are `mantissa x 2^(exponent - 24)`. `R11F_G11F_B10F` has no sign bit and no shared exponent, only 5-bit exponents with 6, 6 and 5 mantissa bits.

## Shader concepts

`sample.frag` reads a `sampler2D`, `sampler3D` or `sampler2DArray` (three different units) and writes `texture(...) * scale + bias`,
`textureLod(...)` or `textureSize(...)`. The default framebuffer is 8-bit unsigned normalized, so the scale and bias bring snorm
(`0.5 * v + 0.5`) and half-float (`0.25 * v`) values into range. All three sampler types are declared `highp`: `sampler2D` defaults to `lowp`,
which need not hold half-float values, and `sampler3D` / `sampler2DArray` have no default at all (GLSL ES 3.00 §4.5.4).

## Validation (`--validate`)

181 checks in `validate()` (the runtime adds its own). Every check compares observable results: pixels, texels read back through a framebuffer, state queries or `glGetError`.

- **Formats (69 checks).** For each of the 23 (format, type) cases: `glTexStorage2D` succeeds and `TEXTURE_IMMUTABLE_FORMAT` / `TEXTURE_IMMUTABLE_LEVELS`
  read `TRUE` / 1; the pair is accepted by `glTexSubImage2D`; and every pixel of the panel (576 probes, texel boundaries skipped) matches a **CPU decode written from the spec**
  (unorm, snorm `max(c/127, -1)`, IEEE half float, 11-bit and 10-bit unsigned float, the RGB9_E5 algorithm of §3.8.3.2, which is run for the `HALF_FLOAT` / `FLOAT` uploads and the
  decode formula for the packed upload). Tolerances (8-bit units, on top of 0.5 for rounding the result): **0** for every exact-bit path
  that needs no conversion (`R8`, `RG8`, `RGB8`, `RGBA8`, the level colors, `textureSize`); **1** for the exact packed and float cases (unorm expansion, half-float and shared-exponent
  values are exact; the 8-bit result of the framebuffer may round either way); **one full step of the destination precision** for 8-bit data stored in fewer bits
  (8.2 for 5 bits, 4.0 for 6 bits, 17 for 4 bits), as equation 2.3 allows either neighbouring value. Alpha of `RGB5_A1` is fed as 0 or 255 only (or 0 / 3 for the 2-bit type), so that an
  implementation that keeps more alpha bits gives the same pixels.
- **Mip ladder and sizes (17 checks).** Each of the 4 levels of the immutable texture has the expected flat color or box, the mutable contrast textures are black, the base-level clamp
  gives the last level, and each `textureSize` is exact (including the layer count of the array and the depth of the 3D texture at every level).
- **Negative controls (4 checks).** The signed, shared-exponent, half-float and 4-4-4-4 panels are compared with the *wrong* decoding (unsigned instead of signed, mantissa without exponent,
  no scale, 5-5-5-1 instead of 4-4-4-4) and must differ. The mutable single-level texture and the mutable `BASE_LEVEL = 100` texture are the negative control
  of the immutable behavior (same filter state, black instead of the level color). A manual mutation (shifting the shared-exponent bias by one) makes
  panels 20 to 22 fail.
- **Queries (5 checks).** `TEXTURE_IMMUTABLE_FORMAT/LEVELS` for 2D, 3D and array textures (and cube map), `FALSE` / 0 for a mutable texture, `glTexParameteri` on an immutable texture reads back.
- **Level limits and storage errors (25 `glGetError` checks, 2 "the refused call had no effect" checks and 1 isolated child process).** `levels` around the limit for 2D, cube map, 3D and array textures (accepted at the limit, `INVALID_OPERATION` above it,
  and the refused texture is still usable), `INVALID_VALUE` for zero or negative levels and sizes, `INVALID_ENUM` for the wrong target (`TEXTURE_3D` with `glTexStorage2D`,
  `TEXTURE_2D` with `glTexStorage3D`, ...), `INVALID_OPERATION` for the default texture object, and for a depth format with a 3D target. That last call,
  `glTexStorage3D(TEXTURE_3D, 1, DEPTH_COMPONENT16, 4, 4, 4)`, runs in a **child process** on every driver, because it crashes one
  driver. The executable re-executes itself (`/proc/self/exe` with a hidden first argument, the pattern of
  `tests/negative/driver-crash-readpixels-none`). The child creates its own context, makes the call between two printed markers,
  expects `INVALID_OPERATION`, `TEXTURE_IMMUTABLE_FORMAT` still `FALSE` and a valid `glTexStorage3D(RGBA8)` afterwards, and exits
  with PASS or FAIL. The parent echoes the child's output and classifies its end (exit status, signal, 30 s timeout). Only a
  death between the markers counts as the documented deviation; any other failure is a plain FAIL.
- **Format validation (about 40 `glGetError` checks).** 30 combinations of internal format, format and type through `glTexImage2D` (invalid: `INVALID_OPERATION`; valid ones
  as positive controls), `glTexSubImage2D` with a pair that does not fit the level, depth formats on 3D and array targets.
- **Respecification (11 checks).** On immutable 2D, 3D and array textures: `glTexImage2D` / `glTexImage3D` (with the very same parameters and with others), a second `glTexStorage*`,
  `glCopyTexImage2D` and `glCompressedTexImage2D` give `INVALID_OPERATION`; afterwards the queries are unchanged and the **contact sheet drawn again is byte-identical** to the first one.
- **Still allowed.** `glCopyTexSubImage2D` into an immutable texture changes its texels (read back through a framebuffer); a cube map made with `glTexStorage2D` has all six faces and levels
  (written with `glTexSubImage2D`, read through framebuffer attachments) and an array layer written with `glTexSubImage3D` is read back.

## Known driver deviation and spec notes

- **ANGLE crashes in `glTexStorage3D(TEXTURE_3D)` with a depth format.** ANGLE 2.1.17841 (Vulkan backend) dies with SIGSEGV
  inside `GL_TexStorage3D` (in the Vulkan driver) instead of generating the error. ES 3.0 §3.8.3: *"Textures with a base internal
  format of DEPTH_COMPONENT or DEPTH_STENCIL are supported by texture image specification commands only if target is
  TEXTURE_2D, TEXTURE_2D_ARRAY, or TEXTURE_CUBE_MAP. Using these formats in conjunction with any other target will result in an
  INVALID_OPERATION error."* §3.8.4: *"If executing the pseudo-code would result in any other error, the error is generated and
  the command will have no effect."* `glTexImage3D(TEXTURE_3D)` with the same format is rejected correctly. The isolated
  check is reported as XFAIL on "ANGLE".
- **Unsized `RGB + FLOAT` depends on `GL_OES_texture_float`.** Core ES 3.0 Table 3.3 (ES 3.2 Table 8.3) lists
  unsized `RGB` only with `UNSIGNED_BYTE` and `UNSIGNED_SHORT_5_6_5`. `OES_texture_float` adds `RGB/RGBA + FLOAT`, and
  every Mesa context here exposes it. So accepting the pair is correct, and the check's expectation follows the
  context. (This was first reported as a Mesa deviation; integration review found the extension.)
  `RGBA + HALF_FLOAT` stays invalid, because `OES_texture_half_float` defines a different token, `HALF_FLOAT_OES`.
- **Not checked: `INVALID_ENUM` for an unsized internal format in `glTexStorage*`** (ES 3.0 §3.8.4: "If internalformat is one of the unsized base internal formats listed in table 3.3, an
  INVALID_ENUM error is generated"). meta-gl types the argument as `SizedInternalFormat` or `CompressedInternalFormat`, so `GL_RGBA` cannot be passed without a cast, and the demos do not cast.
- ES 3.0.6 has no `glGetTexLevelParameter` (it is ES 3.1), so the level sizes are verified with `textureSize` in the shader, and the dimensions of a refused respecification by drawing the sheet again.

## Pitfalls

- `glTexStorage2D` takes a **sized** format. `RGBA` (with a type chosen later) was the ES 2.0 way; `RGBA8` is the immutable one.
- A texture used with `glTexStorage*` cannot be resized or reformatted, ever: to change it create a new texture object. Re-calling the same `glTexStorage*` is an error too.
- Because the data is unspecified after `glTexStorage*`, a texture can be complete and still contain garbage until every level that is sampled has received texels.
- The number of levels is limited by the largest dimension *including depth for 3D*, and *excluding the layer count for arrays*.
- `glTexSubImage2D` still checks `(format, type)` against the internal format of the level: `RGB` data cannot be written into an `RGBA8` level.
- Fractional float values are not exact in 11- and 10-bit floats and in RGB9_E5 once the exponents are shared; the demo uses multiples of 1/16 so that the stored value is the value.
