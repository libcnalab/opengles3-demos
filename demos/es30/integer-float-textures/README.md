# es30-integer-float-textures

**API introduced in:** OpenGL ES 3.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 ·
**Run:** `es30-integer-float-textures [--headless --validate]`

## Purpose

In OpenGL ES 2.0 every texture is a grid of normalized 8-bit colors that a shader reads as floats in [0, 1]. ES 3.0 adds
textures whose texels are *not* colors: **unsigned and signed integers** (`R8UI`, `R16I`, `RGBA16UI`, `R32I`, `RGBA32UI`,
`RGB10_A2UI`, ...), **32-bit and 16-bit floats** (`R32F`, `RGBA32F`, `RGBA16F`, `R16F`), and two packed float encodings,
**`R11F_G11F_B10F`** and the shared-exponent **`RGB9_E5`**. A shader reads integer textures through `usampler2D` / `isampler2D`
(the lookup returns `uvec4` / `ivec4`) and float textures through an ordinary `sampler2D`.

This demo shows what each of them returns, bit for bit, and which sampling state makes them usable: integer
formats can only be point sampled, so a texture that still has its initial filters is *incomplete*; the 16-bit
float and packed formats may be filtered; the 32-bit float formats may not, unless an extension says so.

## What the image shows

A 4 x 4 contact sheet in a 256 x 256 window. Each cell is one 8 x 8 texture, drawn one texel per 8 x 8 pixel block
(texel row 0 is at the bottom of its cell). The texels hold smooth gradients so the picture means something, with the
low-order bits replaced by a hash (so the exact checks cover full-entropy values) and a few extreme texels (white and
black dots in the float cells are `FLT_MAX`, `-FLT_MAX`, `65504`, ...). The color is `clamp(value * scale + bias, 0, 1)`,
for display only; the exact checks do not use it.

| Row | Cells |
|---|---|
| 0 | `R8UI` gray ramp · `R16I` ramp over the signed 16-bit range · `RGBA8I` (red/green ramps, blue checker from the sign bit) · `RGBA16UI` |
| 1 | **`R32I` = 2^24 + k**: the left half of every texel is `(value - 2^24) / 63` (a clean ramp, because the integer subtraction is exact), the right half is green when `int(float(value)) == value` and red when it is not (the stripes: a float cannot hold an odd integer above 2^24) · `R32I` over the whole `int` range (`INT_MIN` .. `INT_MAX`) · `RGBA32UI` · `RGB10_A2UI` |
| 2 | `R32F` (black = `-FLT_MAX`, white = `FLT_MAX`, the second texel is `FLT_MIN`) · `RGBA32F` (alpha holds exponents from 2^-32 to 2^31) · `RGBA16F` from `HALF_FLOAT` data · `R16F` from `FLOAT` data (the GL converts) |
| 3 | `R11F_G11F_B10F` from packed words · `RGB9_E5` from packed words · `RGB9_E5` from `FLOAT` data (the GL encodes) · the `R32F` texture again, sampled through a `LINEAR` sampler object |

The last cell is the `R32F` gradient when the implementation makes 32-bit floats filterable (`GL_OES_texture_float_linear`, which
every Mesa driver advertises) and black = `(0, 0, 0, 1)`, the value of an incomplete texture, otherwise.

## The exact path

Normalized 8-bit render targets cannot show a 32-bit value, so `validate()` uses another route: a fragment shader samples the texture and
writes the result unchanged into an **`RGBA32UI`** (or **`RGBA32I`**, for `isampler2D`) renderbuffer, one fragment per texel. Floats go
through `floatBitsToUint()`. `glReadPixels(RGBA_INTEGER, UNSIGNED_INT / INT)` returns the 32-bit words, which are compared with the CPU
model word for word. Both integer render formats are color-renderable in core ES 3.0 (Table 3.13); the float render targets are not.
Every texture is read twice: with `texture()` at the texel center and with `texelFetch()`.

## Central calls

`glTexImage2D` with `RED_INTEGER` / `RG_INTEGER` / `RGBA_INTEGER` formats and `BYTE`, `SHORT`, `INT`, `UNSIGNED_*`, `HALF_FLOAT`, `FLOAT`,
`UNSIGNED_INT_2_10_10_10_REV`, `UNSIGNED_INT_10F_11F_11F_REV` and `UNSIGNED_INT_5_9_9_9_REV` types (Tables 3.2 and 3.3); `glTexParameter`
(filters), `glGenSamplers` / `glBindSampler` (filters of one unit), `glGenerateMipmap`, `glRenderbufferStorage` with integer formats,
`glReadPixels` with integer formats. In GLSL: `usampler2D`, `isampler2D`, `texture`, `textureLod`, `texelFetch`, `floatBitsToUint`.

## Key ideas

- **Integer textures return integers.** `usampler2D` returns `uvec4`, `isampler2D` returns `ivec4`; the lookup of a `R8UI` texel is its
  byte, not `byte / 255`. Missing components are `(0, 0, 1)` for the integer formats too (`R32I` gives `(r, 0, 0, 1)`).
- **They can only be point sampled.** A format that is not *texture-filterable* in Table 3.13 (every integer format, and `R32F`, `RG32F`,
  `RGB32F`, `RGBA32F`) makes the texture incomplete (section 3.8.13) if "either the magnification filter is not `NEAREST` or the minification
  filter is neither `NEAREST` nor `NEAREST_MIPMAP_NEAREST`". The **initial** filters of every new texture are `NEAREST_MIPMAP_LINEAR` and
  `LINEAR`, so a texture that was only given data is incomplete. The demo reads the initial state back and shows that the same texture
  object becomes complete on a unit whose *sampler object* is `NEAREST` (section 3.8.13.1: the sampling state in effect decides).
  `NEAREST_MIPMAP_NEAREST` is allowed, so integer textures may have mip chains and be read with `textureLod` / `texelFetch(level)`.
- **Float is exact too.** A float32 texel comes back unchanged, a half float converts to float exactly, and an `R16F` texture uploaded from
  `FLOAT` data keeps every value that a half can hold (section 3.8.3 converts on upload). `RGBA16F`, `R11F_G11F_B10F` and `RGB9_E5` are
  texture-filterable; the demo verifies a bilinear lookup half a texel to the right of a texel center returns the mean of two texels.
- **Packed floats decode by the book.** `R11F_G11F_B10F`: an unsigned 11-bit float has a 5-bit exponent and 6-bit mantissa, the 10-bit one
  has 5 and 5 (sections 2.1.3, 2.1.4); `RGB9_E5`: `red = reds * 2^(exps - B - N)` with `B = 15`, `N = 9` (section 3.8.17), and the encoder
  of section 3.8.3.2 (clamp, `exp_p = max(-B-1, floor(log2(max_c))) + 1 + B`, refinement, round half up) is implemented on the CPU
  and agrees with what the GL stores when it is handed `FLOAT` data.
- **The sampler precision matters.** A sampler without a precision qualifier is `lowp` in the fragment language (GLSL ES 3.00 4.5.3), and a
  `lowp` lookup need not return more than about 9 bits. The shaders declare `precision highp usampler2D;` and friends.

## Validation (`--validate`)

78 checks, none of which only checks that a call returned:

- **Display (16 checks).** Every texel center of every cell is compared with the CPU color mapping, +-1/255 (an 8-bit result of single-precision
  arithmetic may land on the neighbor of a rounding boundary).
- **Exact words (30 checks).** Each of the 15 textures, read with `texture()` and with `texelFetch()`: all 64 texels x 4 components must equal
  the CPU model exactly, including the hash-filled low bits, `INT_MIN`, `INT_MAX`, `0xFFFFFFFF`, `2^24 + 1`, `FLT_MAX`, `FLT_MIN`, half
  floats with every exponent, all RGB9_E5 exponents and the GL-encoded `RGB9_E5`.
- **Negative controls (2).** The exact words of `2^24 + k` differ from the same values after a trip through `float` at exactly the 32 odd `k` (so
  the comparison can fail, and the values really need 32 integer bits); one flipped bit in the expected `R32F` words is reported as exactly one wrong word.
- **Completeness (5).** Initial filters read back (`NEAREST_MIPMAP_LINEAR`, `LINEAR`); an incomplete `R32I` texture + a `NEAREST` sampler returns the exact
  values; a four-level `R32I` chain with `NEAREST_MIPMAP_NEAREST` returns level 0 through `texture()` and levels 1 to 3 through `textureLod()` and `texelFetch()`.
- **Filterable formats (6) and float32 (2).** `RGBA16F`, `R11F_G11F_B10F`, `RGB9_E5` with a `LINEAR` sampler: texel centers return the texel and the half-texel
  shift returns the mean of two texels, within 2^-10 of the larger magnitude (half-float precision; the filter arithmetic is
  implementation-dependent, 3.8.10.2). `R32F` with a `LINEAR` sampler: the expectation is computed from `metagl::HasExtension("GL_OES_texture_float_linear")`.
  Present (every Mesa driver): the texture is complete and lookups at texel centers return the exact texel. Absent: incomplete, every lookup is `(0, 0, 0, 1)`.
  Only the extension *string* is used, never a function of the extension (there is none). The "absent" branch was exercised by hand with
  `MESA_EXTENSION_OVERRIDE=-GL_OES_texture_float_linear` (see Spec ambiguity).
- **GenerateMipmap (6).** `INVALID_OPERATION` for `R8UI`, `R32I`, `RGBA16UI`, `RGBA32UI` (not filterable) and `RGB9_E5` (not color-renderable), no error for `RGBA8`
  (section 3.8.10.5: the format must be color-renderable *and* texture-filterable).
- **Format errors (6).** `INVALID_OPERATION` for combinations missing from Table 3.2 (`R32I` with `RED_INTEGER`/`FLOAT`, `R32I` with `RED`/`INT`, `R8UI` with
  `UNSIGNED_INT`, `RGBA32F` with `RGBA_INTEGER`, `RGBA8` with `RGBA_INTEGER`/`UNSIGNED_BYTE`), and `RGBA8UI` with `RGBA_INTEGER`/`UNSIGNED_BYTE` is accepted.
- **Sampler types (4).** `texture(usampler2D)` assigned to a `vec4` does not compile, the same with `vec4(...)` does; a `sampler2D` and a `usampler2D`
  on one texture unit make the draw fail with `INVALID_OPERATION` (section 2.12.9.7), and the same program with two units draws.

## Spec ambiguities and how they were resolved

- **What does an incomplete integer lookup return?** ES 3.0 (sections 2.12.9.4 and 3.9.2.1) says "(0, 0, 0, 1) will be returned for a non-shadow
  sampler" without saying whether `1` is an integer or a float; `texelFetch` of an incomplete texture is undefined (2.12.9.2). ES 3.1/3.2 (11.1.3.5) say
  "(0.0, 0.0, 0.0, 1.0), in floating-point" and "if the sampler is declared as a signed or unsigned integer sampler type, undefined values are returned". The demo
  therefore asserts nothing about the value of such a lookup. It prints what the driver does (Mesa returns `(0, 0, 0, 0x3F800000)`: the float `1.0` in an
  integer vector) as a note.
- **Format/type combinations (Mesa deviation).** ES 3.0 3.8.3 and ES 3.2 8.5 ("if format, type and internalformat are each valid, but the combination is not
  listed ... INVALID_OPERATION") require `INVALID_OPERATION`. Mesa answers `INVALID_ENUM` when an integer format meets `FLOAT` (the desktop `EXT_texture_integer`
  rule): two of the five error checks are recorded as XFAIL on Mesa and need a row in `docs/KNOWN_LIMITATIONS.md`.
- **Two sampler types on one unit (Mesa deviation).** Section 2.12.9.7 requires `INVALID_OPERATION`; Mesa compares only the texture *target* (both are
  `TEXTURE_2D`) and raises nothing. XFAIL on Mesa.
- **`R32F` and `LINEAR` (Mesa deviation, hand-tested only).** With `GL_OES_texture_float_linear` hidden by `MESA_EXTENSION_OVERRIDE`, Mesa still samples the texture
  as complete in an ES 3.x context. The check follows the extension string as the spec requires and is XFAIL on Mesa in that configuration; it never runs
  there in the CTest matrix because all four configurations advertise the extension.
- **`RGB9_E5` "as is".** Section 3.8.3.2 allows an implementation to store `UNSIGNED_INT_5_9_9_9_REV` data "as is"; it need not. All test words are chosen so that the decoded
  values are the same whether stored as is or renormalized.

## Pitfalls

- A new integer texture is incomplete until its filters are `NEAREST` (or a sampler object says so). Nothing reports it: the lookup silently returns garbage.
- A sampler type must match the texture's format: `sampler2D` on an integer texture, `usampler2D` on a float or signed-integer texture and so on "will return undefined values for unsupported combinations" (GLSL ES 3.00 8.8).
- `int(float(x))` is not an identity above 2^24, yet Mesa's optimizer folds it into `x` unless the float is made opaque (the display shader adds a uniform
  that is zero). Check such demonstrations on the GPU, not by reading the source.
- A bilinear lookup next to `FLT_MAX` and `-FLT_MAX` can overflow in `a + w * (b - a)` and return NaN even at weight 0 (llvmpipe). The data keeps the two extremes apart.
- Denormal floats may be flushed to zero by an implementation (2.1.3: "preserved or forced to zero"); the data contains only normal values.
- Reading a float texture exactly needs the integer render target: `RGBA32F` as a *render* target needs an extension in core ES 3.0.
