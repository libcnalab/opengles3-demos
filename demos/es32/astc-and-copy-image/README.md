# es32-astc-and-copy-image

**API introduced in:** OpenGL ES 3.2 · **Minimum context:** ES 3.2 · **Shaders:** GLSL ES 3.20 ·
**Run:** `es32-astc-and-copy-image [--headless --validate] [--window] [--width W --height H]`

Spec numbering follows ES 3.2 (§8.7 Compressed Texture Images, Appendix C.1, §16.2.2 Copying Between Images, §8.17 Texture Completeness,
§8.21 sRGB Texture Color Conversion, §8.23 for the reinterpretation rule) and the "ASTC Compressed Texture Image Formats" chapter of the
Khronos Data Format Specification 1.1 (chapter 18, cited as DFS 18.x), which Appendix C.1 of ES 3.2 refers to for the ASTC encoding.

## Purpose

ES 3.2 made two image features core that earlier versions had only as extensions.

**ASTC LDR textures** (`KHR_texture_compression_astc_ldr` before). Every ES 3.2 implementation must accept the 28 ASTC LDR formats of Table 8.17:
14 block footprints from 4×4 to 12×12 texels, each in a linear (`COMPRESSED_RGBA_ASTC_*`) and an sRGB (`COMPRESSED_SRGB8_ALPHA8_ASTC_*`) variant.
Every block is 128 bits, so the footprint chooses the bit rate (8 bits per texel at 4×4, 0.89 at 12×12). Inside a block the encoder picks the weight grid,
the weight precision, the colour endpoint mode and the number of partitions independently.

The demo does not use an encoder. It builds the blocks bit by bit, following the layout of DFS 18.9–18.23:

- **void-extent blocks**: one constant UNORM16 colour per block;
- **single-partition blocks** with direct LDR endpoints: luminance (CEM 0), luminance + alpha (CEM 4), RGB (CEM 8), RGBA (CEM 12),
  including the CEM 8 case where `v1+v3+v5 < v0+v2+v4` switches the decoder to **blue contraction**;
- **2-bit weights on a 4×4 grid and 3-bit weights on a 3×3 grid**. A grid smaller than the footprint is stretched over the block with the
  bilinear **weight infill** of DFS 18.18, also onto the 5×4, 6×6 and 12×12 footprints;
- **illegal encodings** that must decode to the error colour: an HDR void-extent block (LDR profile), the reserved block mode `0000` and a 5×4 weight grid in a 4×4 footprint.

A CPU decoder written from the same chapter, and implementing only the modes used here, predicts every texel. Each texture is read back
texel by texel with `texelFetch` in a compute shader and compared.

**glCopyImageSubData** (`EXT_copy_image` before) copies a box of texels between two images like a `memcpy`: no filtering, no format conversion.
The images can be textures of any target except buffer textures, or renderbuffers. The two formats must be "compatible": the same format, the same view class
of Table 16.4 (for example RGBA8UI and R32F, both 32 bits), or an uncompressed format whose texel size equals the block size of a compressed format (Table 16.3).
In the last case one texel of the uncompressed image becomes one whole compressed block, and back. Uses are texture atlases and streaming (assemble an atlas from pre-compressed tiles
without decompressing them), moving data between a renderbuffer and a texture, rearranging array layers and cube faces, and viewing the bits of an image in another format.

## What the image shows

Two panels fill the surface (`--width` / `--height`, default 256 × 256): side by side on a wide surface, one above the other on a tall one,
each half the surface minus a margin of `max(2, min(w, h) / 32)` pixels. Surfaces smaller than 128 × 128 report `SKIP_UNSUPPORTED`, because a panel must be
at least 32 pixels wide to show every mosaic texel. Each panel is a texture stretched over its rectangle with `texelFetch`, so every texel is a flat rectangle.
Only RGB is shown; alpha is checked by the read-back passes.

- **Left / top: the tile set**, the 16 × 16 `RGBA_ASTC_4x4` atlas of 16 hand-made blocks. From the bottom row up: sand (an RGB ramp), sand, deep sea and white
  (void extents); deep water, a wave (RGBA, alpha 255 → 63), the gold-and-blue glitter (a checker of weights), misty water (blue contraction); the sun's reflection,
  the horizon sky, the sun (void extent), a cloud (luminance with a 3×3 grid infilled to 4×4); and three sky gradients and a luminance + alpha cloud.
- **Right / bottom: a sunset over the sea**, the 32 × 32 `RGBA_ASTC_4x4` mosaic. It is assembled from 8 × 8 blocks with 64 `glCopyImageSubData` calls:
  48 copy a 4 × 4 texel block from the atlas (compressed → compressed), and 16 copy one `RGBA32UI` texel from the 8-texel "palette" texture, whose texels hold raw ASTC blocks
  (uncompressed → compressed: the waves running the other way, the sun's glow, stars, a cloud edge, foam, pebbles, a rock and dark water). The sky darkens from gold at the horizon to indigo,
  and a square sun with an orange glow stands over a reflection of gold glitter. The beach is at the bottom.

## Central calls

`glCompressedTexImage2D`, `glCompressedTexSubImage2D`, `glTexStorage2D` with a compressed format,
`glGetIntegerv(NUM_COMPRESSED_TEXTURE_FORMATS / COMPRESSED_TEXTURE_FORMATS)`, and
`glCopyImageSubData(srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName, dstTarget, dstLevel, dstX, dstY, dstZ, srcWidth, srcHeight, srcDepth)`.
meta-gl splits it into typed overloads (texture ↔ texture with `ImageCopyTextureTarget`, texture ↔ renderbuffer and renderbuffer ↔ renderbuffer with `RenderbufferId`, which pass
`RENDERBUFFER` and level 0). The original raw-name overload with `TextureTarget` is used only to provoke the `INVALID_ENUM` errors.
GLSL: `texelFetch(sampler2D / usampler2D, ivec2, 0)`, `textureSize`, `floatBitsToUint`.

## Key ideas

### ASTC

- **Block layout** (DFS 18.9). Bits are little-endian: byte 0 holds bits 0..7. Bits 0..10 hold the block mode, bits 11..12 the partition count − 1, and bits 13..16 the CEM (single partition).
  The endpoint values grow upward from bit 17. The weights grow **downward from bit 127**: "Bit n in the stream is thus bit 127-n in the block" (DFS 18.16).
- **Block mode** (Table 82). For `bits[1:0] != 0`: `bits[3:2] = 00` gives a grid of (bits 8..7 + 4) × (bits 6..5 + 2), and `bits[3:2] = 11` with bit 8 set gives (bit 7 + 2) × (bits 6..5 + 2).
  The weight range is `ρ = (bit 1, bit 0, bit 4)` with the precision bit P in bit 9 (Table 81): range 0..3 is 2 bits, range 0..7 is 3 bits.
- **Legality** (DFS 18.24). The weight grid may not exceed the footprint, needs 24..96 bits and at most 64 weights. A 2×2 grid of 3-bit weights (12 bits) is therefore *illegal*.
  That is why the small grids here are 3×3 × 3 bits (27 bits) and 4×4 × 2 bits (32 bits).
- **Endpoint range** (DFS 18.22). It is not stored: it is the largest range whose encoding fits the bits left over. All blocks here leave at least 8 bits per endpoint value,
  so the range is 0..255 and the values are stored as plain bytes.
- **Interpolation** (DFS 18.19). The endpoints are expanded to 16 bits, `(c << 8) | c` (or `(c << 8) | 0x80` for the RGB channels of an sRGB format), and
  `C = floor((C0 (64 − i) + C1 i + 32) / 64)` with the infilled weight `i` in 0..64. For sRGB the top 8 bits of `C` go to the sRGB → linear conversion of §8.21.
  For linear formats the result depends on the *decode mode*.
- **The decode mode is not specified by ES 3.2.** decode_float16 returns `C / 65536` rounded toward zero to FP16 (1.0 for 65535), and decode_unorm8 returns the top 8 bits.
  Selecting the mode is the business of an extension (`EXT_texture_compression_astc_decode_mode`), not of the core. The demo therefore uses only data for which both modes give the same 8-bit value.
  Every endpoint pair differs by a multiple of 64 in every channel, so every interpolated `C` is exactly `v · 257`. The validation asserts this margin for all 12 350 decoded texel channels
  (worst case 0.124 of an 8-bit step) and compares the GPU at 8-bit precision.
- **The error colour** (DFS 18.5) is opaque magenta. For sRGB it is exactly (0xFF, 0x00, 0xFF, 0xFF). For a linear decode the specification allows either magenta (1, 0, 1, 1)
  or four NaNs, so the linear check accepts exactly these two results and the sRGB check accepts only magenta. The DFS also says that applications must not rely on invalid encodings to *produce*
  the error colour, because future extensions may define them. The demo uses them only to test the LDR-profile rule.
- **imageSize and the footprint** (§8.7). A w × h level needs `ceil(w / bw) · ceil(h / bh) · 16` bytes; anything else is `INVALID_VALUE`. Edge blocks are partial: a 17 × 10 texture with
  6×6 blocks has 3 × 2 blocks, and only the texels inside the texture are visible.
- **Sub-image rules** (§8.7). Offsets must be multiples of the block size. Width and height must be multiples too, *unless the region ends at the texture's edge*
  (`width + xoffset == TEXTURE_WIDTH`). Violations are `INVALID_OPERATION`, while an inconsistent imageSize is `INVALID_VALUE`. "The contents of any block … that does not intersect the area being modified are preserved."

### glCopyImageSubData

- **Units are texels of the source image**, even for compressed formats. Copying one RGBA32UI texel into an ASTC 4×4 texture writes 4 × 4 texels. Copying 16 × 16 ASTC texels into RGBA32UI writes 4 × 4 texels.
- **Compatibility** (§16.2.2): the same format, the same row of Table 16.4 (view classes: RGBA8UI, R32F, RGBA8, SRGB8_ALPHA8 … are all `VIEW_CLASS_32_BITS`; `RGBA_ASTC_4x4` and `SRGB8_ALPHA8_ASTC_4x4`
  share `VIEW_CLASS_ASTC_4x4_RGBA`), or the same row of Table 16.3 (128-bit texels RGBA32UI / RGBA32I / RGBA32F with every ASTC format and RGBA8_ETC2_EAC / RG11_EAC; 64-bit texels with RGB8_ETC2 / R11_EAC / punch-through).
  ASTC 4×4 and RGBA8_ETC2_EAC both use 128-bit 4×4 blocks but are *not* compatible: they are neither the same format nor in the same class.
- **Reinterpretation** (§16.2.2 → §8.23). The bits are kept, as if the source texel were read with `ReadPixels` into memory and written with `TexSubImage` in the destination format.
  An RGBA8UI texel (R, G, B, A) becomes the float whose four bytes in memory are R, G, B, A. A linear ASTC block copied into an sRGB ASTC texture decodes as sRGB.
- **Depth formats** appear in no class of Table 16.4. A depth or depth-stencil format is therefore compatible only with itself. A DEPTH24_STENCIL8 copy carries the stencil bits along.
  ES 3.2 has no separate depth/stencil rule for this command; the format rule is the whole restriction.
- **Layers and faces** are selected by z. "Slices of a two-dimensional array, cube map array, or three dimensional texture, or faces of a cube map texture are all compatible …
  multiple slices or faces may be copied … with a single call". Cube faces are numbered in the order of Table 8.20 (+X, −X, +Y, −Y, +Z, −Z), and the target is always `TEXTURE_CUBE_MAP`: the face targets are `INVALID_ENUM`.
- **Completeness** (§16.2.2): "if either object is a texture and the texture is not complete (as defined in section 8.17, but ignoring format-based completeness rules)" → `INVALID_OPERATION`.
  A level-0-only texture with the default `NEAREST_MIPMAP_LINEAR` filter is rejected. An RGBA32UI texture with `LINEAR` filtering is accepted: it cannot be *sampled* (not filterable), but that is a format-based rule.

## How it is validated

`--validate` records 137 checks on every driver.

- **Picture**: every pixel of the frame, at any surface size. Outside the panels the pixel must be the clear colour. Inside, it must be the CPU-decoded colour of the texel that the pixel centre selects
  (`(x + 0.5 − origin) · texels / size`, as in `panel.frag`); within 0.002 texel of a texel edge either neighbour is accepted. The tolerance is ±1 per channel, because the decoded value need not be exactly
  `v / 255` (decode_float16 rounds toward zero) and §2.3.5.2 allows either of the two closest 8-bit values. Further checks require that every atlas and mosaic texel is on screen, and (negative control) that
  a mosaic with two blocks exchanged would fail.
- **Decoding**: every texel of the linear and sRGB atlas, the 6×6 texture (17 × 10), the 5×4 and 12×12 textures after their sub-image edits, the mosaic, the sRGB copy and the error blocks.
  The texture unit's float result (as raw bits) must round to the CPU's 8-bit value. For sRGB the linear result is converted back with the inverse of equation 8.14 and must give the 8-bit code.
  Error texels must be magenta (or NaN for linear). `textureSize` is checked too. The detail line reports how the driver actually decoded (exactly `(C >> 8) / 255`, exactly decode_float16, or neither).
- **Test data**: decode_float16 and UNORM8 agree within 0.375 for every texel channel, and the CPU decoder classifies the three illegal blocks as errors.
- **Negative controls**: the sRGB atlas does not match the linear prediction (236 of 256 texels differ). A prediction with one block's weights mirrored mismatches exactly that block's 16 texels.
  The RGBA32UI block texels do not match a transposed block order. The copied RGBA8 texels differ from the destination's original content. The reinterpreted floats do not match the byte-reversed words.
- **Raw bits**: the ASTC → RGBA32UI copies (atlas, mosaic, 12×12 with edge blocks) and the ETC2 / EAC copies (RGBA8_ETC2_EAC → RGBA32UI, RGB8_ETC2 → RGBA16UI → RGB8_ETC2 → RGBA16UI) must reproduce the uploaded bytes exactly.
- **Uncompressed copies**: RGBA8 textures, the renderbuffer, every 2D array layer and every cube face are read back with `glReadPixels` and must match the CPU model texel for texel
  (copied region and untouched rest). The renderbuffer's cleared colours are read back before the copy and used as the reference, so the clear's float → 8-bit conversion does not matter.
- **Depth**: DEPTH_COMPONENT32F values are copied bit-exactly. For DEPTH24_STENCIL8 a copied texel samples exactly like its source texel (depth through `DEPTH_STENCIL_TEXTURE_MODE = DEPTH_COMPONENT`,
  with a sanity bound of 2⁻²⁰ against D / (2²⁴ − 1); the spec gives no precision for this conversion, and the measured maximum error is printed). The stencil index read through `STENCIL_INDEX` must be exact.
- **Errors**: 29 `glCopyImageSubData` errors (2 `INVALID_ENUM`, 14 `INVALID_VALUE`, 13 `INVALID_OPERATION`), 11 compressed-upload errors and the imageSize rule for all 14 footprints. Each error is quoted in the check text.
  After the rejected calls the RGBA8 destination and the ASTC mosaic are verified unchanged (§2.3.1).

Matrix: radeonsi, llvmpipe and zink PASS (137/137) at 256 × 256, 800 × 800, 640 × 480, 333 × 777, 128 × 128 and 1000 × 130. softpipe (native ES 3.1) reports
`SKIP_UNSUPPORTED`. There is no `es-min` run for an ES 3.2 demo.

**Measured driver behaviour** (informational, not asserted). radeonsi returns every linear ASTC texel as exactly `(C >> 8) / 255`, which is UNORM8 decoding: Mesa decompresses ASTC
in software for this GPU. llvmpipe and zink return values that round to the same 8-bit code but match neither `v / 255` nor decode_float16 bit for bit for most texels.
All three return magenta, not NaN, for the error blocks.

## Pitfalls

- **Weight bits are reversed**: weight 0 occupies bits 127, 126, … with its *least* significant bit at bit 127. Writing the weights from bit 127 in normal bit order gives plausible-looking but wrong images.
- **Blue contraction triggers by itself.** With CEM 8 / 12 the decoder compares the endpoint sums. If the second endpoint is darker than the first, it contracts the blue channel and swaps the endpoints.
  An encoder must order the endpoints (and mirror the weights) unless it wants that.
- **Small grids can be illegal.** Fewer than 24 weight bits makes the block an error block, not a coarse one.
- **Do not compare linear ASTC floats exactly.** Without the decode-mode extension the result can be UNORM8 or FP16 (or, in practice, something in between). Compare at 8-bit precision with data that makes the modes agree.
- **CopyImageSubData counts source texels.** For compressed ↔ uncompressed copies the region in the other image is a factor of the block size larger or smaller.
  Offsets in the compressed image must be block-aligned, and sizes must be block multiples unless they reach the edge (`INVALID_VALUE` otherwise, unlike `INVALID_OPERATION` for `CompressedTexSubImage`).
- **Mutable textures need a non-mipmap filter**, or mipmaps, before they can be copied. A copy from a freshly `glTexImage2D`'d level 0 fails with `INVALID_OPERATION` until `MIN_FILTER` is `NEAREST` or `LINEAR`.
- **Deleted names come back.** A name freed with `glDeleteTextures` may be returned by the next `glGenTextures`. To test the "not a texture" error, delete the name right before the call.
- **Target / object mismatch is ambiguous.** A 2D texture named with `TEXTURE_2D_ARRAY` matches both "target does not match the type of the object" (`INVALID_ENUM`) and "name does not correspond to a valid …
  texture object according to the corresponding target parameter" (`INVALID_VALUE`). The demo does not test that case.
- **Renderbuffer level errors** ("For renderbuffers, this value must be zero") cannot be expressed through meta-gl's typed overloads. They take no level for a renderbuffer, and the raw overload's `TextureTarget` has no
  `RENDERBUFFER`. That error is not tested; it is reported as a meta-gl gap.
