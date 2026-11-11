# es30-etc2-eac — ETC2 / EAC compressed textures

**Minimum:** OpenGL ES 3.0, GLSL ES 3.00 (`#version 300 es`).
On ES 3.1+ contexts the level queries (`glGetTexLevelParameteriv`) are also checked.

## Purpose

ES 3.0 made a family of compressed formats mandatory: every implementation must accept the ten
ETC2/EAC formats of Table 3.19 (ES 3.0 §3.8.6), whose bit-level encoding is defined in Appendix C.
This demo does not load ready-made compressed files. It builds each 64-bit block field by field from
the Appendix C bit layouts. A separate CPU decoder, written from the prose of the appendix,
predicts every texel, and the GPU must agree with it exactly.

The blocks cover every mode of every format:

| format | what the blocks exercise |
|---|---|
| `COMPRESSED_RGB8_ETC2` | individual mode (flip 0), differential mode (flip 1, clamping at 0 and 255), T mode, H mode with base color 1 ≥ base color 2 and with 1 < 2, planar mode (overshoot clamped to 255 and 0, rounding) |
| `COMPRESSED_SRGB8_ETC2` | differential + planar, decoded with equation 3.26 |
| `COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2` | differential with opaque = 1 and opaque = 0 (index 2 is transparent, indices 0/2 add no modifier), T and H with transparent texels, planar with opaque = 0 (treated as 1) |
| `COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2` | transparent differential block; alpha is not gamma-corrected |
| `COMPRESSED_RGBA8_ETC2_EAC` | EAC alpha with multiplier 2, multiplier 0 (modifier × 0) and multiplier 15 (clamps at 0) |
| `COMPRESSED_SRGB8_ALPHA8_ETC2_EAC` | T-mode color + EAC alpha, sRGB on RGB only |
| `COMPRESSED_R11_EAC` / `RG11_EAC` | 11-bit values; multiplier 0 acting as 1/8; clamping at 0 and 2047 |
| `COMPRESSED_SIGNED_R11_EAC` / `SIGNED_RG11_EAC` | two's-complement base, -128 read as -127, clamping at ±1023 |

## Central API calls

- `glCompressedTexImage2D` for most textures, including a 6 × 6 texture that is padded to 2 × 2 blocks
  (32 bytes; the padding texels are irrelevant, Appendix C.1).
- `glCompressedTexSubImage2D`
  - replaces one 4 × 4 block at (8, 4) of the 12 × 8 RGB8 texture;
  - fills the whole level of an immutable `SRGB8_ETC2` texture made with `glTexStorage2D`;
  - writes a 2 × 2 region at (4, 4) of the 6 × 6 texture. That is legal because the region ends at
    the level edge, the relaxation in §3.8.6.
- `glCompressedTexImage3D` creates an 8 × 4 × 3 `TEXTURE_2D_ARRAY`.
- `glCompressedTexSubImage3D` replaces the right block (xoffset 4) of layers 1 and 2
  (zoffset 1, depth 2) of that array.
- `glGetIntegerv(NUM_COMPRESSED_TEXTURE_FORMATS / COMPRESSED_TEXTURE_FORMATS)`,
  `glGetTexParameteriv(TEXTURE_IMMUTABLE_FORMAT)` and, on ES 3.1+,
  `glGetTexLevelParameteriv(TEXTURE_COMPRESSED / INTERNAL_FORMAT / WIDTH / HEIGHT / DEPTH)`.

## Spec sections

- ES 3.0 §3.8.6 Compressed Texture Images, Table 3.19. ES 3.2 §8.7 has the same rules as explicit
  error lists, plus Table 8.17.
- ES 3.0 Appendix C.1, sections C.1.1-C.1.10.
- ES 3.0 §3.8.3 / §3.8.5: border and sub-image bounds. §3.8.4: immutable textures. §3.8.16 / eq. 3.26: sRGB.
  Table 3.24: RED/RG sampling.
- ES 3.0 §6.1.3 with Table 6.29: format queries. ES 3.2 §8.11.3: level queries.

## How the encoding works (summary of Appendix C)

A 4 × 4 block is a big-endian 64-bit word: byte q0 is the most significant. The 128-bit formats are
two such words: EAC alpha first, then color; for RG11, red first, then green. Texel *k = 4x + y* is
pixel a…p of Table C.2, counted column by column.

- **ETC2 color word:** the low 32 bits hold a 2-bit index per texel. The LSB is in bit *k* and the
  MSB in bit *k + 16*.
- **Mode selection:** bit 33 = 0 selects *individual* mode (RGB8 only). Otherwise the three 5-bit
  base values plus 3-bit signed deltas select the mode: if R + dR leaves [0, 31] the block is *T*,
  else if G + dG leaves the range it is *H*, else if B + dB leaves it it is *planar*, else
  *differential*. An encoder must set the bits that T, H and planar leave unused so the overflow
  rule picks the intended mode; `settle()` searches those bits.
- **EAC word:** an 8-bit base, a 4-bit multiplier, a 4-bit table index, then a 3-bit index per texel
  from bit 47 downwards.
  - RGBA8 alpha is `clamp255(base + modifier × multiplier)`.
  - R11 is `clamp(base × 8 + 4 + modifier × multiplier × 8, 0, 2047)`. With multiplier 0 the term is
    the modifier itself (multiplier 1/8).
  - Signed R11 has no +4, clamps to ±1023, and reads base -128 as -127.

## Shaders

- `cell.vert` builds a quad from `gl_VertexID` and interpolates the texel rectangle of a cell.
- `show.frag` samples a `sampler2D` (unit 0) or a `sampler2DArray` (unit 1), selected by a uniform,
  and maps the result for display:
  - RGB as is;
  - RGBA blended over a 6-pixel checkerboard;
  - unsigned R/RG as (r, g, 1 - r);
  - signed values remapped with 0.5 + 0.5·v.
- `probe.frag` runs with the viewport exactly the size of the texture level, so fragment (i, j)
  samples texel (i, j) with NEAREST. It writes `floatBitsToUint(texture(...))` into an RGBA32UI
  target, so the readback holds the sampled floats bit for bit.

## Expected image

The window holds a grid of 31 cells, one per format and mode, each a block (or a whole small
texture) magnified with NEAREST. In order:

1. six RGB8 blocks (individual, differential, T, H, planar, the H block written by the sub-image call);
2. two sRGB8 blocks;
3. six punchthrough blocks, some with the checkerboard showing through transparent texels;
4. one sRGB punchthrough block;
5. three RGBA8 EAC blocks over the checkerboard;
6. one sRGB8_ALPHA8 block;
7. three R11, three signed R11, one RG11 and one signed RG11 cell (purple, tan and red/blue tones);
8. the three layers of the array (8 × 4 each);
9. the 6 × 6 texture.

The grid adapts to the surface: for example 6 × 6 cells at 256 × 256 and 4 × 8 at 333 × 777.

## How it is validated

1. **Exact readback.** Every texel of every texture and layer, 564 in total, is compared with the
   decoder.
   - 8-bit formats: within 0.45 of the 8-bit code, so the GPU returns exactly code/255 (Appendix C:
     values "need to be divided by 255").
   - R11/RG11: within 0.45 of the 11-bit value (/2047, or /1023 for signed). Appendix C says the
     11-bit value is the exact representation and forbids truncating it below 11 bits. The 16-bit
     replication the appendix allows still meets this: measured at most 0.03 units.
   - sRGB RGB channels: within 1/255. Equation 3.26 has no stated accuracy, real decoders use
     tables, and ±1 is the suite's 8-bit policy. The measured deviation is at most 0.47.
   - Missing channels must be 0 and alpha must be 1 (Table 3.24).
2. **Worked examples of the spec text.** These are hard-coded numbers from Appendix C, independent
   of the decoder:
   - individual base (238, 51, 136);
   - (231, 8, 16) + 29 clamped to (255, 37, 45);
   - T distance index 5 = 32;
   - the differential example (231, 33, 24) / (198, 49, 24) on an opaque = 0 punchthrough block,
     where index 0 adds nothing;
   - EAC alpha 83;
   - R11 668 and, with multiplier 0, 818;
   - signed R11 320 and 470;
   - base 255 reaching exactly 1.0.
3. **The picture.** One probe per texel per cell, at the pixel nearest the texel center: 564 at
   256 × 256. The probe is compared with a CPU model of `show.frag`: exact for opaque RGB8 codes,
   ±1 for anything computed (sRGB, 11-bit values, the checker blend). If the probe falls within 2 %
   of a texel edge, the neighbouring texel is also accepted. The demo reports SKIP_UNSUPPORTED if
   the surface is too small for 3 pixels per texel.
4. **State.**
   - The format list must contain all ten formats, and the count must be at least 10.
   - `TEXTURE_IMMUTABLE_FORMAT` must be TRUE for the `glTexStorage2D` texture only.
   - On ES 3.1+, the level queries must report COMPRESSED, the internal format and the size of all
     12 images.
5. **Errors.** 25 calls, each with the error the spec names:
   - **INVALID_VALUE:** imageSize too small, too large, not padded, or negative; border 1;
     a sub-image past the level or past the last layer.
   - **INVALID_OPERATION:**
     - x/y offset 2;
     - width or height 2 that does not reach the level edge;
     - a sub-image format different from the image's (SRGB8 and RGBA8 into RGB8, R11 into the RGB8 array);
     - ETC2/EAC with `TEXTURE_3D` in `CompressedTexImage3D` or `CompressedTexSubImage3D`;
     - `CompressedTexImage2D` on an immutable texture.
   - **INVALID_ENUM:** a 2D command with a `TEXTURE_2D_ARRAY` or `TEXTURE_3D` target (stated
     explicitly in ES 3.2 §8.7).

   Every rejected call carries all-white blocks. The exact readback runs afterwards, so it also
   proves each rejected command had no effect (ES 3.0 §2.5). On ES 3.1+, the texture name on which
   every upload was rejected still has a level 0 width of 0.
6. **Negative controls.** These decoders must disagree with the readback; each does, in 6 to 40
   texels:
   - the next row of Table C.6;
   - subtracting the differential deltas;
   - swapped MSB/LSB of the index;
   - reversed H-mode ordering;
   - the literal "B + dG" (in RGB8 and punchthrough);
   - planar rounding without +2;
   - R11 multiplier 0 as 0;
   - signed base -128 kept;
   - transparent texels keeping RGB.

## Run

```
./build/debug/bin/es30-etc2-eac                       # window
./build/debug/bin/es30-etc2-eac --headless --validate --capture out.png
./build/debug/bin/es30-etc2-eac --headless --validate --width 333 --height 777
ctest --preset debug -R es30-etc2-eac
```

## Pitfalls

- **The appendix has a misprint.** The summary formula for subblock 2 in differential mode reads
  `extend_5to8bits(R + dR, G + dG, B + dG)` in C.1.1 and again in C.1.9. Both the prose ("modifying
  the five-bit codewords R, G and B by the codewords dR, dG and dB") and the worked example
  (B = 3, dB = 0, dG = 2 → blue 24) require **B + dB**. The demo decodes with dB. One negative
  control shows that the literal formula disagrees with every driver.
- **The signed R11 multiplier-0 example is also inconsistent.** It says "base codeword of 65" but
  computes with 60 (480 = 60 × 8 → 470). The demo uses 60.
- **Without mipmaps the textures need `TEXTURE_MIN_FILTER` NEAREST or LINEAR.** The default
  NEAREST_MIPMAP_LINEAR leaves a single-level texture incomplete.
- **imageSize counts whole blocks, including padding.** A 6 × 6 or 5 × 5 RGB8 image is 2 × 2
  blocks = 32 bytes. Pixel-store state (`UNPACK_ALIGNMENT`, ...) is ignored for compressed data.
- **Many drivers decompress ETC2 on upload** (Mesa does on GPUs without native ETC2). The results
  must still be the Appendix C values. For R11 that means at least 11 bits, never 8.
- **ETC2/EAC is two-dimensional.** `TEXTURE_2D_ARRAY` is the only 3D target in ES 3.0;
  `TEXTURE_CUBE_MAP_ARRAY` is added in ES 3.2. `TEXTURE_3D` is always INVALID_OPERATION.
- **There is no ES internal-format query for compressed formats.** `glGetInternalformativ` accepts
  only renderable formats (ES 3.0 §6.1.15). `TEXTURE_COMPRESSED` exists only as a texture level
  query, and only from ES 3.1.
