# es20-copy-and-read-pixels

**API introduced in:** OpenGL ES 2.0 (`PACK_ROW_LENGTH` / `PACK_SKIP_*` and the R8/RG8 formats: ES 3.0) ·
**Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

Two ways get pixels out of a framebuffer: **`glReadPixels`** copies them into client memory, and
**`glCopyTexImage2D` / `glCopyTexSubImage2D`** copy them into a texture without leaving the GPU. Both take a
rectangle `(x, y, width, height)` with its origin at the **lower left** corner of the *read* framebuffer and
both convert the stored values on the way. This demo shows which formats you may ask for, how rows are laid
out in memory (`PACK_ALIGNMENT`, `PACK_ROW_LENGTH`, `PACK_SKIP_*`), which pixel is where, which texture
formats a copy can produce, what is not allowed, and what is explicitly undefined (reading outside the
framebuffer).

Every read and copy starts from a **test chart**: a 64 x 64 RGBA8 framebuffer whose pixel `(x, y)` is
`(4x, 4y, checker, 128 + x + y)` in 8-bit steps. Every pixel is different, so a misplaced rectangle, flipped
rows or swapped channels cannot pass unnoticed.

## What the image shows

The default framebuffer is divided into equal display cells, reading order top row first: 4 x 3 at the default
256 x 256, or 3 x 4, 5 x 2 or 2 x 5, whichever gives the largest cells (the smallest cell edge is maximized). Each
image is centred in its cell and magnified by the **largest integer factor** `m` that fits (cells of 128 pixels or more keep a 4-pixel gutter), so
every window pixel shows exactly one texel and the cells are compared exactly with the magnified texture. At
256 x 256 the chart is shown 1:1 and the 32 x 24 blocks 2x (the factors in the table); at 800 x 800 they are 3x and 6x.
**Minimum surface:** cells of at least 64 x 64 (for example 256 x 192, 192 x 256, 320 x 128 or 128 x 320); smaller
surfaces report `SKIP_UNSUPPORTED`.

| Cell | Content | How it was made |
|---|---|---|
| 1 | the chart | `glBlitFramebuffer` of the chart framebuffer (an RGBA8 renderbuffer) |
| 2 | the chart again | `glCopyTexImage2D(RGBA, 0, 0, 64, 64)` into a texture, sampled |
| 3 | a solid blue-gray texture with two patches | `glTexImage2D` of the solid color, then two `glCopyTexSubImage2D` (a 16 x 16 and a 24 x 24 block of the chart) |
| 4 | a 32 x 24 block of the chart, 2x | `glCopyTexImage2D` with the **sized** format `RGBA8` |
| 5 | the same block, 2x | internal format `RGB` (alpha dropped) |
| 6 | red only, as gray, 2x | `LUMINANCE`: sampled as `(L, L, L, 1)` with `L = R` |
| 7 | alpha as gray, 2x | `ALPHA`: sampled as `(0, 0, 0, A)`, shown through `u_mode = 1` |
| 8 | red only, 2x | `LUMINANCE_ALPHA` (the alpha half is checked numerically) |
| 9 | red ramp, 2x | `R8` |
| 10 | red + green, 2x | `RG8` |

Cell 1 is blitted (NEAREST, magnified `m` times) with the chart's alpha (128..254), so an image viewer shows it paler than cell 2, which is
sampled with alpha forced to 1. All copies are made while `READ_FRAMEBUFFER` is the chart and `DRAW_FRAMEBUFFER` is the default framebuffer,
so the copies can only contain the chart (the default framebuffer never holds the chart at that place, and the
copies equal it exactly).

## Central calls

| Call | Role |
|---|---|
| `glReadPixels(x, y, w, h, format, type, data)` | read a rectangle of the read buffer; rows run bottom to top |
| `glPixelStorei(PACK_*, n)` + `glGetIntegerv` | how the rows are laid out in `data` |
| `glGetIntegerv(IMPLEMENTATION_COLOR_READ_FORMAT / _TYPE)` | the second pair (besides `RGBA`/`UNSIGNED_BYTE`) that the current read buffer accepts |
| `glReadBuffer` | which color buffer is read (`NONE` and an empty attachment are errors) |
| `glCopyTexImage2D(target, level, internalformat, x, y, w, h, border)` | define a texture level from the framebuffer |
| `glCopyTexSubImage2D(target, level, xoffset, yoffset, x, y, w, h)` | overwrite a rectangle of an existing level |

## Spec

- ES 3.0 **§4.3.2 Reading Pixels**: the two accepted combinations, `IMPLEMENTATION_COLOR_READ_*`, "If any of
  these pixels lies outside of the window ... the values obtained for those pixels are undefined", Table 4.4
  (pack parameters), Table 4.5 (conversion to `type`), §4.3.1 read buffer.
- ES 3.0 **§3.7.2 Transfer of Pixel Rectangles**: equation 3.13 (row pitch), the `SKIP_*` / `ROW_LENGTH`
  mechanism (Figure 3.6), §3.7.2.2 (a packed type fixes the number of components).
- ES 3.0 **§3.8.5 Alternate Texture Image Specification**: Table 3.15 (the `format`/`type` used during the
  copy), Table 3.16 (valid source/destination component combinations), Tables 3.17/3.18 (effective formats),
  the error list at the end of the section; Table 3.24 (what LUMINANCE/ALPHA/LUMINANCE_ALPHA return when sampled).
- ES 3.0 §4.4.4.4 (incomplete framebuffer), §2.1.6.2 equation 2.3 (float -> 8 bits).

## Shader concepts

`chart.frag` writes the test chart from `gl_FragCoord` (pixel centers are at `i + 0.5`, so
`ivec2(gl_FragCoord.xy)` is the pixel index; the output is `value / 255.0`). `quad.vert` generates a rectangle
from `gl_VertexID`. `show.frag` samples a copied texture with `NEAREST` filtering and, depending on `u_mode`,
shows `(r, g, b)`, alpha as gray, or the raw RGBA used by the numeric checks. Texture formats lacking
components are expanded by the GL, not by the shader: missing green/blue read as 0 and a missing alpha as 1,
except `LUMINANCE` (`(L, L, L, 1)`) and `ALPHA` (`(0, 0, 0, A)`), Table 3.24.

## How it is validated

### glReadPixels

- **Chart.** The whole chart read with `RGBA`/`UNSIGNED_BYTE` equals the analytic image within **+-1** (the
  shader output is a float converted to 8 bits, eq. 2.3 allows one of the two nearest integers). This is the
  ground truth `S` for everything else.
- **Sub-rectangles** (a 1 x 1 corner at each end, a 13 x 7 block, a one-pixel column, a one-pixel row,
  a 32 x 32 block) must equal that block of `S` **exactly**: reading moves stored values, it does no arithmetic.
  Negative controls: the flipped rows or a block shifted by one pixel are rejected.
- **Reads that leave the framebuffer** (two corners of the chart framebuffer, one corner of the 256 x 256
  default framebuffer). The values of the outside pixels are *undefined* (§4.3.2.1): the demo does **not**
  assert anything about them (it only prints how many this driver left untouched) and asserts that the
  call raises no error and that **every pixel inside the framebuffer is correct** in the same call.
- **PACK_ALIGNMENT.** A 3 x 5 block read as RGBA/UNSIGNED_BYTE has 12-byte rows for alignment 1, 2 and 4 and
  **16-byte rows for alignment 8** (equation 3.13: `k = (a/s) * ceil(s n l / a)` when `s < a`): all 15 pixels
  must sit at the offsets that layout gives. Negative control: the alignment-8 buffer interpreted as tight
  is wrong for 12 of 15 pixels. `glGetIntegerv(PACK_ALIGNMENT)` returns every value set; the initial value is 4.
- **PACK_ROW_LENGTH / SKIP_PIXELS / SKIP_ROWS.** A 5 x 4 block with row length 8, skip 2 pixels, skip 1 row,
  alignment 4 lands at byte 40 with a 32-byte pitch; negative control: read as a tight block it is wrong.
- **The implementation-chosen pair.** `IMPLEMENTATION_COLOR_READ_FORMAT/TYPE` are queried and used, never
  assumed. The values arrive as integers and are mapped to meta-gl's typed enumerators by comparison (no
  casts). (a) For the default framebuffer and the RGBA8 chart the pair's read must agree with the
  `RGBA`/`UNSIGNED_BYTE` read within one step of the pair's component width. (b) Eight renderbuffer formats
  (RGBA8, RGB8, RGB565, RGBA4, RGB5_A1, RGB10_A2, RG8, R8) are cleared to `(0.8, 0.4, 0.2, 0.6)`, read with
  *their* pair, decoded (including 565, 4444, 5551, 2_10_10_10_REV, `RG`, `RED`, and `BGRA` order) and compared
  with the color. Tolerance per component = one step of the storage width plus one step of the pair's width
  (each float -> fixed conversion is within one step, eq. 2.3); components the storage does not have
  (`RENDERBUFFER_*_SIZE` = 0) must read as 0, 0, 0, **1** (§4.3.2.1). (c) For every pair the row layout is
  checked for alignments 1, 2, 4, 8 against the tightly packed read, group by group. (d) Defined error:
  querying the pair with `READ_BUFFER = NONE` -> `INVALID_OPERATION`.
- **Observed pairs** (identical on radeonsi, llvmpipe and zink): default framebuffer `BGRA`/`UNSIGNED_BYTE`
  (`0x80E1` is `GL_BGRA_EXT` from `EXT_read_format_bgra`, which Table 3.2 does not contain; the demo only
  decodes the pair the implementation names, it does not depend on the extension), the chart and the RGBA8 and
  RGB8 renderbuffers `RGBA`/`UNSIGNED_BYTE`,
  RGB565 `RGB`/`UNSIGNED_SHORT_5_6_5`, RGBA4 `RGBA`/`UNSIGNED_SHORT_4_4_4_4`, RGB5_A1 `RGBA`/`..._5_5_5_1`,
  RGB10_A2 `RGBA`/`UNSIGNED_INT_2_10_10_10_REV`, RG8 `RG`/`UNSIGNED_BYTE`, R8 `RED`/`UNSIGNED_BYTE`. A pair
  the demo cannot decode is reported with a note and skipped, never silently passed.

  | Pair | row of 3 pixels, alignment 1 / 2 / 4 / 8 (bytes) |
  |---|---|
  | `RGBA`/`UNSIGNED_BYTE` | 12 / 12 / 12 / 16 |
  | `RGB`/`UNSIGNED_SHORT_5_6_5` (**the "3-pixel RGB row"**) | 6 / 6 / **8** / 8 |
  | `RG`/`UNSIGNED_BYTE` | 6 / 6 / 8 / 8 |
  | `RED`/`UNSIGNED_BYTE` | 3 / 4 / 4 / 8 |
  | (`RGB`/`UNSIGNED_BYTE`, if an implementation chose it) | 9 / 10 / 12 / 16 |

  The padding bytes themselves are not specified, so they are **not** asserted (the demo prints how many a
  driver left untouched); the demo does assert that nothing is written behind the last row.
- **Defined errors.** `RGB` + `UNSIGNED_SHORT_4_4_4_4` and `RGBA` + `UNSIGNED_SHORT_5_6_5` (the packed type fixes
  the component count, §3.7.2.2); `RGBA_INTEGER` + `UNSIGNED_INT` on a fixed-point buffer (§4.3.2.1);
  `READ_BUFFER = NONE`; `READ_BUFFER = COLOR_ATTACHMENT1` with nothing attached; `PACK_ALIGNMENT 3` and
  `PACK_ROW_LENGTH -1` -> `INVALID_VALUE` (`PACK_ALIGNMENT` stays 4). Each family is followed by a control that works again.

### glCopyTexImage2D / glCopyTexSubImage2D

Per copy case (whole chart, and a 32 x 24 sub-rectangle for each internal format), three independent views:

1. **Exact texels.** For renderable formats (`RGBA`, `RGBA8`, `RGB`, `R8`, `RG8`) the texture is attached to a
   framebuffer and read with `glReadPixels`: every texel must equal the source pixel mapped by the format, with
   tolerance **0** (the copy goes through the same 8-bit values).
2. **Sampling.** The texture is rendered 1:1 into a scratch target and read back: `RGBA` -> `(R,G,B,A)`,
   `RGB`/`R8`/`RG8` -> dropped components 0 and alpha 1, `LUMINANCE` -> `(R,R,R,1)`, `ALPHA` -> `(0,0,0,A)`,
   `LUMINANCE_ALPHA` -> `(R,R,R,A)`. Tolerance +-1 for the shader's float -> 8-bit conversion. (`LUMINANCE`,
   `ALPHA` and `LUMINANCE_ALPHA` are not color-renderable, so view 1 does not exist for them.)
3. **The cell** on the screen (magnified by its integer factor `m`; 2x for the 32 x 24 textures at 256 x 256) equals
   view 2's expectation for every window pixel: pixel `(i, j)` shows texel `(i / m, j / m)`. The blitted chart cell
   is compared exactly (a NEAREST blit with an integer magnification copies stored values), the patched texture
   with +-1.

`glCopyTexSubImage2D`: two patches replace exactly their rectangles of the 64 x 64 texture (exact), all other
texels keep the base color; negative control: the unpatched base color differs in more than 400 texels.
A reference taken from a block shifted by 4 pixels is rejected.

**Defined errors** (§3.8.5): from an **RGB8** read framebuffer, `RGBA`, `ALPHA` and `LUMINANCE_ALPHA` ->
`INVALID_OPERATION` ("color buffer components can be dropped ... but new components cannot be added"), while
`RGB` and `LUMINANCE` succeed; from an **RGBA8** framebuffer a *sized* `RGB565` or `RGBA4` ->
`INVALID_OPERATION` ("if the component sizes of internalformat do not exactly match ... the source buffer's
effective internal format"), `RGBA8` succeeds; `border = 1`, a non-square cube-map face, an out-of-range
`glCopyTexSubImage2D` rectangle or a negative offset -> `INVALID_VALUE`; `READ_BUFFER = NONE` ->
`INVALID_OPERATION` for both commands.

## Not tested, on purpose (documented here)

- **Values of pixels outside the framebuffer** (reads and copies), and the **padding bytes** between rows
  after a read: undefined / unspecified.
- **Texture copying feedback loops** (§3.8.5.1, §4.4.3.2): copying into the image that is attached to the read
  framebuffer gives undefined texels. The demo copies into textures that are not attached while copying.
- **Reading a multisampled framebuffer** (`INVALID_OPERATION` while `SAMPLE_BUFFERS` is 1) and sRGB read
  buffers (`COLOR_ENCODING`): other demos.
- **Table 3.16's check marks** are not machine-readable in the text of the specification used here; the
  demo follows the prose of §3.8.5 (components can be dropped, never added; sized formats must match).

## Run

```
./build/debug/bin/es20-copy-and-read-pixels --headless --validate --capture copy-and-read-pixels.png
```

## Pitfalls

- **`RGB` + `UNSIGNED_BYTE` is not guaranteed.** Besides `RGBA`/`UNSIGNED_BYTE` (fixed-point buffers) the only
  other accepted pair is what the implementation names; query it for the read framebuffer you are going to
  read (it may differ between the default framebuffer and framebuffer objects, here `BGRA` versus `RGBA`).
- **Rows are padded to `PACK_ALIGNMENT` (initially 4).** An RGB/UNSIGNED_BYTE image whose width is not a
  multiple of 4 has padded rows; size the buffer with the row pitch, not with `width * pixel_size`. The setting
  is separate from `UNPACK_ALIGNMENT`.
- **Row 0 is the bottom row.** Most image file formats are top-down.
- **Outside the framebuffer is undefined, not an error.** A partly off-screen rectangle returns garbage for the
  missing part with no `GL_INVALID_*`.
- **A copy needs a complete read framebuffer and a read buffer** (`INVALID_FRAMEBUFFER_OPERATION`,
  `READ_BUFFER = NONE` -> `INVALID_OPERATION`). It reads `READ_FRAMEBUFFER`, not the draw framebuffer.
- **A copy can only drop components.** RGB sources cannot become RGBA, LA or ALPHA textures; sized
  destination formats must match the source's component sizes exactly, unsized ones are chosen by the GL.
- **`glCopyTexImage2D` creates a level; the default `TEXTURE_MIN_FILTER` needs the whole mipmap chain.** The
  copied level 0 alone is an *incomplete* texture and samples as `(0, 0, 0, 1)` until `NEAREST`/`LINEAR`
  min filtering is set (or the other levels are specified).
- **`glCopyTexSubImage2D` needs an existing level** of at least the requested size.
- **`glReadPixels` synchronizes** with the GPU. Reading the default framebuffer must happen before the buffer
  swap (its content is undefined afterwards); asynchronous reads need pixel buffer objects (ES 3.0, later demo).
- **ES 3.0 cannot read depth or stencil** with `glReadPixels` (those formats are excluded from the
  implementation-chosen pair): render depth to a texture instead.
