# es30-srgb

**API introduced in:** OpenGL ES 3.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 ·
**Run:** `es30-srgb [--headless --validate]`

## Purpose

Display devices are not linear: a pixel value of 128 is not "half as bright" as 255, and 8-bit images are stored in the **sRGB** color
space, which spends its 256 steps where the eye needs them (dark colors). Shading, blending and filtering, on the other hand, are only
correct in *linear* light. ES 3.0 therefore performs the conversion for you in two places, and gives the application a query to see which
one applies:

- **Texture side (section 3.8.16, equation 3.26).** A texture whose internal format is **`SRGB8`** or **`SRGB8_ALPHA8`** stores sRGB-encoded
  red, green and blue. A lookup returns them *linearized*: `c_l = c_s / 12.92` for `c_s <= 0.04045`, else `((c_s + 0.055) / 1.055)^2.4`.
  **Alpha is always linear** and is returned unchanged. The shader needs no change: the internal format alone decides.
- **Framebuffer side (sections 4.1.7, 4.1.8, equation 4.1).** When the `FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING` of an attachment is `SRGB` (a
  color-renderable `SRGB8_ALPHA8` texture or renderbuffer), the destination color is **linearized before blending**, the blend result is
  **encoded** (`12.92 c_l` below 0.0031308, else `1.055 c_l^(1/2.4) - 0.055`, clamped to [0, 1]) and the *encoded* 8-bit value is stored. The same
  encoding applies to clears (section 4.2.3) and to `glBlitFramebuffer` (section 4.3.3, which also decodes what it reads from an `SRGB` read buffer).
  `glCopyTexImage2D` refuses to copy between the two encodings (section 3.8.5).

This demo shows each of them, verifies the numbers, and reads back the bytes that are actually stored.

## What the image shows

256 x 256 pixels, eight bands of 32 pixels (band 0 on top). In every band, **pixel column *x* is the byte value *x*** (0 on the left, 255 on the right).

| Band | What | What you see |
|---|---|---|
| 0 | `RGBA8` ramp texture, red channel, as is | the straight reference ramp |
| 1 | the same bytes in an **`SRGB8_ALPHA8`** texture, as the shader sees them | the dark curve of equation 3.26: the sampler returns *linear* values, which are small for the first half of the byte range |
| 2 | band 1, re-encoded by a `pow()` in the fragment shader | back to the straight ramp: decode and encode are inverses |
| 3 | stored bytes of an `SRGB8_ALPHA8` target that a *linear* ramp was written to | the bright curve of equation 4.1: the GL encoded the shader's linear values on the way into the buffer (the bytes are read back with `glReadPixels`, put into an `RGBA8` texture and shown raw) |
| 4 | the same target after a 50% white overlay was blended in | blending in linear space: the dark part is lifted a lot (it starts near 0.8 gray) |
| 5 | an `RGBA8` target with the same ramp and overlay | blending in the stored values: a straight line from 0.5 to 1 |
| 6 | the alpha channel of the `SRGB8_ALPHA8` ramp | a straight ramp: **alpha is not converted** |
| 7 | an `SRGB8` (no alpha) ramp as the shader sees it | the same dark curve as band 1 |

The texels are `(x, 255 - x, 37x mod 256, x)`, so a channel mix-up could not go unnoticed; the bands show the red channel.
The ramp written to the render targets is `l = x/255`, color `(l, 1 - l, l*l)`, alpha `1 - l`.

## Central calls

`glTexImage2D(SRGB8_ALPHA8 / SRGB8, ...)`, `glRenderbufferStorage(SRGB8_ALPHA8)`, `glFramebufferTexture2D` / `glFramebufferRenderbuffer`,
`glGetFramebufferAttachmentParameteriv(..., FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING, ...)` (for the default framebuffer with attachment `BACK`: meta-gl's
`DefaultFramebufferQueryAttachment`), `glBlendFunc`, `glClear` / `glClearBufferfv`, `glBlitFramebuffer`, `glCopyTexImage2D`, `glReadPixels`. There is no
`GL_FRAMEBUFFER_SRGB` enable in ES: the conversion is always on for an `SRGB` attachment.

## Key ideas

- **The shader sees linear values.** Lighting equations assume linear light; texture colors painted on screen are almost always sRGB. Put them in an
  `SRGB8_ALPHA8` texture and the sampler hands the shader what it needs. Alpha stays linear (it is coverage, not a brightness).
- **Write linear, store encoded.** A shader writing to an `SRGB` target writes linear values; the GL encodes. Reading the buffer back with
  `glReadPixels` returns the *stored* (encoded) bytes: the read does not convert (section 4.3.2 has no sRGB step).
- **Blending is linear too.** With `SRC_ALPHA, ONE_MINUS_SRC_ALPHA` and a source of (1, 1, 1, 0.5), `RGBA8` computes `0.5 + 0.5 d` on the stored bytes; an
  `SRGB8_ALPHA8` target computes `encode(0.5 + 0.5 decode(d))`. For a destination byte of 64 that is 160 versus 192: visibly different.
- **A render target's encoding is a property of the attachment**, not of the framebuffer or a global switch; `COLOR_ENCODING` reports it
  (`SRGB` for an `SRGB8_ALPHA8` texture or renderbuffer attachment, `LINEAR` for everything else including depth attachments). For the default framebuffer it is
  "determined by the implementation" (6.1.13): the headless EGL pbuffers of this suite report `LINEAR`.
  The demo reads the value and derives what a shader output looks like in the window from it, so it stays correct on a context whose window is sRGB-encoded.
- **Only some formats can be render targets.** `SRGB8_ALPHA8` is color-renderable and texture-filterable; `SRGB8` is texture-only (Table 3.13).
- **Filtering and sRGB.** The ideal is to linearize each texel and then filter; the specification explicitly also allows filtering first and decoding after (3.8.16).

## Validation (`--validate`)

57 checks, none of which only checks that a call returned:

- **Bands (8).** Every pixel column of every band (256 probes each) against a CPU model. Bands 0, 3, 4, 5 and 6 are byte copies and must match within the rounding
  of one step; bands 1, 2 and 7 contain the decode and use the conversion tolerance below. Bands 3 to 5 are compared with the bytes read back from the targets
  (those are checked against the equations separately, below).
- **Exact texture decode (15).** `RGBA8`, `SRGB8_ALPHA8` and `SRGB8` ramps are sampled with `texture()` and with `texelFetch()` into an `RGBA32UI` target
  (`floatBitsToUint`) and the 32-bit floats are compared with the CPU implementation of equations 2.1 and 3.26. `SRGB8` returns alpha 1. Two negative controls:
  the `SRGB8_ALPHA8` red channel differs from the undecoded byte, and the `RGBA8` texture (same bytes) differs from the decoded value, for more than 200 of 256 texels.
  One more check is the strict reading of the accuracy (see Tolerances).
- **Filtering (3).** A two-texel `SRGB8_ALPHA8` texture sampled exactly between the texels with `LINEAR`: the result must match "average of the decoded texels" or "decode of
  the average" (both allowed by 3.8.16) and must not be the plain average of the bytes; alpha is the plain average. The demo prints which order the driver uses (all three
  Mesa drivers decode first).
- **Encode on write (5).** The linear ramp written to the `SRGB8_ALPHA8` target is stored as `255 * equation 4.1` for all 768 color bytes, alpha is stored unencoded;
  negative controls: the `RGBA8` target stores the ramp itself, the two targets differ by more than 8 in most columns, and the stored bytes are not those of a gamma-2.2 power curve.
- **Blending (4).** The 50% overlay on the `SRGB8_ALPHA8` target equals `encode(0.5 + 0.5 * decode(destination byte))` in all 768 color bytes, and does *not* match blending in the stored values;
  the `RGBA8` target blends the stored values; destination alpha is not linearized.
- **Clears (3).** `glClear` and `glClearBufferfv(COLOR)` store the encoded clear color in an `SRGB8_ALPHA8` renderbuffer, `RGBA8` stores it as is.
- **Blits (3).** `SRGB8_ALPHA8` to `RGBA8` arrives decoded, `RGBA8` to `SRGB8_ALPHA8` arrives encoded, `SRGB8_ALPHA8` to `SRGB8_ALPHA8` arrives unchanged.
- **`COLOR_ENCODING` (9).** `SRGB` for an `SRGB8_ALPHA8` texture and renderbuffer attachment, `LINEAR` for `RGBA8` (texture, renderbuffer), `RGBA32UI` and a depth attachment;
  the default framebuffer reports `LINEAR` or `SRGB`, and a linear color drawn into the window is stored accordingly (the pbuffer reports `LINEAR`).
- **Rules (6).** `glCopyTexImage2D` raises `INVALID_OPERATION` for a `LINEAR` read buffer into `SRGB8_ALPHA8` and for an `SRGB` read buffer into `RGBA8` and accepts the matching pairs;
  `glRenderbufferStorage(SRGB8)` raises `INVALID_ENUM`; a framebuffer with an `SRGB8` texture attachment is `FRAMEBUFFER_INCOMPLETE_ATTACHMENT`.

### Tolerances

All 8-bit comparisons are in steps of 1/255 of `|stored byte - 255 * exact value|`.

- **Rounding (`0.52`).** For byte copies and results that are integers or half-integers before the conversion: equation 2.3 rounds to the nearest step (error 0.5, a tie may go either way);
  ES 3.0 section 2.1.1 lets the float arithmetic be off by "about 1 part in 10^5", at most 0.003 steps.
- **Conversion (`1.0`).** For computed floats and for results of an sRGB conversion. docs/VALIDATION.md's policy is "8-bit quantization +-1". The measurements agree: real hardware converts a computed
  float to 8 bits with up to ~0.6 steps of error (radeonsi 0.604), and **the specification gives no accuracy for equations 3.26 and 4.1** beyond the generic 2.1.1 *about 1 part in 10^5*, which none of the three drivers meets
  for the decode (they use tables or short polynomials: the unrounded decode result is off by up to 0.49 steps on radeonsi and 0.31 on llvmpipe/zink; relative error 3.8e-3 and 2.5e-2 at the dark end). The largest deviation measured in any 8-bit
  result is 0.85 steps. One step is far tighter than any plausible mistake: a gamma-2.2 curve instead of the sRGB curve is up to 6 steps off and a missing conversion up to 100.
- **Strict reading, XFAIL on Mesa.** One of the decode checks keeps the 1e-4 relative bound of 2.1.1 for the `SRGB8_ALPHA8` decode and is recorded as a known deviation on radeonsi, llvmpipe and zink.
  The names of the passing decode checks print the measured errors.

## Spec ambiguities and how they were resolved

- **Accuracy of the conversions.** See Tolerances: the specification states formulas, not accuracy; the demo uses the 8-bit policy tolerance, reports the measured numbers, and
  keeps one strict check (known deviation on Mesa). The integrator should add a row to `docs/KNOWN_LIMITATIONS.md`.
- **Filter before or after the decode.** 3.8.16: "Ideally, implementations should perform this color conversion on each sample prior to filtering but implementations are allowed to perform this conversion after
  filtering". Both orders are legal, so the check accepts the two models and rejects everything else; this is not a loosened check but the text of the specification.
- **Does `glClear` encode?** 4.2.3 describes the cleared value as "(possibly sRGB converted and/or dithered)" and lists "sRGB conversion" among the per-fragment operations that apply to `Clear`; `ClearBuffer*` "in the same fashion".
  The demo checks it (all three drivers do it).
- **`COLOR_ENCODING` of the default framebuffer** is implementation-defined, so the demo asserts that the answer is `LINEAR` or `SRGB` and that the window behaves like the answer says.
- **Dithering.** Section 4.1.9 allows dithering of the 8-bit result (on by default); the demo disables `DITHER` so the stored bytes are the rounded values.

## Pitfalls

- An `SRGB8_ALPHA8` texture that is *also* sampled by a shader that already applies `pow(c, 2.2)` decodes twice; likewise writing already-encoded colors to an `SRGB` target encodes twice.
- A texture's *stored* bytes are not visible through the sampler (it decodes). To look at them, read them back with `glReadPixels` (as bands 3 to 5 do) or sample an `RGBA8` copy.
- Mipmap generation is up to the implementation ("no particular filter algorithm is required, though a box filter is recommended", 3.8.10.5); the specification does not say that sRGB
  levels are averaged in linear space. By section 3.8.10.5 `glGenerateMipmap` is allowed on `SRGB8_ALPHA8` (renderable and filterable) but not on `SRGB8` (not renderable); this demo does not test it.
- `glBlitFramebuffer` between a `LINEAR` and an `SRGB` buffer converts; between two `SRGB` buffers decode followed by encode gives back the stored bytes.
- `glCopyTexImage2D` between the encodings is an error rather than a silent conversion.
- Alpha is never converted: premultiplying alpha in linear versus sRGB space is a modelling decision the GL does not make for you.
