# es32-advanced-blend-and-sample-shading

ES 3.2 brought two fragment-stage features into the core API that were extensions in ES 3.1:

- **Advanced blend equations** (`KHR_blend_equation_advanced`): the fifteen compositing modes of image editors
  (MULTIPLY, SCREEN, OVERLAY, DARKEN, LIGHTEN, COLORDODGE, COLORBURN, HARDLIGHT, SOFTLIGHT, DIFFERENCE, EXCLUSION,
  HSL_HUE, HSL_SATURATION, HSL_COLOR, HSL_LUMINOSITY), selected with `glBlendEquation`.
- **Sample shading and per-sample interpolation** (`OES_sample_shading`, `OES_sample_variables`,
  `OES_shader_multisample_interpolation`): run the fragment shader per sample, read which sample is being shaded and
  where it is, and evaluate inputs at chosen positions inside the pixel.

This demo draws a grid of all fifteen blend modes and a sample-shading panel, and checks both against CPU models of
the specification.

- Minimum version: OpenGL ES 3.2, GLSL ES 3.20 (`#version 320 es`). On an ES 3.1 context (softpipe) the run reports
  SKIP_UNSUPPORTED.
- The multisample checks need an RGBA8 multisample texture with 2 to 8 samples. If the driver offers none
  (`MAX_SAMPLES` = 1), only those checks are skipped, the reason is printed, and the bottom panels are drawn grey.

## What you see

The picture fills the window at any size.

- **Top 5/8: the blend grid.** There are 4 x 4 cells, read row by row from the top left: MULTIPLY, SCREEN, OVERLAY,
  DARKEN / LIGHTEN, COLORDODGE, COLORBURN, HARDLIGHT / SOFTLIGHT, DIFFERENCE, EXCLUSION, HSL_HUE / HSL_SATURATION,
  HSL_COLOR, HSL_LUMINOSITY, control. Every cell has the same backdrop: a hue ramp from left to right, brightness rising
  upwards, and alpha falling from 0.95 to 0.45 to the right. Disc A (left, blue to magenta, alpha 0.9 to 0.45) is
  blended over the backdrop, followed by a `glBlendBarrier`. Then disc B (right, orange to yellow, alpha 0.5 to 0.9)
  is blended over both. The lens where the discs overlap therefore shows each equation applied twice. The texture
  holds premultiplied colors with alpha below 1, so it is shown over a grey checkerboard (`C + (1 - A) * checker`), and
  the checkerboard shows through the transparent right side of each cell. The 16th cell is the control: BLEND is
  disabled and the discs simply replace the backdrop.
- **Bottom 3/8: three panels of a 32 x 32 four-sample texture** with concentric rings that become thinner than a pixel
  towards the corners.
  - Left: the samples averaged, shaded once per pixel. The rings break into blocky moiré.
  - Middle: the same with `SAMPLE_SHADING` and `glMinSampleShading(1.0)`. Each sample is shaded at its own position,
    so the average smooths the ring edges to grey-olive.
  - Right: the middle texture "exploded". Each texel is split into 2 x 2 sub-cells, and sub-cell `s` shows sample `s`.
    The fine ring structure is visible inside single texels.

## Central API

| Call | Role |
|---|---|
| `glBlendEquation(MULTIPLY ... HSL_LUMINOSITY)` | selects an advanced equation (Tables 15.3, 15.4) |
| `layout(blend_support_<eq>) out;`, `layout(blend_support_all_equations) out;` | declares which advanced equations the fragment shader may be used with |
| `glBlendBarrier()` | ends a blending pass, so that overlapping draws are blended in order |
| `glEnable(SAMPLE_SHADING)`, `glMinSampleShading(f)` | at least `ceil(f * samples)` distinct input sets per fragment |
| `gl_SampleID`, `gl_SamplePosition`, `gl_SampleMaskIn[]`, `gl_NumSamples` | per-sample built-ins (GLSL ES 3.20 §7.1.5) |
| `sample in`, `centroid in` | per-sample and centroid interpolation (GLSL ES 3.20 §4.5) |
| `interpolateAtSample/Offset/Centroid` | evaluate an input at a chosen position (GLSL ES 3.20 §8.14.2) |
| `glGetMultisamplefv(SAMPLE_POSITION, i)` | position of sample `i` inside the pixel (ES 3.2 §13.4) |
| `glTexStorage2DMultisample`, `texelFetch(sampler2DMS, p, s)` | render per-sample data and read it back sample by sample |

## Specification

- ES 3.2 §15.1.5.3 *Advanced Blend Equations*. With premultiplied source `(Cs, As)` and destination `(Cd, Ad)`, the
  base colors are `Cs' = Cs / As` and `Cd' = Cd / Ad`. Then
  `RGB = f(Cs', Cd') As Ad + Y Cs' As (1 - Ad) + Z Cd' Ad (1 - As)` and `A = X As Ad + Y As (1 - Ad) + Z Ad (1 - As)`,
  with `(X, Y, Z) = (1, 1, 1)` for every mode and `f` from Table 15.3 or from the HSL pseudocode (`ClipColor`,
  `SetLum`, `SetLumSat`) of Table 15.4. The section also defines the draw-time errors, the single-color-buffer rule
  and `BlendBarrier`.
- ES 3.2 §15.1.5.1. `BlendEquation` accepts Tables 15.1, 15.3 and 15.4. `BlendEquationSeparate` raises INVALID_ENUM
  for anything outside Table 15.1. Advanced equations must be computed with at least "the smaller of" the destination
  precision and 16-bit float precision.
- ES 3.2 §13.4, §13.4.1 *Sample Shading*, §14.1 *Fragment Shader Variables*, §11.1.3.3 *Multisample Texel Fetches*,
  Table 21.41 (limits) and Table 21.8 (initial state).
- GLSL ES 3.20 §4.4.2 (*Fragment Outputs*, the `blend_support` qualifiers), §4.3.4 and §4.5 (auxiliary qualifiers),
  §4.7.1 (mediump precision), §7.1.5, §8.9.2, §8.14.2.

## Shaders

- `quad.vert`: one vertex shader for every pass. It takes the corners from `u_corner[]` in pixels, so the output
  `v_pos` is the window-space position of the fragment. It writes the same value to a plain, a `centroid out` and a
  `sample out` output. GLSL ES 3.20 §4.3.4 lets the auxiliary qualifiers differ between stages, and the fragment
  shader's qualifier wins.
- `backdrop.frag`, `source.frag`: the gradients, written premultiplied. `main.cpp` builds 17 programs from
  `source.frag` by replacing a marker line with `layout(blend_support_<eq>) out;` (15 programs),
  `layout(blend_support_all_equations) out;`, or nothing.
- `probe.frag`: nine `MODE`s, each writing one per-sample quantity into an RGBA8 multisample texture. Positions are
  divided by 16 and stored with 16 bits (`hi = floor(255 t)`, `lo = fract(255 t)`), about 1e-4 pixel.
- `fetch.frag`: `texelFetch(sampler2DMS, texel, sample)` copies all samples into a single-sample strip for
  `glReadPixels`.
- `pattern.frag`: the rings. `composite.frag`: the window.

## Validation

**Advanced blending (every texel of all 16 cells).**

- The destination is not predicted. After the grid has been read, the backdrop is drawn again into the same
  framebuffer and read back. Appendix A.1 and A.3 rule 1 make it identical to the first draw.
- The reference applies section 15.1.5.3 in double precision to the source gradient and the read-back destination:
  disc A, then disc B on top. Disc coverage comes from the same `length() <= r` test as the shader. Texels within
  1e-3 of the rim accept both coverages ("accept either neighbour").
- Tolerance:
  - A texel passes if every channel is within ±1 of the exact reference: 8-bit rounding of the stored result.
  - Otherwise it must lie in the precision range that §15.1.5.1 allows. For an RGBA8 destination, "the smaller of"
    the two precisions is 8 bits, so the base colors and the source alpha are known only to ±½ LSB. The reference is
    evaluated at the 256 corners of that input box and at the exact point, and the result may lie anywhere in the
    range of those values, ±1.
  - For the overlap lens, the stored intermediate result after disc A is itself only known to that range plus
    rounding, so disc B is evaluated over that interval.
  - Steep regions (COLORDODGE near `Cs = 1`, SOFTLIGHT's square root, HSL near grey) need the range for at most a few
    dozen texels per cell. The largest deviation from the exact value is 2.6 LSB (COLORDODGE on radeonsi).
- The blend functions are set to `(ZERO, ONE_MINUS_SRC_COLOR)` throughout. The advanced equations have no factors, so
  any driver that applied them would fail every cell.
- **Negative control:** each cell's texels are also judged against the reference of the next equation (MULTIPLY
  against SCREEN, ..., HSL_LUMINOSITY against MULTIPLY). At least 20% must fail; on all three drivers 100% do. The
  check therefore distinguishes the equations.
- Every window pixel of the grid region is compared with the CPU composite (over the checkerboard) of the grid texel
  it maps to, at any window size.

**Errors and state.**

- All 15 equations read back through `BLEND_EQUATION_RGB` and `BLEND_EQUATION_ALPHA`.
- `glBlendEquationSeparate(MULTIPLY, FUNC_ADD)` and `glBlendEquationSeparate(FUNC_ADD, HSL_COLOR)` raise INVALID_ENUM
  and change nothing.
- With BLEND enabled, every equation raises INVALID_OPERATION at the draw for two programs: one without a
  `blend_support` qualifier, and one whose qualifier names a different equation. The program with its own qualifier,
  and the `all_equations` program, draw without error.
- The rejected draw leaves the color buffer untouched (§2.3.1).
- Controls that raise no error: FUNC_ADD without a qualifier; MULTIPLY with BLEND disabled.
- Draw buffers `{COLOR0, COLOR1}` and `{NONE, COLOR1}` raise INVALID_OPERATION ("the draw buffer for any other color
  output is not NONE"). `{COLOR0, NONE}` draws without error.

**Sample shading (16 x 16 texture, 4 samples, every sample read back through `texelFetch`).**

- `glGetMultisamplefv`: every position lies in [0, 1]; INVALID_VALUE for index = SAMPLES; the positions are distinct.
- MODE 0: `gl_SampleID == s` in sample `s`, `gl_NumSamples == SAMPLES`, and `gl_SampleMaskIn[0] == 1 << s`. Static
  use of `gl_SampleID` makes the shader run per sample, and an invocation "covers" only its own sample.
- MODE 1: `gl_SamplePosition` equals the `glGetMultisamplefv` position, within 1/512 (mediump relative precision
  2^-10).
- The sample positions:
  - MODE 2: a `sample in` input.
  - MODE 3 with `SAMPLE_SHADING` and `MinSampleShading(1.0)`: an ordinary input.
  - MODE 4: `interpolateAtSample(v, i)`, for every `i`, in a pixel-rate shader.

  Each must equal pixel + the position of the sample. The specification gives no precision for interpolation, so the
  tolerance is a sanity bound: 1/32 pixel, or a quarter of the closest distance between samples. All drivers measure
  1.2e-4 pixel.
- Negative control: compared with the position of sample `s + 1`, no sample matches.
- `MinSampleShading(0.5)`: every pixel has at least `ceil(0.5 * 4) = 2` distinct input values. Without sample shading
  the count is printed only, because the specification does not define it.
- MODE 5: `interpolateAtOffset` with offsets (0, 0), (MIN, MIN), (MAX, MAX), (MIN, MAX), (0.25, -0.375) and
  (-0.125, 0.4375). The offsets lie on the 2^-bits grid inside the queried limits, so no rounding applies. The input
  must be the pixel center + offset. The limits themselves are checked against Table 21.41 (≤ -0.5, ≥ 0.5 - 1 ULP,
  ≥ 4 bits).
- MODES 6, 7 and 8 on a triangle. In every covered sample, `interpolateAtCentroid(v)` and a `centroid in` input lie
  inside both the pixel and the triangle, and they are equal. The triangle gives 29 partially covered pixels, 21 of
  them with the center outside the triangle. `gl_SampleMaskIn[0]` of the pixel-rate shader contains the sample it
  wrote and only covered samples.
- The displayed rings: with `MinSampleShading(1.0)`, each sample shows the pattern at its own position. Samples where
  the pattern changes within the tolerance accept both colors. 712 of 1024 pixels hold both colors.
- `SAMPLE_SHADING` is initially FALSE and `MIN_SAMPLE_SHADING_VALUE` is initially 0. 0.25 reads back as 0.25. 1.5 and
  -0.5 are clamped to 1.0 and 0.0. `IsEnabled` follows Enable and Disable.

## Known deviation (radeonsi)

On radeonsi (Mesa 25.0.7), a draw with `SAMPLE_SHADING` and `MinSampleShading(1.0)` is shaded once per pixel when it
follows a draw into a single-sample framebuffer and no flush comes in between. Every sample then receives the inputs
interpolated at sample 0. This violates §13.4.1: "When the sample shading fraction is 1.0, a separate set of fragment
shader input values are evaluated for each sample, and each set of values is evaluated at the sample location".

With `glFlush()` between the two draws, the result is correct. The demo checks both orders. The order without a flush
is a strict check that is recorded as XFAIL on radeonsi; the order with a flush is a strict check that passes on every
driver. The displayed rings are drawn after a `glFlush()`, so that the panel is right on radeonsi too. llvmpipe and
zink pass both orders.

## How to run

```
./build/debug/bin/es32-advanced-blend-and-sample-shading                       # window
./build/debug/bin/es32-advanced-blend-and-sample-shading --headless --validate --capture out.png
ctest --preset debug -R es32-advanced-blend-and-sample-shading
```

## Pitfalls

- Advanced equations expect **premultiplied** colors. A straight-alpha source with `Rs > As` is outside the model, and
  with `As = 0` it is undefined.
- A fragment shader without the matching `layout(blend_support_*) out;` makes every draw fail with
  INVALID_OPERATION, but only while BLEND is enabled.
- Only one color buffer may be written. Any second draw buffer that is not NONE is an error, even if the shader does
  not write it.
- Without `glBlendBarrier` (or `KHR_blend_equation_advanced_coherent`, which is not core), blending into a sample that
  the same pass already touched is undefined. That includes the clear or backdrop draw before the first blended draw.
- `glBlendEquationSeparate` cannot take advanced equations.
- `gl_SampleMaskIn` of a per-sample invocation holds only that invocation's sample, not the coverage of the whole
  pixel.
- Static use of `gl_SampleID` or `gl_SamplePosition`, or any `sample in` input, already makes the whole shader run
  per sample. `glMinSampleShading` is needed only for shaders that use none of them.
- `interpolateAtOffset` offsets outside `[MIN, MAX]_FRAGMENT_INTERPOLATION_OFFSET` give undefined positions, and
  offsets are rounded to `FRAGMENT_INTERPOLATION_OFFSET_BITS` fraction bits.
