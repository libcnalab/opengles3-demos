# es20-blending

**API introduced in:** OpenGL ES 2.0 (`MIN`/`MAX` equations: ES 3.0) · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

Blending is the last programmable-looking step of the pipeline that is not programmable: after the
fragment shader has produced a *source color* `(Rs, Gs, Bs, As)`, the GL combines it with the
*destination color* `(Rd, Gd, Bd, Ad)` already in the framebuffer using a **blend equation**, two
pairs of **blend factors** and a **constant color**. This demo shows every factor of Table 4.2, every
equation of Table 4.1, separate RGB/alpha state, `glBlendColor`, and the everyday use of all this:
alpha compositing of translucent shapes. Every pixel is compared with an independent CPU
implementation of the spec's arithmetic.

## What the image shows

256 x 256 default framebuffer (RGBA8).

**Top: a 16 x 4 contact sheet.** In every cell the thin *frame* is the destination color (drawn with
blending disabled, so it is exact) and the *inner rectangle* is a source color drawn over it with one
blend configuration. Reading order is row by row; the check output lists the cells by number.

| Row | Cells | Configuration | What it proves |
|---|---|---|---|
| 0 | 0-14 | `glBlendFunc(X, ZERO)` for each of the 15 factors `ZERO`, `ONE`, `SRC_COLOR`, `ONE_MINUS_SRC_COLOR`, `DST_COLOR`, `ONE_MINUS_DST_COLOR`, `SRC_ALPHA`, `ONE_MINUS_SRC_ALPHA`, `DST_ALPHA`, `ONE_MINUS_DST_ALPHA`, `CONSTANT_COLOR`, `ONE_MINUS_CONSTANT_COLOR`, `CONSTANT_ALPHA`, `ONE_MINUS_CONSTANT_ALPHA`, `SRC_ALPHA_SATURATE` | result = `src * factor`: each *source* factor, RGB and alpha |
| 0 | 15 | `SRC_ALPHA_SATURATE` over a destination with `Ad = 0.3` | the other branch of `f = min(As, 1 - Ad)` |
| 1 | 16-29 | `glBlendFunc(ZERO, X)` for the 14 destination factors | result = `dst * factor`: each *destination* factor |
| 1 | 30, 31 | blending disabled; classic `SRC_ALPHA, ONE_MINUS_SRC_ALPHA` | control and the standard "over" |
| 2 | 32-35 | `FUNC_ADD`, `FUNC_SUBTRACT`, `FUNC_REVERSE_SUBTRACT` with half weights; `FUNC_ADD` with `ONE, ONE` | equations, clamping of negative / >1 results |
| 2 | 36-39 | `MIN` and `MAX`, each with factors `ZERO,ZERO` and `ONE,ONE` | the factors are ignored |
| 2 | 40-43 | `glBlendEquationSeparate`: ADD/MAX, SUBTRACT/REVERSE_SUBTRACT, MIN/ADD, MAX/MIN | different equation for RGB and alpha |
| 2 | 44, 45 | `glBlendColor(1.5, -0.5, 0.5, 2.0)` | out-of-range constant colors are clamped to [0, 1] |
| 2 | 46, 47 | additive `SRC_ALPHA, ONE`; premultiplied `ONE, ONE_MINUS_SRC_ALPHA` | two common recipes |
| 3 | 48-51 | `glBlendFuncSeparate` with four independent factors | RGB and alpha factors really are separate |
| 3 | 52-63 | `CONSTANT_ALPHA, ONE_MINUS_CONSTANT_ALPHA` with `glBlendColor` alpha 0/11 ... 11/11 | a cross-fade driven only by the blend color |

The capture is a PNG of the RGBA framebuffer *including its alpha channel*. Most cells end with an
alpha below 1 (that is part of what is being tested), and image viewers composite such pixels over
their own background: cell 0 (`ZERO, ZERO`, result `(0,0,0,0)`) therefore looks white in many viewers
although its RGB is black.

**Bottom: painter's-order compositing.** A light/dark checkerboard with three translucent discs
(red, green, blue, `alpha = 0.6`) in a Venn arrangement. The left cluster is drawn red, green, blue;
the right cluster blue, green, red. The "over" operator is not commutative, so the overlaps are
different colors, and the figure makes that visible. The backdrop shows through everywhere, as it
must for `Cs*As + Cd*(1-As)`.

## Central calls

| Call | Role |
|---|---|
| `glEnable(GL_BLEND)` / `glDisable` / `glIsEnabled` | blending is off by default; one switch for all draw buffers |
| `glBlendFunc(src, dst)` | sets RGB **and** alpha factors to the same pair |
| `glBlendFuncSeparate(srcRGB, dstRGB, srcA, dstA)` | four independent factors |
| `glBlendEquation(mode)` | `FUNC_ADD`, `FUNC_SUBTRACT`, `FUNC_REVERSE_SUBTRACT`, `MIN`, `MAX` for RGB and alpha |
| `glBlendEquationSeparate(rgb, alpha)` | one equation per part |
| `glBlendColor(r, g, b, a)` | the constant color used by `CONSTANT_*` factors |
| `glGetIntegerv(BLEND_SRC_RGB ... BLEND_EQUATION_ALPHA)`, `glGetFloatv(BLEND_COLOR)` | state read-back |

## Spec

- ES 3.0 §4.1.7 Blending: §4.1.7.1 Blend Equation (**Table 4.1**), §4.1.7.2 Blend Functions
  (**Table 4.2**), §4.1.7.3 Blend Color, §4.1.7.4 Blending State (initial values).
- ES 3.0 §4.1.8 (the result is clamped to [0, 1] and converted to fixed point), §2.1.6 (normalized
  fixed-point conversions, equations 2.1 and 2.3), §4.1.9 Dithering.
- GLSL ES 3.00 §4.3.5 uniforms, §4.3.6/§4.3.8 fragment outputs and `layout(location)`, §6.4 `discard`,
  §7.1 `gl_VertexID`.

Rules that matter (all in §4.1.7): source, destination and factors are clamped to [0, 1] before the
equation is evaluated; blending only applies to fixed-point color buffers; the constant color is
clamped when the destination is normalized fixed point; if a buffer has no alpha, `Ad` is taken as 1;
initial state is `ONE`/`ZERO` with `FUNC_ADD`, blend color `(0,0,0,0)`, blending disabled.
**`MIN` and `MAX` take no factors** (Table 4.1 contains none), which the demo proves by giving
them `ZERO` and `ONE` factors and getting the same pixels.

## Shader concepts

`rect.vert` builds a rectangle from `gl_VertexID` (no vertex buffer, GLSL ES 3.00) and places it with
one `vec4` uniform in NDC; `color.frag` only outputs the *source color* from a uniform, through a
user-declared `layout(location = 0) out vec4`. For the discs it `discard`s fragments outside the
inscribed circle: a discarded fragment never reaches blending. Blending itself is fixed-function
state, not shader code.

## How it is validated

- **Reference.** `reference_blend()` in `main.cpp` implements Tables 4.1 and 4.2 on the 8-bit
  quantized values the framebuffer holds: clamp inputs and factors to [0, 1], evaluate the equation
  in floating point, clamp, round to nearest 8-bit (eq. 2.3). All colors in the demo are multiples of
  1/255, so source and destination are exactly representable and no input quantization error enters.
- **Cells.** All 64 contact-sheet cells are read back (a 7 x 12 pixel block each, 1 pixel inside the
  source rectangle so primitive edges are never probed) and every pixel must match the CPU result
  per channel within **+-2/255**, alpha included. The frames are checked separately to be exact.
- **Why +-2.** The result is the float evaluation rounded to 8 bits, so a GPU that blends in float
  differs by at most one step (rounding exactly at a .5 boundary). An implementation that blends in
  8-bit fixed point approximates each factor product with a rounded division by 255, which is one
  step per product, so a sum of two products can differ by two. The spec only demands "a precision no
  lower than that used to represent destination components" (§4.1.7.1). The demo prints the largest
  deviation it saw: 1/255 on both radeonsi and llvmpipe.
- **Composite.** The disc scene is compared at one pixel for each of the 8 membership patterns
  (backdrop, R, G, B, RG, RB, GB, RGB) in both clusters. The probe pixel for each pattern is chosen
  by search as the pixel farthest from every disc outline and every checker edge (at least 3 px), and
  the expectation applies the model once per covering disc, in draw order, re-quantizing to 8 bits
  after every draw as the framebuffer does. Tolerance is 2, 3, 4 for one, two, three layers (the
  error of an operation is propagated scaled by `1 - As = 0.4`, so the bound is `2*(1 + 0.4 + 0.16)`);
  a pixel without discs is a plain write (+-1). An extra check requires the RGB-order and BGR-order
  triple overlaps to differ by the amount the model predicts.
- **State.** Initial values, then `glBlendFunc`, `glBlendFuncSeparate`, `glBlendEquation`,
  `glBlendEquationSeparate`, `glBlendColor` and `glIsEnabled` are read back with `glGet*`.
  `DITHER` is reported enabled initially (§4.1.9) and is disabled for rendering.
- **Framebuffer.** The demo needs 8 bits per color channel (otherwise `SKIP_UNSUPPORTED`). Destination
  alpha is stored on the EGL pbuffer; if a framebuffer without alpha were used, `Ad` would be modelled
  as 1 and alpha comparisons skipped, so the `DST_ALPHA` cells would still be correct.

## Run

```
./build/debug/bin/es20-blending --headless --validate --capture blending.png
```

## Pitfalls

- **Dithering** (initially enabled) can change a blended value by one step. Disable `GL_DITHER` when
  you compare pixels.
- **`glBlendFunc(SRC_ALPHA, ONE_MINUS_SRC_ALPHA)` also blends alpha**: `A = As*As + Ad*(1-As)`, so an
  opaque target gets alpha < 1 (cell 31). Use `glBlendFuncSeparate(..., ONE, ONE_MINUS_SRC_ALPHA)` to
  keep alpha correct (cell 51, and the composite scene); it matters whenever the framebuffer's alpha
  is used by a compositor, and the captured PNG keeps this alpha.
- **Premultiplied alpha** wants `ONE, ONE_MINUS_SRC_ALPHA` and a source whose RGB is already
  multiplied by alpha (cell 47); mixing the two conventions darkens or brightens edges.
- **Order matters.** Blending is not commutative; translucent geometry must be drawn back to front.
- **Every draw re-quantizes** to 8 bits. Stacked translucent layers therefore accumulate rounding
  error (bounded in the composite probes), and a reference must round after each layer.
- **`MIN`/`MAX` ignore the factors**; `FUNC_SUBTRACT`/`REVERSE_SUBTRACT` clamp negative results to 0.
- **`SRC_ALPHA_SATURATE`** is a *source* factor: its RGB factor is `min(As, 1 - Ad)` and its alpha
  factor is 1. The demo only uses it as a source factor, the conventional usage.
- **`glBlendColor` is clamped** for normalized targets, so `(1.5, -0.5, ...)` behaves as `(1, 0, ...)`.
- **Integer color buffers are never blended** (§4.1.7: the fragment goes to the next operation
  unchanged); blending only has meaning for normalized fixed-point (and floating-point) targets.
