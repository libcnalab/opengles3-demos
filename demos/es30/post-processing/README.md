# es30-post-processing

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es30-post-processing [--headless --validate | --window]`

## Purpose

Most of what makes a 3D frame look finished happens after the geometry is drawn. This demo renders a lit,
animated 3D scene into an offscreen framebuffer object and then treats the result as an image: full-screen passes
read it back as a texture and write a new image into the next framebuffer, and the last pass puts the outcome in
the window. The passes are the classic set: **gamma correction** (in the shader, and with an `SRGB8_ALPHA8` render
target that the hardware encodes and decodes), a **separable Gaussian blur** that ping-pongs between two targets,
**Sobel edge detection**, **bloom** (bright-pass, blur, add) and a **depth-of-field** look that reads the scene's
depth texture. Every pass is validated by recomputing it on the CPU from the bytes the GPU actually produced.

## Read this first: the passes in order

`render()` calls `draw_frame()` in `main.cpp`, which runs these passes in this order (the names in the Reads and Writes
columns are members of the demo class, created in `create_targets()`; `scene_a_`, `scene_b_`, `tmp_`, `blur_t_`,
`edge_t_`, `bright_t_` and `bloom_t_` are framebuffer objects with one color texture, RGBA8 unless noted):

| # | Pass (C++ / shader) | Reads | Writes | What it does |
|---|---|---|---|---|
| 1 | scene, `draw_scene(scene_a_.fbo)` / `sky.frag`, `scene.vert/frag` | vertex buffers | `scene_a_` + the `DEPTH_COMPONENT24` **texture** `depth_tex_` | the lit 3D scene in linear light |
| 2 | sRGB scene, `draw_scene(scene_b_.fbo)` / the same programs | vertex buffers | `scene_b_` (`SRGB8_ALPHA8`) + a `DEPTH_COMPONENT24` **renderbuffer** | the same draws; the hardware encodes on write |
| 3 | blur, `blur_chain()` = two `pass_blur()` / `blur.frag` | `scene_a_`, then `tmp_` | `tmp_` (horizontal), then `blur_t_` (vertical) | separable Gaussian, ping-pong through `tmp_` |
| 4 | Sobel, `pass_sobel()` / `sobel.frag` | `scene_a_` | `edge_t_` | edge magnitude, thresholded to white lines |
| 5 | bright-pass, `pass_bright()` / `bright.frag` | `scene_a_` | `bright_t_` | keeps only what is brighter than the threshold |
| 6 | bloom blur, `blur_chain()` / `blur.frag` | `bright_t_`, then `tmp_` | `tmp_`, then `bloom_t_` | blurs the bright parts |
| 7 | present, `pass_present()` / `present.frag` | `scene_a_`, `scene_b_`, `blur_t_`, `edge_t_`, `bloom_t_`, `depth_tex_` | default framebuffer | the selected look: gamma, bloom add, depth of field mix, or the overview stripes |
| - | depth dump, `pass_depth_dump()` / `depth_dump.frag` | `depth_tex_` | `tmp_` | validation only: writes depth as color (ES 3.0 `glReadPixels` cannot read depth) |

In a single-look mode only the passes that look needs run; the overview runs all of them. **Start with the
bright-pass** (`pass_bright()` and `bright.frag`): one input texture, one `texelFetch`, one line of arithmetic and one
full-screen draw, so the whole pattern - bind the target FBO, bind the input texture, draw one triangle - is visible
without any other detail. The blur and present passes are the same pattern with more uniforms.

## The scene

A checkerboard floor (80 x 80 units, unit squares, analytically filtered in the shader so it does not alias in the
distance) fades into exponential-squared fog that is the color of the horizon. On it stand a gold torus that tumbles,
a polished blue sphere, a teal cube that spins, a magenta cube close to the camera, and seven small red and yellow
spheres on a spiral that give depth cues. Two point lights, one warm and one cool, orbit the scene; each is drawn as a
small emissive bulb. A dim blue directional "moon" and a weak ambient term complete the lighting, which is per-fragment
Blinn-Phong in **linear** light. The sky is a gradient computed from the elevation of each pixel's view ray, so the haze
sits exactly on the horizon whatever the camera does. All meshes (sphere, torus, cube, floor) are generated in
`main.cpp`; the camera orbits slowly (0.12 rad/s, time is `frame / 60` s) with a 45 degree vertical field of view.

## Looks and controls

The window shows an **overview** by default: seven vertical stripes, left to right, each one a different look of the
same frame, separated by one-pixel lines. Keys select one look full screen:

| Key | Look | What it shows |
|---|---|---|
| `0` | overview | all seven as stripes |
| `1` | raw | the linear scene written to the window unchanged: dark and contrasty, the wrong look |
| `2` | gamma | the RGBA8 scene encoded to sRGB by the fragment shader |
| `3` | gamma, SRGB8_ALPHA8 | the scene rendered into an `SRGB8_ALPHA8` texture (hardware encode on write, decode on every lookup), then encoded for the window |
| `4` | blur | two-pass separable Gaussian blur, radius 10, sigma 4 (key `b` cycles the radius 2, 5, 10, 16) |
| `5` | sobel | gradient magnitude of the brightness, thresholded to white lines on black |
| `6` | bloom | bright parts of the scene blurred and added back |
| `7` | depth of field | sharp near the focal plane (the point the camera looks at), blurred in front of it and behind it |

`n` / `p` step through the looks, `a` / `d` orbit the camera, `w` / `s` dolly, `q` / `e` change the height, `c` toggles
the automatic orbit, `r` resets the camera, `space` pauses the animation, `esc` quits. Only the passes the selected look
needs are drawn; the overview draws everything.

## How the full-screen passes work

Every pass except the scene is `glDrawArrays(TRIANGLES, 0, 3)` of one oversized triangle whose corners come from
`gl_VertexID` (`fullscreen.vert`); there is no vertex buffer. Each fragment shader reads its input with
`texelFetch(sampler, ivec2(gl_FragCoord.xy), 0)`: pixel (x, y) of the pass reads texel (x, y), with no filtering and
no texture coordinates. Because the viewport equals the target size, that is a one-to-one image operation.

The central calls are `glGenFramebuffers`, `glFramebufferTexture2D`, `glFramebufferRenderbuffer`,
`glCheckFramebufferStatus`, `glTexImage2D` (with a `NULL` pointer: storage only), `glRenderbufferStorage`,
`glBindFramebuffer`, `glViewport`, `glActiveTexture` / `glBindTexture` (six units in the present pass),
`glDrawElements` for the scene and `glDrawArrays` for the passes, and `glReadPixels` for validation.

## Gamma: linear light, sRGB, and the hardware path

Lighting is only meaningful in **linear** light: doubling a light's intensity should double the pixel value. A monitor,
however, expects non-linear values (sRGB), so the last step has to apply the sRGB encoding (ES 3.0 equation 4.1)

```
c_srgb = 12.92 c                      c <= 0.0031308
c_srgb = 1.055 c^(1/2.4) - 0.055      c >  0.0031308
```

Look 1 writes the linear numbers directly (everything is too dark), look 2 encodes them in the fragment shader. Blur,
bloom and depth of field are computed on the linear numbers and encoded once at the end, because averaging is only
physically right in linear light (the average of black and white is 0.5 linear, which is 188 in sRGB, not 128).

There is a precision problem with storing linear values in 8 bits, though: the dark end of the range gets too few
levels, and a smooth dark gradient turns into visible bands once it is encoded for the window. The second render target
therefore has the internal format `SRGB8_ALPHA8`: when it is a color attachment, its `FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING`
is `SRGB`, so the hardware encodes the shader's linear output on write (§4.1.8), and every texture lookup (including
`texelFetch`, whose remaining texture-access steps still apply, §2.12.9.2) decodes the stored bytes back to linear
(§3.8.16, equation 3.26). The code downstream does not change, but the stored bytes keep far more precision in the
dark. In the demo the effect is visible in the dark sky gradient: in one 640 x 480 capture a column of the sky has 48
distinct colors with steps of up to 18 (summed over the channels) between neighboring rows through the RGBA8 path,
and 101 distinct colors with steps of at most 3 through the `SRGB8_ALPHA8` path. Stripes 2 and 3 of the overview show
the two paths; the difference is small but systematic.

Two cautions: `glReadPixels` of an `SRGB8_ALPHA8` target returns the **stored** (encoded) bytes, not linear values,
and the alpha channel is always linear.

## Blur, Sobel, bloom, depth of field

- **Gaussian blur.** A 2D Gaussian is the product of two 1D Gaussians, so a horizontal pass followed by a vertical pass
  gives the same image as one (2r+1) x (2r+1) kernel with 2(2r+1) instead of (2r+1)^2 fetches per pixel. The weights
  `exp(-i^2 / (2 sigma^2))` are computed in the shader and normalized by their sum. `texelFetch` has no wrap mode and an
  out-of-range coordinate is undefined (§2.12.9.2), so the coordinates are clamped by hand, which equals
  `CLAMP_TO_EDGE`.
- **Sobel.** The kernels `Gx = [-1 0 1; -2 0 2; -1 0 1]` and `Gy = [-1 -2 -1; 0 0 0; 1 2 1]` are applied to the
  square root of the Rec. 709 luma (a cheap stand-in for the sRGB curve, so dark edges are not suppressed). The magnitude
  goes through `smoothstep(0.15, 0.6, .)`: shading gradients and the one-step bands of 8-bit gradients vanish, outlines
  and checker edges are white.
- **Bloom.** The bright-pass keeps `max(c - 0.7, 0) / 0.3` per channel, the blur spreads it, and the present pass adds
  1.4 times the result to the scene. It is "lite" because the scene is stored with 8 bits and clamped to 1.0; a real
  implementation keeps HDR values in a float or half-float target.
- **Depth of field.** The scene depth is a `DEPTH_COMPONENT24` *texture* attached to the scene framebuffer. With
  `TEXTURE_COMPARE_MODE = NONE` a lookup returns the stored depth (§3.8.15.1). The shader linearizes it
  (`z = 2nf / (f + n - (2d - 1)(f - n))`), computes a circle of confusion
  `coc = clamp((|z - focus| - 0.75) / 3, 0, 1)` and mixes the sharp scene with its blurred copy by `coc`. One blurred
  copy is an approximation: colors of the in-focus object bleed into the defocused area next to it (a soft halo around
  the torus), which real depth-of-field implementations avoid by scattering or gathering per pixel.

## Shader concepts

- All post shaders declare `precision highp float; precision highp int; precision highp sampler2D;`. In a fragment shader
  `sampler2D` defaults to `lowp` (GLSL ES 3.00 §4.5.4), and the result of a lookup has the precision of its sampler, so
  without `highp sampler2D` a 24-bit depth texture is read with the precision of `lowp`.
- `gl_VertexID` generates the full-screen triangle; `gl_FragCoord` addresses the texel; `textureSize` gives the bounds for
  the manual clamp; a `for` loop with a uniform bound is the blur kernel.
- The scene fragment shader uses `fwidth` (GLSL ES 3.00 §8.9) for an analytically filtered checkerboard. Derivatives
  are undefined in non-uniform control flow, so the branch on `u_kind` is a uniform branch.
- Full-screen draws also carry the integer arithmetic of the overview stripes (`(x * 7) / width`) and the 24-bit split of
  the depth dump (`>>`, `&` on `uint`), both of which depend on `highp int`.

## Expected result

The 256 x 256 capture shows the overview: from the left, a dark raw stripe, two gamma stripes that look alike, a blurred
stripe, a black stripe with white outlines (Sobel), a stripe with glowing lamp and bright floor patches (bloom) and a
stripe where the foreground and the background are soft while the torus is sharp (depth of field). Behind the stripes is
one continuous picture: a checkerboard floor with a blue sphere, a gold torus, a teal cube and a magenta cube under a
dark purple sky, lit by a warm and a cool light. At larger sizes (`--width 640 --height 480`) and with keys `1` to `7` the
individual looks fill the window.

## Validation (`--validate`, 256 x 256, after 45 frames)

The runtime checks (`glGetError` clean after init, after the frame, after shutdown) apply as usual. The demo records 38
checks:

- **Scene, analytically.** The CPU picks probe pixels from the camera and the scene: 16 sky pixels, 24 floor pixels (more
  than 0.2 units away from every square edge and from the footprint of the pixel, not covered by any object's bounding
  sphere) and the CPU-projected centers of the two lamp bulbs (model-view-projection, perspective divide, viewport
  transform). Sky = the smoothstep gradient of the ray elevation; floor = the same Blinn-Phong with the directional and
  both point lights, the filtered checker in the middle of a square and the fog; lamps = the emissive color. All are
  checked in the RGBA8 scene (the exact value) and in the `SRGB8_ALPHA8` scene (the stored bytes against the sRGB encoding
  of the exact value, equation 4.1). A second comparison checks the whole sRGB image against the encoding of the RGBA8
  image with a per-pixel tolerance derived from the quantization of the 8-bit linear image (below).
- **Depth.** The depth texture is read through `depth_dump.frag` (ES 3.0 has no depth `glReadPixels`) and compared at the
  floor probes with the analytic window-space depth `0.5 ((f+n)/(f-n) - 2fn / ((f-n) z)) + 0.5`; sky pixels must keep the
  cleared depth 1.0.
- **Passes.** Each offscreen image is read back and the next pass is recomputed on the CPU **from those bytes**, so the
  8-bit quantization between passes cannot accumulate: horizontal blur of the scene, vertical blur of the horizontal
  result, the whole blur chain with an explicit 8-bit rounding in between, the bright-pass, both bloom blur passes, and
  Sobel. Every pixel and every channel (alpha must be exactly 255) is compared.
- **Presentation.** Each of the eight modes is drawn into the default framebuffer, read back and compared with the CPU
  model of the same effect (gamma, the decode of the `SRGB8_ALPHA8` lookup followed by the encode, blur, Sobel, bloom,
  depth of field, the stripes with their dividers). The image that was in the window before `validate()` started must equal
  the CPU model of the selected look.
- **State queries.** `FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING` is `GL_SRGB` for the `SRGB8_ALPHA8` texture and `GL_LINEAR`
  for the RGBA8 texture.

**Tolerances.** The expected values are computed in double precision from the same inputs. A float is converted to 8 bits
as "one of the two integers closest to f * 255" (§2.1.6, equation 2.3), so rounding alone can be 0.5 steps off; measured
hardware reaches about 0.6 (`es30-srgb` documents 0.604 on radeonsi) and the shader's float arithmetic (sums, `exp`,
`pow`, `sqrt`, "about 1 part in 10^5" per operation, §2.1.1) adds a few hundredths. One 8-bit step is therefore the bound
for a single pass; the measured maximum is 0.62 on radeonsi and 0.58 on llvmpipe and zink, and 0.00 for the presentations that only
decode or copy. The two-pass chain with an 8-bit intermediate allows 2 steps (it measures 1.16). The sRGB-vs-linear scene comparison
uses `0.6 * enc'(x) + 0.6` steps per pixel, where `enc'` is the slope of the encode curve at the lower end of the interval
of exact values that the 8-bit linear byte stands for: about 8.4 steps in the darkest areas, where one linear step is 12.9
encoded steps, and about 1.2 in bright ones. The depth bound 2e-5 is a sanity bound (ES 3.0 states no depth accuracy): with near 0.5 and far 80 it is 0.004
world units at 10 units from the camera, and the measured error is 3e-7. No tolerance is loosened for any driver.

**Negative controls (the checks can fail).** Six wrong CPU models are compared with the same GPU images and must disagree
by more than the tolerance in many pixels: a blur with radius 7 instead of 10 (47 000 samples of 262 144 differ, up to 13
steps), a box kernel instead of the Gaussian (78 000, up to 74), the unblurred scene against the blurred result (65 000), a
Sobel with the horizontal kernel only (31 000), a plain gamma-2.2 power curve instead of the sRGB curve (about 110 000 differ,
up to 7.5 steps, in the dark sky), and an `SRGB8_ALPHA8` lookup that skips the decode (181 000, up to 73 steps).

## Pitfalls

- **Texture completeness.** The default `TEXTURE_MIN_FILTER` is `NEAREST_MIPMAP_LINEAR`; a texture that has only level 0
  is then incomplete and samples as (0, 0, 0, 1) (§3.8.13). Render targets are created with `NEAREST` filters. `texelFetch`
  itself ignores filtering and wrap modes.
- **Feedback loops.** A pass must never sample a texture that is attached to the framebuffer it draws into (§4.4.3). That is
  why the blur uses a ping target and a result target instead of blurring in place.
- **Out-of-range `texelFetch`.** Coordinates outside the image are undefined results (§2.12.9.2), not clamped; clamp in the
  shader.
- **Dithering.** `DITHER` is enabled by default (§4.1.9) and may change a stored byte by one step; the demo disables it so
  that comparisons with the CPU are not blurred by noise.
- **Gamma twice.** The encoding of the default framebuffer is implementation-defined (§6.1.13). The demo queries it; if it
  reports `GL_SRGB`, `present.frag` decodes before writing so the hardware's encode does not double the gamma. The tested
  configurations (headless on radeonsi, llvmpipe and zink; windowed on llvmpipe under Xvfb) report `GL_LINEAR`, so that
  branch was not exercised.
- **Depth is not readable.** ES 3.0 `glReadPixels` has no depth format; to compare depth with the CPU the demo samples
  the depth texture and writes it as color (`depth_dump.frag`).
- **Windowed resizing.** All targets are created with the size of the window and re-created when it changes.
- **Light bleed in the depth-of-field look.** See above: it is a property of the one-blurred-copy approximation.

## Spec references

ES 3.0 §2.1.1 Floating-Point Computation · §2.1.6 Fixed-Point Data Conversions (equations 2.1, 2.3) · §2.12.9.2 Texel Fetches ·
§2.13.1 Controlling the Viewport · §3.6.1 Basic Polygon Rasterization · §3.8.13 Texture Completeness · §3.8.15.1 Depth Texture
Comparison Mode · §3.8.16 sRGB Texture Color Conversion (equation 3.26) · §4.1.5 Depth Buffer Test · §4.1.8 sRGB Conversion
(equation 4.1) · §4.1.9 Dithering · §4.2.3 Clearing the Buffers · §4.3.2 Reading Pixels · §4.4.1 Binding and Managing
Framebuffer Objects · §4.4.2.1 Renderbuffer Objects · §4.4.2.3/§4.4.2.4 Attaching Images · §4.4.3 Feedback Loops · §4.4.4
Framebuffer Completeness · §6.1.13 Framebuffer Object Queries.
GLSL ES 3.00 §4.5.4 Default Precision Qualifiers (and section 12, Issues, on sampler precision) · §6.3 Iteration · §7.1 / §7.2 Special
Variables · §8.2 Exponential Functions · §8.3 Common Functions · §8.8 Texture Lookup Functions · §8.9 Fragment Processing Functions.
