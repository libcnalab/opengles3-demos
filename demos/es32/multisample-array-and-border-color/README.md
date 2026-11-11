# es32-multisample-array-and-border-color

**API introduced in:** OpenGL ES 3.2 · **Minimum context:** ES 3.2 · **Shaders:** GLSL ES 3.20 ·
**Run:** `es32-multisample-array-and-border-color [--headless --validate] [--window]`

Spec numbering follows ES 3.2 (§8.2–8.3 Sampler Objects, §8.8 Multisample Textures, §8.10 Texture Parameters, §8.11 Texture Queries, §8.14.2 Coordinate Wrapping and Texel Selection,
§8.20.1 Depth Texture Comparison Mode, §9.2.8 Attaching Texture Images, §9.8 Layered Framebuffers, §14.2.2 Shader Inputs, §20.3 Internal Format Queries) and GLSL ES 3.20 (§4.1.7, §7.1, §8.9).

## Purpose

ES 3.2 moved two texture features into the core that earlier versions only had as extensions. This demo covers both and checks each against a CPU model.

**2D multisample array textures** (`TEXTURE_2D_MULTISAMPLE_ARRAY`, formerly `OES_texture_storage_multisample_2d_array`) combine the 2D array texture from ES 3.0 with the multisample texture from ES 3.1.
Each layer is a multisample image, and every texel has `TEXTURE_SAMPLES` samples. The texture is allocated once with `glTexStorage3DMultisample`, where `depth` is the number of layers, and it is immutable.
It is written by rendering into one layer (`glFramebufferTextureLayer`) or into all layers through a layered attachment (`glFramebufferTexture`, with a geometry shader choosing `gl_Layer`).
It is read sample by sample with `texelFetch(sampler2DMSArray, ivec3(x, y, layer), sample)`. No filtering or resolve happens. Typical uses are MSAA rendering of several views or cascades into one texture,
followed by a custom resolve, and keeping the samples of a G-buffer.

**The border color** (`CLAMP_TO_BORDER` and `TEXTURE_BORDER_COLOR`, formerly `OES/EXT_texture_border_clamp`) is a fourth wrap mode. A lookup outside `[0, 1]` returns a constant color instead of the edge texel or a repeat.
Shadow maps use it so that everything outside the light's frustum counts as lit (border depth 1). Texture atlases and decals use it so that nothing bleeds in from outside. The color can be set as float, signed-normalized integer,
pure signed integer or pure unsigned integer, on the texture or on a sampler object. The specification says precisely how it is converted and clamped for each texture format, and most of the checks here test those rules.

## What the image shows

The layout follows the surface size (`--width` / `--height`, 256 × 256 by default); both halves fill the window at any size. Surfaces smaller than 128 × 128 report
`SKIP_UNSUPPORTED`, because below that a tile or a quad cell has too few pixels to show every texel.

- **Top half: the RGBA8 multisample array, unresolved.** One tile per (layer, sample), filling the top half. The columns are samples (at most 8 shown; radeonsi and llvmpipe allocate 4), and the rows are layers 0..3 with layer 0 at the top.
  Each tile shows the 8 × 8 texels of that sample of that layer, scaled up with `texelFetch`. A gap of `max(1, min(tile width, tile height) / 24)` pixels on its right and bottom keeps the clear color. Every sample of every layer holds a different
  colour ramp because the fill shader writes `pattern(x, y, layer, gl_SampleID)`. The sudden colour changes inside a tile are channels wrapping past 255.
- **Bottom half: four square border quads**, in a row of four or a 2 × 2 block (whichever gives the larger squares, so a tall window gets 2 × 2). Each quad is 7/8 of its cell's shorter side,
  rounded down to a multiple of 8 so that every texel boundary falls on a pixel edge. The quads are drawn with `NEAREST` filtering and texture coordinates from −0.5 to 1.5, so the 4 × 4 texture fills the middle quarter and the frame around it is the border:
  1. RGBA8 with the texture's border `(1.5, −0.25, 0.2, 0.6)`, which clamps to `(1, 0, 0.2, 0.6)`: a **red** frame (255, 0, 51).
  2. The same texture with a sampler object bound whose border is `(0, 1, 0.4, 1)`: a **green** frame. The sampler replaces the texture's border.
  3. RGBA8I with the integer border `(300, 40, −200, 5)`, which clamps to `(127, 40, −128, 5)` and is drawn as `(v + 128) / 255`: an **orange** frame (255, 168, 0) around the bluish-grey integer texture.
  4. DEPTH_COMPONENT16 with `LEQUAL` comparison against `Dref = 0.5`: the texture's depths grow upward, so its lower half fails (black) and its upper half passes (white). The border depth is 0.875, so the frame passes (white).

## Central calls

`glTexStorage3DMultisample(TEXTURE_2D_MULTISAMPLE_ARRAY, samples, format, w, h, layers, fixedsamplelocations)`, `glFramebufferTexture` (layered), `glFramebufferTextureLayer`,
`glGetTexLevelParameteriv(TEXTURE_SAMPLES / TEXTURE_FIXED_SAMPLE_LOCATIONS / TEXTURE_DEPTH)`, `glGetInternalformativ(TEXTURE_2D_MULTISAMPLE_ARRAY, ..., SAMPLES)`;
`glTexParameter{fv, iv, Iiv, Iuiv}(TEXTURE_BORDER_COLOR)`, `glGetTexParameter{fv, Iiv, Iuiv}`, `glSamplerParameter{fv, Iiv, Iuiv}`, `glGetSamplerParameter{fv, Iiv, Iuiv}`, wrap mode `CLAMP_TO_BORDER`.
In GLSL: `sampler2DMSArray`, `isampler2DMSArray`, `usampler2DMSArray`, `texelFetch(gsampler2DMSArray, ivec3, int)`, `textureSize(gsampler2DMSArray)`, `gl_Layer` (geometry shader), `gl_SampleID`.

## Key ideas

### Multisample arrays

- **The sample count is a minimum.** "The resulting value for `TEXTURE_SAMPLES` is guaranteed to be greater than or equal to `samples` and no more than the next larger sample count supported by the implementation" (§8.8).
  The supported counts come from `glGetInternalformativ(TEXTURE_2D_MULTISAMPLE_ARRAY, format, SAMPLES)` in descending order (§20.3.1). llvmpipe supports only 4, so a request for 1, 2 or 3 samples gets 4.
  A request above the maximum for that format is `INVALID_OPERATION`, not `INVALID_VALUE`. The limit is per format and target, so check the integer formats separately.
- **`depth` counts layers**, and `glGetTexLevelParameteriv(TEXTURE_DEPTH)` returns it. `textureSize(gsampler2DMSArray)` returns `ivec3(width, height, layers)` and takes no LOD argument.
- **Immutable.** "Multisample textures do not have multiple image levels, and are immutable" (§8.8). A second `glTexStorage3DMultisample` on the same texture is `INVALID_OPERATION`, and so is a call with texture 0 bound.
  `glFramebufferTextureLayer` accepts only level 0 (`INVALID_VALUE` otherwise).
- **No sampler state.** Filters, wrap modes, LOD clamps, compare mode and the border color are sampler state (Table 21.12) and are `INVALID_ENUM` on a multisample target. `TEXTURE_BASE_LEVEL` may be set only to 0
  (`INVALID_OPERATION` otherwise, §8.10).
- **Writing.** A layered attachment (`glFramebufferTexture` on the array, `FRAMEBUFFER_ATTACHMENT_LAYERED` = TRUE) sends each primitive to the layer that the geometry shader wrote to `gl_Layer`, and
  `glClear` / `glClearBuffer*` clear all layers (§9.8). A `glFramebufferTextureLayer` attachment is an ordinary single-layer multisample target.
- **Giving every sample a known value.** Sample positions and the coverage of a primitive edge are implementation-defined, so the demo never relies on them. It draws a triangle that covers the whole viewport,
  so every sample is covered. The fragment shader reads `gl_SampleID`, and "using `gl_SampleID` in a fragment shader causes the entire shader to be executed per-sample" (§14.2.2), so each sample receives its own value.

### Border color

- **Which texel is a border texel.** `CLAMP_TO_BORDER` wraps a texel coordinate with `clamp(coord, −1, size)` (Table 8.21), and any `i < 0` or `i ≥ size` (likewise `j`) is a border texel.
  With `NEAREST`, every `s` below 0 or at or above 1 returns the border. With `LINEAR`, the 2 × 2 footprint can contain border texels: at `s = 0` with `t` at a texel centre, the result is half border and half edge texel.
- **Clamping depends on the format, not on the call.** "For signed and unsigned normalized fixed-point formats, border values are clamped to [−1, 1] and [0, 1], respectively. For floating-point and integer formats, border values are clamped to
  the representable range of the format" (§8.14.2). So `(1.5, −0.25, ...)` returns `(1, 0, ...)` on RGBA8 and returns unchanged on RGBA32F. `300` returns 127 on RGBA8I, and `70000` returns 65535 on RGBA16UI.
  The **stored** value is not clamped: `glGetTexParameterfv` and `glGetTexParameterI{i,ui}v` return exactly what was set (§8.10).
- **Missing components.** The border is "interpreted as an RGBA color to match the texture's internal format in a manner consistent with table 8.8". An RG8 texture has no B or A component, so the lookup returns
  `(R, G, 0, 1)` (Table 14.1) and ignores the border's B and A.
- **Depth.** "If the texture contains depth components, the first component of `TEXTURE_BORDER_COLOR` is interpreted as a depth value." With comparison enabled it is compared with `Dref` like a texel
  (Table 8.22). Without comparison the lookup returns it as the depth, clamped to [0, 1] for a fixed-point depth format (§8.20.1).
- **The four setters.** `fv` stores floats unchanged. `iv` converts with equation 2.2 (`c / (2^31 − 1)`, so `INT_MAX` is 1.0). `Iiv` and `Iuiv` store pure integers. An integer texture must get its border through `Iiv`/`Iuiv`
  ("the internal data type of the border values must be consistent with the type returned by the texture ... or the result is undefined"), and a float or normalized texture through `fv`/`iv`.
  Querying with a type that does not match the setter is also undefined, so each border is read back with its matching query.
- **Only vector forms.** `glTexParameteri/f(TEXTURE_BORDER_COLOR, ...)` and `glSamplerParameteri/f(...)` are `INVALID_ENUM` ("called for a non-scalar parameter").
- **Sampler objects win.** A sampler bound to the unit replaces *all* of the texture's sampler state, including the wrap mode and the border (§8.2). `glBindSampler(unit, 0)` brings the texture's own state back.

## Shader concepts

- `layers.geom` takes the full-viewport triangle and emits it four times, with `gl_Layer = 0..3` and a flat `v_layer` for the fragment shader. One draw fills every layer of the layered framebuffer.
- `pattern.frag` writes `pattern(x, y, layer, gl_SampleID)` as `vec4 / 255` (RGBA8), `ivec4 − 128` (RGBA8I) or `uvec4` (RGBA8UI). The CPU uses the same 32-bit unsigned arithmetic.
- `fetch.comp` reads every sample with `texelFetch(gsampler2DMSArray, ivec3(x, y, layer), s)` into a shader storage buffer, together with `textureSize`. The sampler type is injected as a macro (`SAMPLER`).
- `border.comp` is a lookup harness. Probe coordinates arrive in one storage buffer, and `textureLod(..., 0.0)` results go out as raw bits in another (`floatBitsToUint`, or `uvec4(ivec4)`, which keeps the bit pattern).
  It is compiled for `sampler2D`, `isampler2D`, `usampler2D` and `sampler2DShadow`. Compute shaders have no derivatives, so the LOD is explicit. With `MIN_FILTER = MAG_FILTER` the result does not depend on it.
- `show_ms.frag` and `show_border.frag` draw the picture. Opaque types other than `sampler2D` / `samplerCube` have no default precision in GLSL ES 3.20, so every sampler is declared `highp`.

## Validation (`--validate`)

All references are computed on the CPU. There are 135 checks on radeonsi, llvmpipe and zink at every surface size tested (128 × 128 to 2000 × 2000, including 640 × 480 and 333 × 777). With a 1-sample implementation the three per-sample negative controls are skipped and a note is printed.

- **Picture (4).** Every pixel of the top half equals the CPU's integer replica of `show_ms.frag` and `pattern()`, with tolerance 0 (texel `k/255` converts back to `k`). The shader works only on integer pixel
  positions, so this comparison is exact at any size. A second check confirms that every texel of every shown (layer, sample) is actually on screen.
  Every pixel of the bottom half is either the clear color (outside the quads, exact) or the texel or border cell that its pixel centre selects, within ±1 because a border not stored at 8-bit precision may round either way.
  The texel coordinate at a pixel centre is `u = −2 + 8 (x + 0.5) / size`. Where it lies within 0.02 texel of a boundary, both neighbouring cells are accepted, because the interpolated coordinate is not exact.
  Because quad sizes are multiples of 8, this only happens for quads larger than 200 pixels. A fourth check confirms that each quad shows both texels and border.
- **Sample counts (9).** For RGBA8, RGBA8I and RGBA8UI on `TEXTURE_2D_MULTISAMPLE_ARRAY`: at least one count is reported, the counts are positive and strictly descending, and the largest is at least the §20.3.1 guarantee.
  The demo reads that guarantee as the minimum of all limits that may apply (`MAX_COLOR_TEXTURE_SAMPLES`, `MAX_SAMPLES`, plus `MAX_INTEGER_SAMPLES` for the integer formats).
- **Allocation and state (43).** For each array: width, height, depth = layers, internal format, `TEXTURE_SAMPLES` in `[request, next supported count]`, `TEXTURE_FIXED_SAMPLE_LOCATIONS` (TRUE, and FALSE for a fourth texture),
  `TEXTURE_IMMUTABLE_FORMAT`, and `TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY`. Every request from 1 to the maximum is allocated and checked. Each framebuffer is complete, its `SAMPLES` equals `TEXTURE_SAMPLES`,
  and `FRAMEBUFFER_ATTACHMENT_LAYERED` is TRUE for the layered attachment and FALSE for the per-layer one.
- **Contents (12).** For each format, every sample of every texel of every layer (1024 values) equals `pattern(x, y, layer, sample)` exactly, and `textureSize` is (8, 8, 4).
  Negative controls: the same comparison against layer + 1, and against sample + 1, fails for more than half of the values (measured: all of them). This shows that layers and samples really hold different data and that a
  layer or sample mix-up would be caught.
- **Layered clear (3).** `glClearBufferuiv` on a layered RGBA8UI array sets all 3 layers × all samples. A second clear through a layer-1 attachment changes only layer 1.
- **Multisample errors (19).** These are the `INVALID_ENUM` / `INVALID_VALUE` / `INVALID_OPERATION` cases listed in `demo.json`. After the failed calls, `TEXTURE_SAMPLES` is still 0 and `TEXTURE_IMMUTABLE_FORMAT` is FALSE.
  After the rejected re-specification, `TEXTURE_DEPTH` is unchanged.
- **Border queries (7).** The initial border of a texture and of a sampler is (0, 0, 0, 0). `fv`, `Iiv` and `Iuiv` borders read back unmodified, and unclamped, through the matching query. The wrap modes read back as `CLAMP_TO_BORDER`.
- **Border lookups and setter conversions (24).** Each lookup check covers 30 probes: the 16 texel centres and 14 points outside `[0, 1]` in s, t, both, and far outside (−3, 7.25). The shadow checks use 3 references, so 90 probes.
  - RGBA8 with the texture's `fv` border, clamped. A fresh RGBA8 texture with the initial border (0, 0, 0, 0). The same texture after `glTexParameteriv(INT_MAX, −INT_MAX, 0, INT_MAX)`, stored as (1, −1, 0, 1) and looked up as (1, 0, 0, 1).
    RG8 with border (0.6, 0.2, 0.8, 0.4), looked up as (0.6, 0.2, 0, 1). Tolerance: half an 8-bit step. All values used are multiples of 1/255, so any conversion to the format's precision is exact.
  - RGBA32F: the border (−2.5, 3, 1000, 0.125) comes back **bit-exactly**.
  - RGBA8I `Iiv` → clamped to [−128, 127]. RGBA16UI `Iuiv` → clamped to 65535. RGBA32UI `Iuiv` → unchanged. Integer sampler objects with `SamplerParameterIiv/Iuiv` override the texture's border. All of these compare exact integers.
  - DEPTH_COMPONENT16 with `LEQUAL`: border depth 0.875 against `Dref` 0.1, 0.5 (pass) and 0.95 (fail), and the texels inside. Without comparison, borders 0.25 / 1.5 / −0.5 give a first component of 0.25 / 1.0 / 0.0
    (tolerance 1e-4, about 6 steps of 16 bits).
  - The RGBA8 texture with the `fv` sampler bound, then again after `glBindSampler(0, 0)`.
  - **Negative control:** a `CLAMP_TO_EDGE` sampler on the same `CLAMP_TO_BORDER` texture returns edge texels, so the sampler's wrap mode wins. At every outside probe the edge texel differs from both border colors by more than
    the tolerance, so a driver that ignored `CLAMP_TO_BORDER` would fail the border checks.
  - `LINEAR` at s or t = 0 or 1 with t or s at a texel centre: `(border + edge texel) / 2`. The spec does not give the precision of filter weights, so this uses a sanity tolerance of 2/255 and prints the measured error
    (0 on radeonsi, 6e-8 on llvmpipe).
- **Border errors (13).** `glTexParameteri/f` and `glSamplerParameteri/f` with `TEXTURE_BORDER_COLOR` give `INVALID_ENUM`, and the stored borders are unchanged. `glSamplerParameterIiv` on a deleted sampler gives `INVALID_OPERATION`.
  On `TEXTURE_2D_MULTISAMPLE_ARRAY`: `TexParameterfv` / `TexParameterIiv(BORDER_COLOR)`, `WRAP_S` and `MIN_FILTER` give `INVALID_ENUM`, `BASE_LEVEL` 1 gives `INVALID_OPERATION`, and `BASE_LEVEL` 0 is accepted.
- **Clean state (1).** No GL error is left in the queue at the end of `validate()`.

## Known driver deviation

- **Mesa (radeonsi, llvmpipe, zink), `glTexParameteriv(TEXTURE_BORDER_COLOR)`.** §8.10 says integer borders given to `TexParameteriv` "are converted to floating-point using equation 2.2", which is
  `f = max(c / (2^31 − 1), −1)`, so `c = 0` gives exactly 0. Mesa stores 2.3283064e-10 = 2^-32. That is `(2c + 1) / (2^32 − 1)`, the conversion used by OpenGL ES 2.0 and desktop GL before 4.2.
  The check stays strict and is reported as XFAIL on renderers containing "Mesa". The other components (INT_MAX → 1, −INT_MAX → −1) are checked separately and pass.
  The lookup is not affected visibly, because 2^-32 rounds to 0 in RGBA8.

## Spec ambiguities and how they were resolved

- **`FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER` of a multisample array.** §9.2.3 returns the layer only when the attached texture is "a three-dimensional texture or a two-dimensional array texture". "Otherwise params will contain zero."
  A 2D multisample array (or a cube map array) is not in that list. The literal reading is 0, which Mesa returns. The intent is presumably the attached layer. The demo prints the value and does not check it.
- **Layer ≥ number of layers for a multisample array.** The attachment-completeness rule in §9.4.1 names only "three-dimensional, two-dimensional array, or cube map array" textures, so it is unclear whether
  `glFramebufferTextureLayer(layer = layers)` on a multisample array must make the framebuffer incomplete. Not tested.
- **`TEXTURE_IMMUTABLE_FORMAT` after `TexStorage3DMultisample`.** §8.18 does not list the multisample commands, but §8.8 says multisample textures "are immutable" and defines the re-specification error through
  `TEXTURE_IMMUTABLE_FORMAT` being TRUE. The demo therefore expects TRUE. All drivers agree.
- **The §20.3.1 guarantee** ("at least the lowest of the following") lists conditions that overlap for color-renderable integer formats. The demo uses the minimum of all limits that may apply, which is the weakest reading.
- **Depth lookups without comparison** return `r = Dt` (§8.20.1). The specification does not say what the other three components are for a depth texture, so only the first is checked.
- **The exact value of a border in an 8-bit format** (is it quantised to the format?) is not specified. All normalized borders used are multiples of 1/255, so both answers agree.
- **Unsized internal formats** (`INVALID_ENUM` in §8.8) cannot be passed: meta-gl's `SizedInternalFormat` type does not contain them. That is a deliberate compile-time restriction, not a gap.
- The ES 3.2 text prints `TexParameterI{i ui}v( uint texture, ...)` in §8.10. That is a typo for `enum target`, and gl.xml and the queries in §8.11.2 use `target`.

## Pitfalls

- Integer textures need their border through `glTexParameterIiv` / `glTexParameterIuiv` (or the sampler equivalents). A border set with `fv` on an integer texture is undefined, and so is reading an `Iiv` border with `glGetTexParameterfv`.
- `glTexParameteri(TEXTURE_BORDER_COLOR, ...)` is an error. The border is always a 4-vector.
- A bound sampler object overrides the texture's wrap mode **and** border. Setting `CLAMP_TO_BORDER` on the texture does nothing while a sampler with `REPEAT` is bound.
- On a normalized texture a border of 1.5 is clamped when it is used. Read back, it is still 1.5. On a float texture it is returned as 1.5.
- Formats with fewer than four components ignore the border's extra components: an RG texture returns `(R, G, 0, 1)`.
- `glTexStorage3DMultisample`'s `depth` is the number of layers. Multisample arrays have no mipmaps, no sampler state and no `texture()` lookups: only `texelFetch` with an explicit sample number, and only for samples 0 to `TEXTURE_SAMPLES − 1`.
  Fetching a sample outside that range is undefined.
- Ask `glGetInternalformativ` for the supported counts on `TEXTURE_2D_MULTISAMPLE_ARRAY` and the actual format. `MAX_SAMPLES` is not the limit for integer formats, and a request is rounded up, not down.
