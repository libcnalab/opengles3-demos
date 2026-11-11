# es31-image-load-store

**API introduced in:** OpenGL ES 3.1 · **Minimum context:** ES 3.1 · **Shaders:** GLSL ES 3.10 ·
**Run:** `es31-image-load-store [--headless --validate]`

Spec numbering follows ES 3.2 (the same text is section 8.22 with Table 8.27 in ES 3.1).

## Purpose

Until ES 3.1 a shader could only *sample* a texture (a filtered, read-only operation) or write it as a render target through the fixed-function
output stage. ES 3.1 adds **images**: one level (or one layer) of a texture is bound to an *image unit* with `glBindImageTexture`, and a shader reads
and writes individual texels with `imageLoad`, `imageStore` and `imageSize`: no filtering, no framebuffer, arbitrary addresses, in compute and (if the
implementation allows it) fragment shaders. Because these accesses are not tracked by the GL, they are ordered with the rest of the pipeline only by
explicit `glMemoryBarrier` calls.

This demo exercises that whole path, not only "does a store land": every image format the ES specification lists, both kinds of binding (layered and
single-layer), mip level selection, format reinterpretation, the memory qualifiers, the state that describes an image unit and the errors that guard it.

## What the image shows

A 4 x 4 contact sheet in a 256 x 256 window. Each cell is 64 x 64 pixels; row 0 is the top row.

| Row | Cells |
|---|---|
| 0 | the 64 x 64 source image (gradient, disc, square, checker patch) · a 3x3 Gaussian blur of it (compute shader, `imageLoad` x 9 + `imageStore`) · a Sobel edge magnitude of its luminance · the source color-inverted by a **fragment** shader that reads it with `imageLoad` |
| 1 | the four layers of an 8 x 8 x 4 2D array texture. Layers 0, 1 and 3 were written through one **layered** binding (`image2DArray`), layer 2 (blue) was overwritten afterwards through a **non-layered** `image2D` binding of layer 2 |
| 2 | the four slices of an 8 x 8 x 4 3D texture: the same experiment with `image3D`; slice 1 (blue) is the one rewritten through a single-slice binding |
| 3 | levels 0, 1 and 2 of a 16 x 16 texture: only level 1 (the middle cell, inverted colors) was bound to an image, the other two keep their flat color · a texture written by a **fragment** shader with `imageStore` (a gray ramp `3x + 5y + 1`) |

In rows 1 and 2 the texel value is `1000 * tag + index`; the display maps the index to gray and the tag to a tint, so "layer 2 comes from another pass" is visible.

## Central calls

`glTexStorage2D` / `glTexStorage3D` (an image can only be bound to a texture with immutable storage), **`glBindImageTexture(unit, texture, level, layered, layer, access, format)`**,
`glMemoryBarrier`, `glDispatchCompute`, `glGetIntegeri_v` / `glGetBooleani_v` with `IMAGE_BINDING_*`, `glGetTexParameteriv(IMAGE_FORMAT_COMPATIBILITY_TYPE)`.
In GLSL: `layout(rgba8, binding = 0) uniform readonly highp image2D`, `imageLoad`, `imageStore`, `imageSize`, `coherent` / `volatile` / `restrict` / `readonly` / `writeonly`,
`memoryBarrierImage()`, `barrier()`.

## Key ideas

- **An image unit binds one level.** `level` selects a single mip level; the image's `imageSize` is the size of that level. With `layered = TRUE` a 2D array, 3D, cube or cube
  array texture is bound as a whole (the third coordinate selects the layer: Table 8.25); with `layered = FALSE` exactly one layer, slice or face is bound and the shader
  declares a plain `image2D` (for a cube map the layer number selects the face: Table 8.24, layer 3 = `NEGATIVE_Y`). A texture without layers ignores both arguments.
- **The format is a promise, not a conversion of the texture.** The `format` argument and the `layout(...)` qualifier name how texels are interpreted. Table 8.26 lists 13 formats
  (`rgba32f rgba16f r32f rgba32ui rgba16ui rgba8ui r32ui rgba32i rgba16i rgba8i r32i rgba8 rgba8_snorm`). A texture of another internal format is allowed if the texel sizes
  match (Table 8.27, "compatible by size" for GL-allocated textures): an `R32F` texture bound as `r32ui` loads its IEEE bit patterns, an `RGBA8` texture bound as `r32ui` loads the four
  bytes of a texel as one word (in client byte order, the way the spec describes the reinterpretation through scratch memory), and `r32ui` stores into an `RGBA8` texture change its bytes.
- **Loads and stores work in the image's component type.** `imageLoad` returns `vec4`, `ivec4` or `uvec4` and fills missing components with `(0, 0, 0, 1)`; `imageStore` of a `vec4` into a
  normalized format rounds to one of the two closest integers (section 2.3.5.2).
- **Memory qualifiers are enforced by the compiler.** A `readonly` image cannot be stored to, a `writeonly` image cannot be loaded from, every image needs a format qualifier of the matching
  family, and only `r32f`, `r32i` and `r32ui` images may be neither (read-write); every other image must say `readonly` or `writeonly` (GLSL ES 3.10 4.4.7, 4.9). The demo provokes all of these.
  `coherent` (and `volatile`, which implies it) lets invocations of one work group exchange data through an image once `memoryBarrierImage()` and `barrier()` have been passed.
- **Barriers connect the stages.** Image stores are not synchronized with anything. The demo uses `SHADER_IMAGE_ACCESS` between dependent image passes, `TEXTURE_FETCH` before a sampler
  reads what an image wrote, `FRAMEBUFFER` before `glReadPixels` of such a texture and `BUFFER_UPDATE` before mapping a buffer that a shader filled (section 7.11.2).
- **Image units have state.** Six values per unit (`IMAGE_BINDING_NAME`, `LEVEL`, `LAYERED`, `LAYER`, `ACCESS`, `FORMAT`, Table 21.33) are queried with indexed `glGet*i_v`;
  `LAYERED` is a boolean and needs `glGetBooleani_v`. Deleting a bound texture detaches it.
- **Fragment shaders may not have images.** `MAX_FRAGMENT_IMAGE_UNIFORMS` is 0 in a minimal ES 3.1 implementation (4 in ES 3.2). The demo reads the limit, and when it is 0 skips the
  fragment-shader cells and checks and says so (every Mesa driver tested reports 16 or 32).

## Shader concepts

`transform.comp` is one template instantiated for the 13 formats (a macro block is inserted after `#version 310 es`): `imageLoad` from a `readonly` image, a permutation of the components
(or `x * 2 + 1` for the single-channel formats) and `imageStore` to a `writeonly` image. `dump.comp` reads a texture back with `texelFetch` (a *sampler*, an access path that has nothing in
common with `imageLoad`) and writes the raw 32-bit words into a shader storage buffer, which the CPU maps. `blur.comp` and `sobel.comp` are the image filters (`readonly restrict` / `writeonly
restrict`). `layered_store.comp` / `single_store.comp` / `face_color.comp` show layered and non-layered stores, `sizes.comp` queries `imageSize` of four differently bound images with `readonly writeonly`
(allowed: it neither reads nor writes), `coherent.comp` makes a work group reverse a row segment through a coherent (or volatile) `r32ui` image. `frag_store.frag` and `frag_load.frag` use images in fragment shaders.

## Validation (`--validate`)

143 checks (the same at es-max and es-min, on radeonsi, llvmpipe and zink), grouped:

- **Limits and initial state (9).** `MAX_IMAGE_UNITS`, `MAX_COMPUTE_IMAGE_UNIFORMS`, `MAX_COMBINED_IMAGE_UNIFORMS` are at least the specified minimum 4; read before anything is bound, every unit has
  name 0, level 0, layered FALSE, layer 0, access `READ_ONLY`, format `R32UI` (Table 21.33).
- **13 formats (28).** For each format: the destination texture, read through a sampler, equals the CPU model texel for texel (floats and integers exactly, 64 texels each, extreme values included);
  the `readonly` source is unchanged. `rgba8` and `rgba8_snorm` are compared within one stored level (a float is converted to "one of the two closest integers", section 2.3.5.2); a texture that was
  only uploaded is compared within 0.01 level (equation 2.1/2.2 evaluated in float). Two negative controls feed the same comparison a model with one perturbed word and require it to fail.
- **Filters (3).** All 4096 texels of the blur and of the Sobel output (read with `glReadPixels` through a framebuffer object, a path independent of the shaders) equal the CPU filters
  within 1 level (the measured maximum is 1 for the blur, 0 for Sobel); a control requires the blurred image to differ from the source by up to 97 levels, so the comparison can fail.
- **Layered and non-layered bindings (14).** The four layers of the array and the four slices of the 3D texture hold exactly the values of the layered store, except layer 2 / slice 1 which hold those of the
  single-layer store (and nothing else changed); the cube map faces `NEGATIVE_Y` (layer 3) and `NEGATIVE_Z` (layer 5) hold the gradient, the other four faces stay black (Table 8.24).
- **Levels and `imageSize` (10).** Level 1 of texture B is the inverted level 1 of A, levels 0, 2, 3, 4 are untouched; `imageSize` is (8, 8) for level 1 of a 16 x 16 texture, (4, 4) for level 2, (8, 8, 4) for the layered
  3D texture and 2D array, (8, 8) for one layer of that array.
- **Reinterpretation (3).** `R32F` as `r32ui` returns the float bit patterns, `RGBA8` as `r32ui` the packed bytes, `r32ui` stores appear as bytes in an `RGBA8` texture; all 64 texels each.
- **`coherent`, `volatile`, `restrict` (5).** The row-reversal through the shared image is exact for both qualifier sets; the scratch image holds `3 * in + 1`; a control shows that the un-reversed model differs.
- **Fragment shader (1).** `imageStore` wrote `3x + 5y + 1` at every one of 4096 pixels (the rest of the texture holds a sentinel no fragment writes), and the inverted-source cell shows what `imageLoad` read.
- **Image unit queries (30).** For four bindings (layered array, one layer, level 2, RGBA16F) all six `IMAGE_BINDING_*` values; unbinding with texture 0; detach on `glDeleteTextures`; `IMAGE_FORMAT_COMPATIBILITY_TYPE` is
  `IMAGE_FORMAT_COMPATIBILITY_BY_SIZE` for immutable 2D, immutable array and mutable textures.
- **Errors (16).** A mutable texture gives `INVALID_OPERATION`; `unit == MAX_IMAGE_UNITS`, a negative level or layer, a name that is not a texture, and seven formats outside Table 8.26 give `INVALID_VALUE`;
  the failed calls left unit 0 unchanged; indexed queries with an index of `MAX_IMAGE_UNITS` give `INVALID_VALUE` (section 20.1).
- **Compile-time errors (6).** One control shader compiles; store to `readonly`, load from `writeonly`, `rgba8` without `readonly`/`writeonly`, no format qualifier, `rgba8` on a `uimage2D` do not.
- **Display (18).** Probes of every cell of the capture against the CPU model (`texelFetch`, so the pixels are exact up to the 8-bit output), and a negative control (the blur cell does not match the unfiltered source).

## Spec ambiguities and how they were resolved

- **Section numbers.** The GLSL ES 3.10 image functions are section 8.12 (not 8.14); `barrier` is 8.14 and the memory barriers 8.15. The inventory entry `GLSL310.FN.IMAGE` quotes 8.14. ES 3.1 calls the
  image section 8.22 and the format table 8.27; ES 3.2 renumbered them 8.23 and 8.26. The README and `demo.json` use the ES 3.2 numbers.
- **`IMAGE_BINDING_LAYERED` of a texture without layers.** Table 21.33 says "texture object bound with multiple layers", section 8.23 says the whole level is bound regardless of `layered`.
  The demo therefore binds `layered = TRUE` only to array textures and asserts what it passed; it does not assert the value for a plain 2D texture.
- **Byte order of format reinterpretation.** The text defines the reinterpretation as a copy through scratch memory with ReadPixels/TexSubImage pixel types, which is native (client) byte order. The expected
  words are built with `memcpy`, so the demo is correct on any host, but it has only been run on little-endian machines.
- **`glBindImageTexture` format errors (Mesa deviation).** Section 8.23: "An INVALID_VALUE error is generated if format is not one of the formats listed in table 8.26." Mesa (radeonsi, llvmpipe, zink,
  25.0.7) accepts `RG32F`, `RG16F`, `R16F`, `R8`, `RG8` and `R11F_G11F_B10F` without error (it validates against the desktop `ARB_shader_image_load_store` list) but rejects `RGB8` correctly. Recorded as XFAIL on
  "Mesa". No exposed extension changes Table 8.26: the only matching extension is `GL_OES_shader_image_atomic`, which adds atomic image functions and no image unit formats.

## Pitfalls

- Forgetting the barrier is undefined behavior, not an error, and it usually works on a desktop GPU. The demo cannot show a failure without it; the barrier calls are there because the specification requires them.
- A shader whose `layout(...)` format differs from the unit format (or whose image type differs from the bound target) returns undefined values from `imageLoad` and `imageSize`; it does not raise an error. The shaders here
  always match the formats they are bound with.
- An access that is invalid (no texture bound, level outside the base/max range, layer out of range) loads zeros for R, G, B and writes nothing; it is not an error and not tested here.
- A texture made with `glTexImage2D` cannot be bound to an image unit even though it looks complete: use `glTexStorage*`.
- A `readonly`/`writeonly` memory qualifier is about the *variable*; the `access` argument of `glBindImageTexture` (`READ_ONLY`, `WRITE_ONLY`, `READ_WRITE`) is about the binding, and a mismatch (a shader that writes through a `READ_ONLY` binding) is undefined, not an error.
- `texelFetch` of an `RGBA32F` or integer texture needs `NEAREST` filters; the sampler read-backs set them, otherwise the texture is incomplete and the read-back returns zeros.
