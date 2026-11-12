# es20-framebuffer-objects

**API introduced in:** OpenGL ES 2.0 (`DEPTH_STENCIL_ATTACHMENT`, multisample renderbuffers, separate
read/draw targets, `glBlitFramebuffer`: ES 3.0) · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

The default framebuffer belongs to the window system. A **framebuffer object** (FBO) is a framebuffer you
build yourself out of *attachments*: images that live in a **texture** (so a later pass can sample what an
earlier pass drew: render to texture) or in a **renderbuffer** (a write-only image that exists to be
drawn into, read back, or blitted). A framebuffer has one attachment point per color buffer, one for depth,
one for stencil, and `DEPTH_STENCIL_ATTACHMENT`, which fills both with one image.

An FBO is only usable when it is **complete** (§4.4.4). This demo shows every kind of attachment working
(each one is rendered into and compared pixel by pixel with a CPU model), proves that the depth and stencil
attachments really are consulted by the depth and stencil tests, walks through the incompleteness cases that
ES 3.0 defines, shows what an incomplete framebuffer does to draw/read commands
(`INVALID_FRAMEBUFFER_OPERATION`), and checks the object queries and lifecycle rules.

## What the image shows

The default framebuffer is divided into equal cells, 4 x 3 at the default 256 x 256 (or 3 x 4, 6 x 2 or 2 x 6,
whichever gives the largest cells). Each cell shows an off-screen 64 x 64 target, centred and magnified by the
**largest integer factor** that fits (cells of 128 pixels or more keep a 4-pixel gutter) (1x at 256 x 256, 3x at 800 x 800). Texture targets are
shown by **sampling** them in a second pass. Renderbuffers cannot be sampled, so those are shown with
`glBlitFramebuffer` (NEAREST). Reading order is top row first, left to right. **Minimum surface:** cells of at least
64 x 64 (for example 256 x 192, 192 x 256, 384 x 128 or 128 x 384); smaller surfaces report `SKIP_UNSUPPORTED`.

| Cell | Target | What is drawn | What it proves |
|---|---|---|---|
| 1 | RGBA8 **texture**, no depth | red grows along x, green along y, blue is an 8 x 8 checkerboard (`pattern.frag`) | render to texture; the second pass samples exactly what pass 1 wrote |
| 2 | RGBA8 texture | the photographic negative of cell 1, drawn by a pass that *samples cell 1's texture and renders into another texture* | texture -> texture chain |
| 3 | **level 2** of a 7-level mipmapped texture (16 x 16), shown 4x magnified | the same pattern at 16 x 16 | a non-zero mipmap level is an attachable image; `TEXTURE_LEVEL` = 2 |
| 4 | RGBA8 texture + `STENCIL_INDEX8` **renderbuffer** | a stencil counter: a square is stamped with 1, a smaller one is incremented to 2, then the target is painted three times where stencil is 0 (blue), 1 (yellow), 2 (red) | a stencil attachment is used by the stencil test |
| 5 | RGBA8 **renderbuffer** + `DEPTH_COMPONENT16` renderbuffer | near green quad, far red quad drawn *later*, a blue stripe in between | the far quad does not overwrite the near one; the stripe is only visible in front of the far quad |
| 6 | RGBA8 renderbuffer + `DEPTH_COMPONENT24` renderbuffer | the same list of draws | identical result with a 24-bit depth renderbuffer |
| 7 | control: RGBA8 renderbuffer + the *same* D16 renderbuffer, `DEPTH_TEST` disabled | the same list of draws | painter's order wins: red over green, the stripe over everything (the checks must be able to fail) |
| 8 | control: stencil renderbuffer attached, `STENCIL_TEST` disabled | the same draws as cell 4 | all three paints land: the final red covers the target |
| 9 | RGBA8 texture + **one** `DEPTH24_STENCIL8` renderbuffer via `DEPTH_STENCIL_ATTACHMENT` | upper half: green near / red far quads resolved by depth. Lower half: the same quads with color writes off and stencil operations that record the *depth test outcome*, then three bands painted where stencil is 1 (cyan), 2 (magenta), 3 (yellow) | the packed image serves depth **and** stencil: stencil 2 only arises where the depth test failed |
| 10 | RGBA8 texture + `DEPTH_COMPONENT24` **texture** | the depth-tested scene of cell 5 | a texture can be the depth attachment |
| 11 | the depth texture of cell 10, sampled and shown as gray | window depths 0.25 (near quad), 0.5 (stripe), 0.75 (far quad), 1.0 (clear) as 64/128/191/255 | the depth values the GPU stored |

The captured PNG is the final framebuffer; the remaining checks (completeness, errors, queries, lifecycle)
have no image and use scratch framebuffers.

## Central calls

| Call | Role |
|---|---|
| `glGenFramebuffers`, `glBindFramebuffer`, `glDeleteFramebuffers`, `glIsFramebuffer` | name -> object on first bind; `FRAMEBUFFER` = draw + read; deleting the bound one binds 0 |
| `glGenRenderbuffers`, `glBindRenderbuffer`, `glRenderbufferStorage`, `glRenderbufferStorageMultisample`, `glDeleteRenderbuffers`, `glIsRenderbuffer` | a renderbuffer starts with 0 x 0 storage; storage gives it a **sized** internal format |
| `glFramebufferTexture2D(target, attachment, textarget, texture, level)` | attach one level of a 2D texture (render to texture) |
| `glFramebufferRenderbuffer(target, attachment, RENDERBUFFER, rb)` | attach a renderbuffer; `rb = 0` detaches |
| `glCheckFramebufferStatus` | `COMPLETE` or the violated rule |
| `glGetFramebufferAttachmentParameteriv`, `glGetRenderbufferParameteriv`, `glGetInternalformativ` | what is attached and how big its components really are |
| `glBlitFramebuffer` | only used to put renderbuffers on the screen |
| meta-gl: `ColorAttachment::Color0` + `to_framebuffer_attachment()` | color attachment points are a separate enum in meta-gl; the helper converts one to a `FramebufferAttachment` |

## Spec

- ES 3.0 §4.4.1 (binding, deleting), §4.4.2 (renderbuffers §4.4.2.1, required formats §4.4.2.2, attaching
  §4.4.2.3-4), §4.4.3 (feedback loops), **§4.4.4 (completeness)**, §4.4.4.4 (effects on commands),
  §4.1.4 stencil test, §4.1.5 depth test, §4.2.3 clearing, §6.1.13-6.1.15 (queries; initial state in
  Tables 6.14 / 6.15), Table 3.13 / 3.14 (renderable formats), §2.5 Table 2.3 (errors are ignored commands).
- GLSL ES 3.00 §7.1 `gl_VertexID`, §7.2 `gl_FragCoord`, §8.8 `texture` and `texelFetch`.

### The completeness rules the demo exercises (§4.4.4.2)

| Situation | `glCheckFramebufferStatus` | Rule |
|---|---|---|
| default framebuffer | `COMPLETE` | exists |
| new FBO, nothing attached | `INCOMPLETE_MISSING_ATTACHMENT` | at least one image |
| renderbuffer that never got storage (0 x 0) attached | `INCOMPLETE_ATTACHMENT` | "width and height of image are non-zero" |
| same FBO after `glRenderbufferStorage(RGBA8, 32, 32)` | `COMPLETE` | storage calls change completeness |
| `DEPTH_COMPONENT16` renderbuffer at `COLOR_ATTACHMENT0` | `INCOMPLETE_ATTACHMENT` | must be color-renderable |
| `RGBA8` renderbuffer at `DEPTH_ATTACHMENT` / `STENCIL_ATTACHMENT` | `INCOMPLETE_ATTACHMENT` | must be depth- / stencil-renderable |
| `RGB9_E5` texture at `COLOR_ATTACHMENT0` | `INCOMPLETE_ATTACHMENT` | texture-only format; `glTexImage2D(RGBA8)` on the same texture fixes it |
| `DEPTH_COMPONENT16` + `STENCIL_INDEX8` as two renderbuffers | `UNSUPPORTED` | "Depth and stencil attachments, if present, are the same image" |
| 32 x 32 color + 16 x 16 depth | `COMPLETE` | **`INCOMPLETE_DIMENSIONS` "cannot be generated by an OpenGL ES 3.0 implementation"**; rendering is limited to the intersection (§4.4.1) |
| multisampled color + single-sampled depth renderbuffer | `INCOMPLETE_MULTISAMPLE` | `RENDERBUFFER_SAMPLES` must be equal |
| last image detached (`texture = 0`) | `INCOMPLETE_MISSING_ATTACHMENT` | |

(In ES 3.1/3.2 a framebuffer without attachments is complete if its default width and height are non-zero;
they are zero initially, so the first row is the same on an ES 3.2 context.)

## Shader concepts

`quad.vert` builds a rectangle from `gl_VertexID` (`TRIANGLE_STRIP`, 4 vertices, no buffers). The rectangle
and its NDC depth come from two uniforms, so the same shader draws full-target quads and the small
depth/stencil rectangles. `pattern.frag` computes the test image from `gl_FragCoord` (pixel centers are at
`i + 0.5`, so `ivec2(gl_FragCoord.xy)` is the pixel index). `flat.frag` writes one color. `show.frag` is the
second pass: `texture()` with `NEAREST`, or `texelFetch()` of one mip level, optionally inverted or shown as
gray (`u_gray` copies red into all channels, which is how a depth texture's value is displayed). The branches
are on uniforms, so every invocation takes the same path and implicit derivatives stay defined.

## How it is validated

- **CPU models.** The pattern image is an analytic formula. The depth/stencil scenes are lists of
  rectangles, and **one** list drives both the GL calls and a small CPU model of the per-fragment pipeline
  (stencil test, depth test, stencil update rules of §4.1.4/§4.1.5, color/depth/stencil buffers), so they
  cannot drift apart. Rectangle edges lie on pixel boundaries, so rasterization is unambiguous.
- **Each target is read back through its own framebuffer** with `glReadPixels` and compared with the model
  for every pixel of the 64 x 64 image. Tolerance **+-1** wherever a shader output is converted to 8 bits:
  equation 2.3 only promises "one of the two integers closest to f x 255", so a value that lands near a
  half-step may legally differ by one. The measured maximum deviation is printed with each check (1 on
  radeonsi for the gradient, 0 elsewhere).
- **Each cell on the screen** is compared with what was read back from its framebuffer, magnified by the
  cell's integer factor `m` (window pixel `(i, j)` shows texel `(i / m, j / m)`, sampled at least `1/(2m)` of a
  texel away from every texel edge): +-1 for the sampled textures (a second float -> 8-bit conversion) and
  **exactly 0** for the blitted renderbuffers (NEAREST with an integer magnification copies stored values
  between equal 8-bit formats).
- **Negative controls (checks that can fail).** (1) References with a gradient step of 1/64 instead of
  1/63, and shifted by one texel, are rejected for 3000+ and 4000+ of 4096 texels. (2) The two control cells
  (depth test off, stencil test off) are compared with the model *and* required to differ from the tested
  cells at named pixels: the near/far overlap, the stripe behind the near quad, the pixel outside the stencil
  square. As a mutation test, leaving out the depth attachment of cell 5, the stencil attachment of cell 4 and
  replacing `DEPTH_STENCIL_ATTACHMENT` by a depth-only attachment made 18 checks fail (image comparisons,
  named probes and attachment queries).
- **Packed depth/stencil.** The stencil value 2 in cell 9 only exists where quad B *failed the depth test*
  and was then incremented, so the cell proves the depth part and the stencil part of one `DEPTH24_STENCIL8`
  image at once.
- **Completeness** is the table above: each row is a scratch framebuffer in exactly that state.
- **`INVALID_FRAMEBUFFER_OPERATION`** is provoked on an incomplete framebuffer with `glClear`,
  `glClearBufferfv`, `glDrawArrays`, `glReadPixels`, `glCopyTexImage2D`, `glCopyTexSubImage2D` and
  `glBlitFramebuffer` (incomplete *read* framebuffer). The same calls succeed on the completed framebuffer
  (control), and a rejected clear and draw leave the attached texture unchanged (Table 2.3: the erroneous
  command is ignored).
- **Queries.** `OBJECT_TYPE` / `OBJECT_NAME` / `TEXTURE_LEVEL` / `TEXTURE_CUBE_MAP_FACE`, component sizes
  (at least the requested ones: §4.4.2.2 says "at least"), `COMPONENT_TYPE` (`UNSIGNED_NORMALIZED`),
  `COLOR_ENCODING` (`LINEAR`), the shared object of `DEPTH_STENCIL_ATTACHMENT`, an empty attachment point
  (`NONE`, name 0, `INVALID_OPERATION` for a size query, §6.1.13), and renderbuffer width/height/format/
  samples for seven sized formats plus the initial state of a new renderbuffer (0 x 0, `RGBA4`, Table 6.15).
- **Lifecycle.** `glIs*` is false for a generated-but-unbound name and for 0; `FRAMEBUFFER_BINDING`,
  `DRAW_/READ_FRAMEBUFFER_BINDING` and `RENDERBUFFER_BINDING` follow the binds; deleting a bound object binds
  0; deleting a renderbuffer or texture that is attached to the *bound* framebuffer detaches it (§4.4.2.3/4).
- **Defined errors of the attach/storage calls.** Default framebuffer bound -> `INVALID_OPERATION`; a name
  that is no renderbuffer -> `INVALID_OPERATION`; a 2D texture with a cube-map face `textarget` ->
  `INVALID_OPERATION`; `width > MAX_RENDERBUFFER_SIZE` -> `INVALID_VALUE`; more samples than the format
  supports -> `INVALID_OPERATION`.

## Not tested, on purpose (documented here)

- **Feedback loops (§4.4.3).** Sampling a texture that is attached to the draw framebuffer (or copying into
  an image attached for reading) gives undefined results while the framebuffer stays *complete*. The demo
  unbinds every texture before rendering into it and renders into a texture only while no sampler can read it.
- **Contents of an attachment outside the intersection of unequal-sized attachments**, and of a renderbuffer
  after `glRenderbufferStorage` (undefined).
- **Dangling attachments of unbound framebuffers.** Deleting a texture or renderbuffer detaches it only from
  the *bound* framebuffer; other framebuffers keep a reference to a deleted name and the application must
  clean up (§4.4.2.3/4). Drivers may keep the object alive.
- **`FRAMEBUFFER_UNSUPPORTED` beyond the one normative case** (separate depth and stencil images): which
  other format combinations are unsupported is implementation-dependent.
- Multiple render targets, blits with scaling/filtering, multisample resolve, layered attachments and
  invalidation belong to later ES 3.0 demos.

## Run

```
./build/debug/bin/es20-framebuffer-objects --headless --validate --capture framebuffer-objects.png
```

## Pitfalls

- **Check the status.** `glCheckFramebufferStatus` is cheap; an incomplete framebuffer turns every draw,
  clear and read into `INVALID_FRAMEBUFFER_OPERATION`, and the pixels simply do not change.
- **A new renderbuffer has no storage** and attaching it gives `INCOMPLETE_ATTACHMENT` until
  `glRenderbufferStorage` is called. The status depends on the *current* state of the attached objects, so
  changing a texture's format or a renderbuffer's storage after attaching can break a framebuffer that was
  complete.
- **Not every format is color-renderable.** The unsized `RGB`/`RGBA` and the sized formats marked in Table
  3.13 are; `RGB9_E5`, `SRGB8`, the `SNORM` formats and (in ES 3.0) the floating-point formats are
  texture-only. Depth/stencil formats go to the depth/stencil points only.
- **Attaching a non-base mip level** needs a mipmap-complete texture (the whole chain, §4.4.4.1) unless the
  base level is attached; undefined contents of the other levels do not matter.
- **Depth and stencil must be one image** in ES 3.0: attach a `DEPTH24_STENCIL8` renderbuffer to
  `DEPTH_STENCIL_ATTACHMENT` instead of two renderbuffers.
- **No depth buffer, no depth test.** A framebuffer object without a depth attachment behaves as if the depth
  test always passed, even with `DEPTH_TEST` enabled (cells 5, 6, 9 and 10 would silently draw in painter's
  order). The default framebuffer's depth buffer is not available to an FBO.
- **Framebuffer objects have no back buffer.** `glReadBuffer` / `glDrawBuffers` use `COLOR_ATTACHMENTi` for
  them and `BACK` only for the default framebuffer.
- **Different attachment sizes are legal in ES 3.0** and rendering is clipped to their intersection;
  ES 2.0's `INCOMPLETE_DIMENSIONS` is gone.
- **Dithering** (initially enabled) may change a stored 8-bit value; the demo disables it before comparing.
- **Depth textures** need `NEAREST` (or no mipmap filter) to be sampled with a plain `sampler2D`; the default
  `NEAREST_MIPMAP_LINEAR` would make a one-level texture incomplete.

## Known driver deviations (XFAIL)

- **ANGLE** reports `RENDERBUFFER_RED/GREEN/BLUE/ALPHA_SIZE` = 4 for a renderbuffer that has no storage yet (the sizes of
  its initial internal format `RGBA4`). Table 6.15 (ES 3.2 Table 21.17) lists 0 as the initial value of every
  `RENDERBUFFER_*_SIZE`, and §4.4.2.1 creates the object "with a zero-sized memory buffer". XFAIL on "ANGLE".
