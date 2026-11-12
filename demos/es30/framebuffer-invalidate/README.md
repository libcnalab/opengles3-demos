# es30-framebuffer-invalidate

**API introduced in:** OpenGL ES 3.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 ·
**Run:** `es30-framebuffer-invalidate [--headless --validate]`

## Purpose

`glInvalidateFramebuffer` and `glInvalidateSubFramebuffer` tell the GL that some framebuffer contents are
not needed any more, so that it may skip saving them. On tile-based GPUs that saves a memory write per tile (the
classic use: invalidate the depth and stencil buffers, and the color buffer of a multisample target, once a
frame is complete); on every GPU it documents intent. The price is that the invalidated contents become
**undefined**. This demo therefore does something unusual for a validation program: it never asserts the contents of anything
it invalidated. It validates what the specification *does* define:

1. every legal call raises no error, including calls that name attachments that do not exist, an empty list, regions that
   are partly or entirely outside the image, regions of size zero, and incomplete framebuffers (which "may be ignored");
2. every illegal call raises the exact error section 4.5 lists, and a failed call has no effect;
3. **nothing but the invalidated part changes**: other attachments, the other half of a packed depth/stencil image,
   pixels outside a region, the other framebuffer when READ and DRAW differ, other layers or faces of the same texture;
4. rendering after an invalidation works normally.

It also covers framebuffer **attachment queries** that the other framebuffer demos do not use: `TEXTURE_LAYER`,
`TEXTURE_LEVEL` and `TEXTURE_CUBE_MAP_FACE` of texture attachments, `COMPONENT_TYPE` of float, signed and unsigned integer
and normalized formats, and the rules for the combined `DEPTH_STENCIL_ATTACHMENT` point. Attaching a layer of a 3D or
array texture with `glFramebufferTextureLayer` is part of that (it is what makes `TEXTURE_LAYER` non-zero) and is verified
by rendering into it.

## What the image shows

The window is a 4 x 4 grid of cells, top row first, that fills it at any size: the cell edges sit at `i * width / 4` and
`j * height / 4`. Every cell is a `NEAREST` `glBlitFramebuffer` that stretches one 64 x 64 image onto the cell, and every
image has its own base color, dark at the left and brighter to the right, with horizontal stripes:

```
 +X  -X  +Y  -Y         the six faces of a cube map
 +Z  -Z  L0  L1         then the four layers of a 2D array texture
 L2  L3  S0  S1         then the four slices of a 3D texture
 S2  S3  C0  C1         then two attachments of a framebuffer whose attachments were all invalidated and re-rendered
```

The last two cells are the "gradient with checkerboard" charts of `mrt.frag` (the same fragment shader writes three
attachments). The images themselves are always 64 x 64; only the blit that shows them scales. The window check compares
every pixel of every cell with the source pixel that a stretching `NEAREST` blit reads for it: the destination pixel
centre `d + 0.5` maps to `(d + 0.5) * 64 / n` in a cell `n` pixels long, and the source pixel whose square contains that
point is the expected one. Where the point lies within 0.05 pixel of a source pixel edge (this happens exactly at some
cell sizes, e.g. 160), the implementation's rounding decides and either neighbour is accepted. The smallest window is
64 x 64; below that the demo reports SKIP. If a face, layer or slice had been attached wrongly, two cells would show the same base color. The default
framebuffer's depth and stencil are invalidated at the end of the frame (`glInvalidateFramebuffer(FRAMEBUFFER, {DEPTH,
STENCIL})`); the picture is the color buffer, which that call must not touch.

## Central calls

| Call | Role | Spec |
|---|---|---|
| `glInvalidateFramebuffer(target, n, attachments)` | invalidate whole attachments | §4.5 |
| `glInvalidateSubFramebuffer(target, n, attachments, x, y, w, h)` | invalidate a rectangle of them | §4.5 |
| `glGetFramebufferAttachmentParameteriv(target, attachment, pname, &v)` | what is attached and how | §6.1.13 |
| `glFramebufferTextureLayer(target, attachment, texture, level, layer)` | one layer of a 3D / array texture | §4.4.2.4 |
| `glFramebufferTexture2D(target, attachment, textarget, texture, level)` | a cube-map face or a mip level | §4.4.2.4 |

In meta-gl the attachment list has two types because the tokens differ: `FramebufferAttachment` for framebuffer objects
(`Depth`, `Stencil`, `DepthStencil`, and the 32 color points converted with `to_framebuffer_attachment(ColorAttachment)`) and
`DefaultFramebufferAttachment` (`Color`, `Depth`, `Stencil` = `GL_COLOR`, `GL_DEPTH`, `GL_STENCIL`) for the default
framebuffer. Passing the wrong kind for the bound framebuffer compiles, and is the `INVALID_ENUM` case of the spec. Querying the default framebuffer
uses `DefaultFramebufferQueryAttachment` (`Back`, `Depth`, `Stencil`).

## Key ideas

- **Invalidation is a hint with undefined results, not a clear.** After `glInvalidateFramebuffer` the attachment may keep
  its value, hold anything, or be zeroed. The demo prints (never asserts) what this driver does; on the Mesa drivers of the
  test matrix nothing changes. The checks below are therefore guarantees *against over-invalidation*: they are what an
  implementation that discards too much would violate.
- **Only the listed part is invalidated.** `DEPTH_ATTACHMENT` of a `DEPTH24_STENCIL8` image invalidates the depth part
  only, "the other portion must be preserved" (§4.5), `DEPTH_STENCIL_ATTACHMENT` invalidates both. A region leaves all
  pixels outside it alone; parts of the region outside the image or the window are ignored.
- **Which tokens are legal depends on the bound framebuffer.** Framebuffer object: `COLOR_ATTACHMENTi` (`i` below
  `MAX_COLOR_ATTACHMENTS`), `DEPTH_ATTACHMENT`, `STENCIL_ATTACHMENT`, `DEPTH_STENCIL_ATTACHMENT`. Default framebuffer:
  `COLOR`, `DEPTH`, `STENCIL`. The wrong kind is `INVALID_ENUM`; a color point at or above `MAX_COLOR_ATTACHMENTS` is
  `INVALID_OPERATION`; negative counts or sizes are `INVALID_VALUE`. An attachment that is legal but has no image is
  ignored.
- **The target argument.** `FRAMEBUFFER` means `DRAW_FRAMEBUFFER`; `READ_FRAMEBUFFER` invalidates the *read* framebuffer. The
  demo binds two different framebuffers and shows the other one is untouched in each case.
- **A texture attachment is one image.** Invalidating the attachment that is layer 1 of a 2D array texture, slice 2 of a
  3D texture or one cube face leaves the other layers, slices and faces unchanged.
- **Attachment queries.** `TEXTURE_LAYER` is the attached layer for 3D and array textures and 0 otherwise;
  `TEXTURE_CUBE_MAP_FACE` is the face's target (`TEXTURE_CUBE_MAP_POSITIVE_X`, ...) for cube maps and 0 otherwise;
  both, and `TEXTURE_LEVEL`, are `INVALID_ENUM` for a renderbuffer attachment and `INVALID_OPERATION` for an empty
  one. `COMPONENT_TYPE` is `UNSIGNED_NORMALIZED`, `SIGNED_NORMALIZED`, `FLOAT`, `INT` or `UNSIGNED_INT` for the *format*
  of the attached image: the query works on any attached image, also on formats that are not color-renderable in ES 3.0
  (RGBA16F and RGBA8_SNORM: the framebuffer need not be complete for the query), and it is `NONE` for a
  texture level that has no image.
- **`DEPTH_STENCIL_ATTACHMENT` queries** describe "that object" and require that the *same* object is at the depth and the
  stencil point: with one `DEPTH24_STENCIL8` image the sizes are the object's (`DEPTH_SIZE >= 24` and
  `STENCIL_SIZE >= 8`, `RED_SIZE` 0) and `COMPONENT_TYPE` is `INVALID_OPERATION` (a packed image has no single format); with
  two different renderbuffers (or only one of the two points filled) every query is `INVALID_OPERATION`.
- **Cube maps and `glFramebufferTextureLayer`.** ES 3.0 and 3.1 accept only 3D and 2D array textures; ES 3.2 §9.2.8 also
  accepts a cube map and reads the layer as a face. The demo asserts the 3.2 behaviour in a context that reports 3.2 and only prints it
  in a context that reports less (the minimum-version runs only change what the driver reports).

## Shader concepts

`pattern.frag` computes the colour of every image from `gl_FragCoord` and a uniform base color; `mrt.frag` writes three
attachments with `layout(location = 0..2)`; `quad.vert` builds a rectangle from `gl_VertexID`. The depth/stencil
state is observed with draws of `flat.frag` (see below).

## Validation (`--validate`)

303 checks in an ES 3.2 context (302 in a context that reports 3.0/3.1, where the cube-map `glFramebufferTextureLayer` assertion
is a printed note). The demo is built around one **subject** framebuffer: `COLOR_ATTACHMENT0` and `1` are RGBA8
textures, `COLOR_ATTACHMENT2` an RGBA8 renderbuffer and `DEPTH_STENCIL_ATTACHMENT` a `DEPTH24_STENCIL8` renderbuffer. A scene
fills it: the three color attachments with the `mrt.frag` patterns, depth/stencil with 0.25/stencil 1 in the left half and
0.75/stencil 2 in the right half.

- **Observing the subject.** Colors are read back with `glReadPixels` (exactly, against a snapshot). Depth and stencil
  cannot be read in ES, so they are *probed*: a second framebuffer with its own color texture but the **same**
  depth/stencil renderbuffer is drawn with a full quad four times: `DEPTH_TEST LESS` at window depth 0.5 passes where the
  stored depth is greater, `STENCIL_TEST EQUAL` 1, 2 and 0 pass where the stored stencil equals the reference (depth writes
  off, stencil ops `KEEP`, so probing does not change what it probes). The fill is compared with the model first (colors +-1, equation 2.3; depth/stencil exactly).
- **Negative controls.** After a `glClear` of the subject, every color attachment differs from the snapshot in over 3000
  of 4096 pixels, so the exact comparison can fail; a framebuffer whose depth/stencil was just cleared probes differently from the scene in
  every pixel, so the probe can fail.
- **Invalidation scenarios** (each: refill, snapshot, call, no error, compare): `COLOR_ATTACHMENT0`; `COLOR_ATTACHMENT1` and `2` on
  `DRAW_FRAMEBUFFER`; `DEPTH_ATTACHMENT` alone (stencil unchanged); `STENCIL_ATTACHMENT` alone (depth unchanged);
  `DEPTH_STENCIL_ATTACHMENT`; depth and stencil in one list; a legal color point with no image; an empty list;
  `InvalidateSub` regions of color and of depth/stencil (everything outside the hole unchanged), a region with a negative
  origin, one reaching beyond the image, one entirely outside, and widths or heights of 0 (nothing changes);
  `READ_FRAMEBUFFER`, `DRAW_FRAMEBUFFER` and `FRAMEBUFFER` with two different framebuffers (the other one is untouched in every
  part); an incomplete framebuffer and absent depth/stencil on a color-only one (no error); and invalidating the attachment that
  is a layer, a slice or a cube face (the other images unchanged).
- **Errors** (`glInvalidateFramebuffer` and `...Sub...`): `GL_COLOR`, `GL_DEPTH`, `GL_STENCIL` with a framebuffer object bound,
  `GL_NONE` in a list, and (on the default framebuffer) `DEPTH_ATTACHMENT`, `DEPTH_STENCIL_ATTACHMENT`, `COLOR_ATTACHMENT0`
  and `GL_NONE`: `INVALID_ENUM`; `COLOR_ATTACHMENT(MAX_COLOR_ATTACHMENTS)`: `INVALID_OPERATION`; `numAttachments`, `width` or
  `height` -1: `INVALID_VALUE`. After all of them the subject is unchanged ("a command that generates an error has no effect"),
  including a list that mixes a legal attachment with an illegal one.
- **Default framebuffer:** every combination of `COLOR`, `DEPTH`, `STENCIL`, with a region, with a region far larger than the
  window and with an empty list raises no error. After invalidating all three a clear gives the clear color, a depth-tested
  nearer quad survives a farther one, and without the depth test the farther quad overwrites it. The frame ends with an
  invalidation of depth and stencil, and all 16 window cells still equal their CPU patterns (+-1).
- **Rendering after invalidation:** after invalidating *all* attachments of the subject, filling again reproduces the
  images (cells C0 and C1 of the window); at the end of validation every image is rendered again and all 16 cells are
  compared a second time.
- **Attachment queries:** `TEXTURE_LAYER` 0..3 for array layers and 3D slices and the mip-level-1 attachments
  (`TEXTURE_LEVEL` 1); `TEXTURE_CUBE_MAP_FACE` of each of the six faces (and `TEXTURE_LAYER` 0); 0 for all of them on the
  other texture kinds; a 2D texture's mip level 2; a detached point (`NONE`, name 0, `INVALID_OPERATION` for the rest); a
  renderbuffer attachment (`INVALID_ENUM` for the three texture pnames); rendering into level 1 of the array draws a 32 x 32
  pattern. `COMPONENT_TYPE` of 22 texture formats and 9 renderbuffer formats, `DEPTH_COMPONENT32F` as `FLOAT`, and a texture without an
  image as `NONE`; every renderable format's framebuffer is complete. The `DEPTH_STENCIL_ATTACHMENT` rule: packed
  renderbuffer and packed texture, two different renderbuffers, depth only, nothing attached; which attachment tokens are
  legal on the default framebuffer and on an object.
- **`glFramebufferTextureLayer`:** rendering into one layer, slice or face changes that image and no other (all images read
  back); `INVALID_VALUE` for layer -1, `MAX_ARRAY_TEXTURE_LAYERS`, `MAX_3D_TEXTURE_SIZE`, level 100; `INVALID_OPERATION` for a
  plain 2D texture; a failed call leaves the attachment as it was.

## Known driver deviations and limits of the evidence (Mesa 25.0.7: radeonsi, llvmpipe, zink)

- **Querying a non-BACK/DEPTH/STENCIL attachment of the default framebuffer** (`DEPTH_STENCIL_ATTACHMENT`, `COLOR_ATTACHMENT0`,
  `DEPTH_ATTACHMENT`) raises `INVALID_ENUM`; ES 3.0 §6.1.13 and ES 3.2 §9.2.3 say `INVALID_OPERATION`. The three checks keep the
  strict expectation and are reported as XFAIL on these drivers. (For `glInvalidateFramebuffer` the spec says `INVALID_ENUM`, and
  the drivers agree.)
- **Mesa does not discard anything**: the invalidated attachments read back unchanged, so the "preserved" checks cannot tell a no-op
  from a precise implementation; they would catch an implementation that invalidates more than it was asked to. The negative
  controls show the comparisons can fail.
- `es-min` runs use `MESA_GLES_VERSION_OVERRIDE`, which only changes the reported version; the cube-map
  `glFramebufferTextureLayer` rule of ES 3.0 / 3.1 (`INVALID_OPERATION`) is therefore not asserted.
- `INVALID_ENUM` for a `target` that is not `FRAMEBUFFER` / `DRAW_FRAMEBUFFER` / `READ_FRAMEBUFFER` is not reachable through
  meta-gl's typed enums, hence not checked.

ANGLE (2.1.17841, Vulkan backend), reported as XFAIL on "ANGLE":

- **`glBlitFramebuffer` from a 3D texture slice reads slice 0.** Cells 11..13 show slice 0's pattern. The slices hold the right
  images (a `glReadPixels` of the same attachment returns them, checked strictly), 2D array layers blit correctly, and a fresh
  framebuffer object does not help. Nothing about invalidation is involved: these slices are never invalidated before the check.
  The window check is therefore split into the 13 other cells (strict), the slice read-back (strict) and the three 3D blits.
- **The two `glGetFramebufferAttachmentParameteriv` attachment errors are swapped**: `BACK` with a framebuffer object bound
  raises `INVALID_OPERATION` (§9.2.3: `INVALID_ENUM`, it is not in Table 9.1) and `COLOR_ATTACHMENTm`, m >= `MAX_COLOR_ATTACHMENTS`,
  raises `INVALID_ENUM` (§9.2.3: `INVALID_OPERATION`).
- Not a deviation: with nothing at the depth and stencil points, querying `DEPTH_STENCIL_ATTACHMENT` raises `INVALID_OPERATION` on
  ANGLE and returns `OBJECT_TYPE` = `NONE` on Mesa. The text supports both ("the same object must be bound to both" is not met;
  the listed error is only for "different objects"), so either outcome is accepted.

## Pitfalls

- **Never read an invalidated attachment expecting your old pixels.** And never rely on invalidation to *clear*: it may
  leave the old values, which hides bugs on desktop and Mesa and shows them on a tile-based GPU.
- Invalidate **after** the last use of the contents (after the resolve blit, after the last draw that reads the depth buffer), and
  before the next `glBindFramebuffer` or buffer swap, so the implementation can act on it.
- The attachment list's tokens depend on the bound framebuffer: `GL_COLOR` for the default framebuffer is not `COLOR_ATTACHMENT0`.
- `glInvalidateFramebuffer` takes no region; its implicit rectangle is the largest image of the framebuffer. Use
  `glInvalidateSubFramebuffer` to keep part of an attachment.
- Reading `FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER` of a renderbuffer attachment is an error, not 0.
