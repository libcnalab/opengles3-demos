# es32-layered-rendering

Layered rendering in OpenGL ES 3.2. `glFramebufferTexture` attaches a *whole level* of a 2D array, 3D or cube map texture
to a framebuffer; every layer is then a render target, and a geometry shader chooses the layer of each primitive it
emits by writing the built-in `gl_Layer`. One draw call can therefore render different content into every layer, which is
the basis of single-pass cube map rendering (shadow cubes, environment maps) and of rendering volume slices.

**Minimum version:** OpenGL ES 3.2 and GLSL ES 3.20 (`#version 320 es`). `glFramebufferTexture` and `gl_Layer` do not exist
in ES 3.1; softpipe reports `SKIP_UNSUPPORTED`.

## What you see (256x256, a 2x2 grid of 128x128 cells)

| cell | content |
|---|---|
| top left | a **mirror ball** (a 96x48 UV sphere) inside a box-shaped room. The room's walls, window, door, lamp, carpet, stripe, poster and three boxes were rendered once into a cube map; the ball reflects it with `reflect()` and the background is a skybox, both read from that cube map. The camera orbits in windowed mode. |
| top right | the **same cube map as a cube net**: six 32x32 tiles (-X +Z +X -Z in the middle row, +Y above, -Y below). Each tile pixel looks up a texel of a face through a *direction* vector. |
| bottom left | a **2D array texture** 64x64x4 shown as four tiles. Layer n has its own background and body colour, a body rectangle that shrinks with n, and n+1 white squares (the layer index as a count). It was painted by four POINTS: the layer of point n is `gl_PrimitiveIDIn`. |
| bottom right | a **3D texture** 64x64x4 shown as four tiles. Slice k has a background that goes from blue to green, a block that walks along the diagonal and a bar that grows. It was painted by ONE point with `layout(invocations = 4)`: the slice is `gl_InvocationID`. |

The cube map is rendered by `shaders/env.geom`: `layout(triangles, invocations = 6) in`, every input triangle is emitted
six times, invocation k with the view-projection matrix of face k and `gl_Layer = gl_InvocationID`. The fragment shader reads
`gl_Layer` back (GLSL ES 3.20 section 7.1.5) and stores `40 * layer + 15` in alpha.

## Central calls and where the specification talks about them

* `glFramebufferTexture(target, attachment, texture, level)` (ES 3.2 section 9.2.8): if the texture is of a layered type the
  attachment is **layered**; `FRAMEBUFFER_ATTACHMENT_LAYERED` returns TRUE. `glFramebufferTextureLayer` (3D and array
  textures, and cube maps: layer -> face by Table 8.24) and `glFramebufferTexture2D` attach one image and give FALSE.
  Section 9.2.8.1 lists the state set by each call. `glGetFramebufferAttachmentParameteriv` (section 9.2.3): for a
  non-layered 3D or array attachment `TEXTURE_LAYER` is the layer, otherwise zero.
* Completeness (sections 9.4.1 and 9.4.2): "If any framebuffer attachment is layered, all populated attachments must be
  layered. Additionally, all populated color attachments must be from textures of the same target" - otherwise
  `FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS`. A non-layered attachment must select an existing layer
  (`FRAMEBUFFER_INCOMPLETE_ATTACHMENT`), a layered one needs at most `MAX_FRAMEBUFFER_LAYERS` layers.
* Section 9.8 (Layered Framebuffers) and 11.3.4.5 (Layer Selection):
  * the layer is zero if there is no geometry shader or it does not statically assign `gl_Layer`;
  * "A layer number written by a geometry shader has no effect if the framebuffer is not layered";
  * `Clear` / `ClearBuffer*` clear **all** layers; `ReadPixels` reads **layer zero**;
  * a cube map has six layers, 0..5 = +X, -X, +Y, -Y, +Z, -Z (Table 8.24; layer*6+face for cube map arrays);
  * the layer of a primitive comes from its provoking vertex, "implementation-dependent" and reported by
    `LAYER_PROVOKING_VERTEX` (FIRST_VERTEX_CONVENTION, LAST_VERTEX_CONVENTION or UNDEFINED_VERTEX); to get defined results
    all vertices of a primitive set the same layer, and `EndPrimitive()` may separate vertices with different layers;
  * a layer that is negative or at least the layer count of the smallest attachment makes the effect on the framebuffer
    **undefined**; this demo never draws such a primitive and asserts nothing about it.
* `glFramebufferParameteri(FRAMEBUFFER_DEFAULT_LAYERS)` (section 9.2.1): an attachment-less framebuffer is layered iff the
  value is non-zero; limited by `MAX_FRAMEBUFFER_LAYERS` (>= 256).
* The cube map face table (section 8.13, Table 8.19) decides which direction maps to which face and texel. The demo's view
  matrices (`cpu_model.hpp: face_camera`) make window x grow with `s` and window y with `t`: face +X looks along +X with up =
  -Y, +Y looks along +Y with up = +Z, and so on. This is why a cube map image looks upside down when read with
  `glReadPixels`; the cube net flips the rows to show the side faces upright.

## Shader concepts

* `shaders/array_layers.geom`, `shaders/volume_slices.geom` - three ways to choose a layer: from `gl_PrimitiveIDIn`
  (point n), from `gl_InvocationID` (invocation k), from a plain constant. All coordinates are integers / 32, so every
  rectangle is pixel-aligned and the result is exact.
* `shaders/env.geom`, `shaders/env.frag` - single-pass cube map rendering with a `uniform mat4 u_viewproj[6]` and the
  fragment-shader input `gl_Layer`.
* `shaders/sky.frag`, `shaders/sphere.frag` - `texture(samplerCube, direction)` and `reflect()`; the ball adds a small rim term.
* `shaders/show_*.frag` - `texelFetch` from `sampler2DArray` / `sampler3D`; `layout(binding = 0)` on the samplers.

## How it is validated (`--validate`)

1. **Every layer, byte for byte.** The four layers of the array and the four slices of the 3D texture are read back
   (`glFramebufferTextureLayer` on a scratch read framebuffer + `glReadPixels`) and compared with the CPU images
   (`cpu_model.hpp: array_layer`, `volume_slice`) with tolerance zero. The six cube faces are read back with
   `glFramebufferTexture2D` and compared texel by texel with a **ray caster** of the same room (the room is a list of
   axis-aligned quads that feeds both the vertex buffer and the ray caster; the direction of a texel comes from the face
   table). This proves the view matrices, the face numbering and the orientation of Table 8.19 at once. 21 736 of 24 576
   texels are compared; the 2 840 texels whose 3x3 neighbourhood (+-0.75 texel) crosses a colour edge are skipped because
   rasterization decides there. The alpha tag proves that the fragment shader's `gl_Layer` equals the face written.
2. **The displays.** The two bottom cells (all 128x128 pixels each) equal the layer images exactly. The 6 tiles of the cube
   net equal the texels they sample. The mirror-ball cell is compared with a model that casts the view ray, intersects the
   analytic sphere, applies `reflect()` and the rim term, and looks up the room: about 1 500 sky and 570 ball pixels
   (the ball reflects 12 different environment colours), +-2/255 (8-bit rounding plus bilinear weights on equal texels; the
   model uses the analytic sphere; the facet error of the tessellated ball is far below the 0.08 rad neighbourhood). Probes are dropped within 3 px of the ball's
   outline and where the room's colour is not constant within 0.08 rad of the reflected direction. Measured: at most 1/255.
3. **Queries.** `LAYERED`, `OBJECT_TYPE`, `TEXTURE_LEVEL` and `TEXTURE_LAYER` for eight attachment kinds (layered 2D array, 3D,
   cube, plain 2D; single layer of array and 3D; cube face; plain 2D via `glFramebufferTexture2D`), and after detaching.
4. **Completeness.** sixteen configurations: COMPLETE for each layered target alone, `INCOMPLETE_LAYER_TARGETS` for a layered
   colour beside a depth renderbuffer, beside a non-layered colour, beside a layered colour of another target (3D, cube),
   each with a control that differs only in the rule under test (non-layered colour + the same renderbuffer is COMPLETE;
   two layered array colours are COMPLETE), `INCOMPLETE_ATTACHMENT` for layer 4 of a 4-layer array, and the demo's own
   environment framebuffer (layered cube colour + layered cube depth).
5. **Errors.** `glFramebufferTexture*` with the default framebuffer bound (INVALID_OPERATION), with a name that is not a
   texture, level 1 of a one-level immutable texture and level -1 (INVALID_VALUE), `glFramebufferTextureLayer` with layer =
   `MAX_ARRAY_TEXTURE_LAYERS` or -1 (INVALID_VALUE) and with a plain 2D texture (INVALID_OPERATION),
   `glFramebufferTexture2D(TEXTURE_2D)` with an array texture's name (INVALID_OPERATION); a failed call changes no state.
6. **The semantics of `gl_Layer`.** On an 8x8x4 array: no geometry shader -> only layer 0; a geometry shader without a
   `gl_Layer` assignment -> only layer 0; `gl_Layer = gl_InvocationID` with the fragment shader reading `gl_Layer` (red =
   64k + 63 in layer k); `glReadPixels` on the layered framebuffer returns layer 0; `glClearBufferfv` clears all four
   layers; `gl_Layer = 3` into a non-layered framebuffer with layer 1 attached paints layer 1 only; two triangles with layers
   1 and 3 separated by `EndPrimitive()` paint exactly layers 1 and 3; and, when `LAYER_PROVOKING_VERTEX` is FIRST or LAST, a
   triangle with vertex layers 0, 1, 2 lands in layer 0 or 2.
7. `FRAMEBUFFER_DEFAULT_LAYERS` round trip, `INVALID_VALUE` above `MAX_FRAMEBUFFER_LAYERS` and for -1.

**Negative controls:** layer 0 compared with the model of layer 1; a cube face compared with the model of the opposite face
(thousands of mismatching texels) - the check would fail if layers or faces were mixed up.

## Pitfalls and findings

* `glFramebufferTexture` is the only way to attach a layered image; with a layered attachment *all* populated attachments
  (depth too) must be layered, so the environment framebuffer uses a cube map **depth texture**, not a renderbuffer.
* Even `texelFetch` needs a complete texture, and the default minification filter is a mipmap filter, so the single-level
  textures get `NEAREST` filters; sampler objects then override the filters for the cube map (NEAREST for the net, LINEAR
  for the ball and the sky).
* gl_Layer values outside the layer count are undefined; never write one deliberately.
* **Known deviation (Mesa: radeonsi, llvmpipe, zink):** `glFramebufferTextureLayer` with a cube map texture raises no
  error but does not attach it (the previous attachment stays). ES 3.2 section 9.2.8 lists cube maps and Table 8.24 as the
  layer -> face translation. The check is strict and reported as XFAIL on "Mesa"; the demo reads cube faces with
  `glFramebufferTexture2D` instead, which is equivalent.
* All three Mesa drivers report `LAYER_PROVOKING_VERTEX` = `UNDEFINED_VERTEX`, which the specification allows. The branch
  that tests FIRST / LAST therefore runs on no machine of the matrix.
* The cube map face images are stored with t pointing "down" in world space for the side faces (t grows with -y); the cube
  net flips the rows for display.

Run: `build/debug/bin/es32-layered-rendering --headless --validate --capture out.png`.
