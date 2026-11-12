# es30-multiple-render-targets

**API introduced in:** OpenGL ES 3.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 ·
**Run:** `es30-multiple-render-targets [--headless --validate]`

## Purpose

OpenGL ES 2.0 had one color output per fragment. ES 3.0 lets a fragment shader write several
outputs in one pass and lets the framebuffer object route each of them to its own attachment. That
is the basis of deferred shading, of storing object ids next to color, and of any technique that
needs more than four numbers per pixel. This demo uses one framebuffer object with four color
attachments of three different kinds and one fragment shader with four `layout(location = N)`
outputs:

| Output | Type in GLSL | Attachment | Format | Holds |
|---|---|---|---|---|
| `layout(location = 0) out vec4 o_albedo` | `vec4` | `COLOR_ATTACHMENT0` | `RGBA8` | gradient on a 16x16 checkerboard |
| `layout(location = 1) out vec4 o_data` | `vec4` | `COLOR_ATTACHMENT1` | `RGBA8` | inverted gradient, alpha 0.75 |
| `layout(location = 2) out uvec4 o_ids` | `uvec4` | `COLOR_ATTACHMENT2` | `RGBA32UI` | x, y, a value with the top bits set, a downward counter |
| `layout(location = 3) out ivec4 o_signed` | `ivec4` | `COLOR_ATTACHMENT3` | `RG32I` | a negative product, a value next to `INT_MIN` |

Table 3.13 of the specification marks the 1-, 2- and 4-component 8/16/32-bit integer formats as
color-renderable (the 3-component ones are not), so integer render targets are core ES 3.0. A
`DEPTH24_STENCIL8` renderbuffer is attached as well; it is used by the `glClearBufferfi` part.

## What the image shows

The window shows the four attachments of the 128 x 128 offscreen target, one per quarter, each scaled to
fill its quarter at any window size (the quarters split the window at `width / 2` and `height / 2`):

```
 top-left:     albedo (RGBA8)       top-right:    data (RGBA8)
 bottom-left:  ids (RGBA32UI)       bottom-right: signed (RG32I)
```

The composite shader reads the attachments with `texelFetch`, because integer textures cannot be
filtered (Table 3.13) and need a `usampler2D` / `isampler2D`. The integer attachments are turned into colors
with simple formulas (see `shaders/composite.frag`), which the CPU repeats: the ids panel shows
x as red and y as green with a striped blue channel from the low byte of the counter, the signed
panel shows `x*y` as red, `x` as green (from the low byte of `INT_MIN + x`) and a constant blue
for "negative".

The scaling uses integer texel mapping only: pixel `(x, y)` of a quarter that is `w x h` pixels shows texel
`(x * 128 / w, y * 128 / h)`, computed with integer division in the shader from `ivec2(gl_FragCoord.xy)`.
There is no filtering and no float rounding, so the CPU picks exactly the same texel for every pixel and the
comparison stays exact at every size. The offscreen target stays 128 x 128 at every window size, because the
integer models are functions of the texel coordinate. The smallest supported window is 32 x 32; below that the
demo reports SKIP.

## Central calls

| Call | Role in this demo | Spec |
|---|---|---|
| `glDrawBuffers(n, bufs)` | maps fragment outputs (by `location`) to attachments | §4.2.1 |
| `glClearBufferfv / iv / uiv(COLOR, i, value)` | clears draw buffer `i` in the format of that buffer | §4.2.3 |
| `glClearBufferfi(depth, stencil)` | clears depth and stencil together | §4.2.3 |
| `glReadBuffer(COLOR_ATTACHMENTi)` | chooses the attachment `glReadPixels` reads | §4.3.1 |
| `glReadPixels(RGBA_INTEGER, INT / UNSIGNED_INT)` | reads integer attachments | §4.3.2 |
| `glGetFragDataLocation(program, name)` | asks which draw buffer an output was bound to | §3.9.2.3 |
| `glGetIntegerv(MAX_DRAW_BUFFERS, MAX_COLOR_ATTACHMENTS, DRAW_BUFFERi, READ_BUFFER)` | limits and state | §4.2.1, §4.3.1, §6.1 |

In meta-gl a color attachment is a `ColorAttachment` value. `to_framebuffer_attachment()`,
`to_draw_buffer()` and `to_read_buffer()` convert it for `glFramebufferTexture2D`, `glDrawBuffers`
and `glReadBuffer`, which also accept the non-color members (`None`, `Back`, `DepthStencil`).

## Key ideas

- **`location` is the draw buffer index.** The fragment output with `layout(location = N)` is
  written to draw buffer N; `glDrawBuffers` then says which attachment draw buffer N is. If a shader
  has more than one output, every output needs an explicit location (GLSL ES 3.00 §4.3.8.2).
- **ES 3.0 is stricter than desktop GL.** In a framebuffer object, draw buffer `i` must be
  `COLOR_ATTACHMENTi` or `NONE`; anything else (for example `{COLOR_ATTACHMENT1, COLOR_ATTACHMENT0}`)
  is `INVALID_OPERATION`. The default framebuffer takes exactly one buffer, `BACK` or `NONE`.
  Gaps are therefore made with `NONE`: `glDrawBuffers({C0, NONE, C2, NONE})` writes outputs 0 and 2 and
  throws outputs 1 and 3 away. The demo shows that attachments 1 and 3 keep their old contents.
- **`n = 0` means all `NONE`.** Depth and stencil work still happens, color is not written.
- **Initial state.** A framebuffer object starts with `DRAW_BUFFER0 = COLOR_ATTACHMENT0`, all other
  draw buffers `NONE`, and `READ_BUFFER = COLOR_ATTACHMENT0`; the default framebuffer starts with
  `BACK`. This state belongs to the framebuffer object, not to the context.
- **Never write a type the buffer cannot hold.** A float output into an integer attachment, or the
  reverse, gives undefined results (§3.9.2.3). The component *count* may differ: the `RG32I`
  attachment simply drops `z` and `w` of the `ivec4`, and reads back with `B = 0`, `A = 1` (§4.3.2.1).
- **Clear in the type of the buffer.** `glClearBufferfv` for normalized (clamped to [0,1], converted
  with equation 2.3) and float buffers, `glClearBufferiv` for signed and `glClearBufferuiv` for unsigned
  integer buffers. The `drawbuffer` argument is a draw buffer index, so a draw buffer that is `NONE`
  is not cleared (stated explicitly in ES 3.2 §15.2.3.1). `glClear` on an integer buffer is undefined.
- **Reading integer buffers.** `glReadPixels` needs `RGBA_INTEGER` with `INT` (signed) or
  `UNSIGNED_INT` (unsigned); the plain `RGBA/UNSIGNED_BYTE` pair is only valid for normalized buffers.
- **Depth and stencil values cannot be read back in ES,** so the `glClearBufferfi` values are proven
  by what passes the tests afterwards (see Validation).

## Validation (`--validate`)

81 checks. All of them are observable behavior compared with a CPU model or a defined error; none only checks that a call returns.

- **The four attachments** are read back completely (16384 pixels each) and compared with the CPU
  model of `mrt.frag`: exactly for `RGBA32UI` and `RG32I` (including values above 2^31 and next to
  `INT_MIN`), within +-1/255 for the `RGBA8` attachments (equation 2.3 rounds to nearest; an
  implementation may be off by one step). **Negative controls:** the albedo image compared with the
  model of the data image differs at every pixel, and an `RGBA32UI` model that is off by one in `w`
  differs at every pixel, so the comparison really is exact.
- **The window** shows the four panels; every pixel of each panel is compared with the formulas of
  `composite.frag` (+-1/255: the value passes through an 8-bit attachment once and through the 8-bit
  window once).
- **NONE gaps.** All attachments are set to a sentinel with `glClearBuffer*`, then `{C0, NONE, C2, NONE}`
  is drawn: attachments 0 and 2 equal the model, 1 and 3 still equal the sentinel everywhere. The
  complementary mapping `{NONE, C1, NONE, C3}` is checked the same way. A second program with a single
  output at `location = 1` is drawn with `{NONE, C1, NONE, NONE}`: only attachment 1 changes.
  `glDrawBuffers` with `n = 0` leaves all four sentinels untouched.
- **DRAW_BUFFERi and READ_BUFFER state** is read back after each mapping, in a fresh framebuffer
  object and in the default framebuffer.
- **Errors** (each followed by a state check where the spec says "no side effect"): the four invalid
  `glDrawBuffers` calls above, `n = MAX_DRAW_BUFFERS + 1` (`INVALID_VALUE`), three default
  framebuffer cases, `glReadBuffer` with `BACK` on an FBO, with `COLOR_ATTACHMENT0` on the default framebuffer and with
  `COLOR_ATTACHMENT(MAX_COLOR_ATTACHMENTS)`, `glReadPixels` with a mismatching format for integer
  and for normalized buffers, with `READ_BUFFER = NONE`, and from an attachment point without image,
  `glClearBuffer*` with a draw buffer index that is out of range or non-zero for depth/stencil.
- **Clears.** `glClearBufferuiv {0xFFFFFFFF, 0x80000000, 1, 0xDEADBEEF}` and `glClearBufferiv
  {INT_MIN, -1, ...}` read back exactly; clearing one draw buffer leaves the others alone; floats
  outside [0,1] are clamped; a `NONE` draw buffer is not cleared.
- **`glClearBufferfi`.** Four quads with window depth 0.2 or 0.3 and stencil reference 5, 5, 4 or 6 are drawn with
  `DEPTH_TEST` (`LESS`) and `STENCIL_TEST` (`EQUAL`) after clearing to depth/stencil (0.25, 5), (0.35, 6) and (0.35, 5). A quad is
  visible only if its depth is below the cleared depth and its reference equals the cleared stencil value, so each of the three
  clears must produce its own distinct pattern of visible quads: {quad 0}, {quad 3} and {quads 0, 1}.
- **`glGetFragDataLocation`** returns 0..3 for the four outputs of `mrt.frag`, 1 for the single
  output of `single.frag`, -1 for an unknown name, and `INVALID_OPERATION` with -1 for a program that
  was never linked or whose link failed.
- **Output rules.** Two outputs at locations 0 and 1, and at 2 and 0 (out of order, with a gap) build; two outputs at the same
  location and an output at `MAX_DRAW_BUFFERS` do not.

## Known driver deviations and limits of the evidence (Mesa 25.0.7: radeonsi, llvmpipe, zink)

- **Crash.** `glReadPixels` with `READ_BUFFER = NONE` generates `SIGSEGV` inside the driver when *any* attachment of the framebuffer
  object has an integer format (reproduced with `COLOR_ATTACHMENT0 = RGBA8`, `COLOR_ATTACHMENT3 = RG32I`;
  it works and returns `INVALID_OPERATION` when all attachments are normalized). The demo therefore tests the
  `READ_BUFFER = NONE` rule on a separate framebuffer object with one `RGBA8` attachment.
- **Missing output locations are accepted.** ES 3.0 §3.9.2.3 says that "if there is more than one output variable, all output
  variables must be explicitly bound to fragment colors within the shader text. Missing or conflicting binding assignments will
  cause CompileShader to fail", and GLSL ES 3.00 §4.3.8.2 says "if there is more than one output, the location must be specified for
  all outputs". Mesa compiles and links `out vec4 a; out vec4 b;` and `layout(location = 1) out vec4 a; out vec4 b;` and assigns the
  locations itself. This is not checked here. (The existing `glsl300/multiple-outputs-need-locations` shader test
  expects `build-fail` and passes only because its program has no vertex shader.)
- **Duplicate and out-of-range locations are rejected at link time,** not at compile time as §3.9.2.3 words it. The demo accepts either stage.
- **ANGLE (Vulkan backend): `glDrawBuffers` on the bound framebuffer is not applied to the next draw.** After
  `{C0, NONE, C2, NONE}` (drawn), a clear with all four buffers and `{NONE, C1, NONE, C3}`, `DRAW_BUFFERi` reads back the new
  mapping, but the draw writes outputs 0 and 2 to attachments 0 and 2, as with the previous draw's mapping. Re-binding the
  framebuffer before the draw makes the new mapping take effect, so this is a missed state update, not a reading of §4.2.1.
  The two mapping checks are XFAIL on "ANGLE".
- The error code of an illegal `glClearBuffer*` enum (`INVALID_ENUM`) cannot be provoked, because meta-gl's typed buffer enums only
  contain legal values.

## Pitfalls

- Rendering into a texture does not require the texture to be mipmap complete, but *sampling* it does. With the
  default `NEAREST_MIPMAP_LINEAR` minification filter, a one-level texture is incomplete and samples as (0, 0, 0, 1),
  and integer textures additionally need `NEAREST` or `NEAREST_MIPMAP_NEAREST` filters (§3.8.13). The demo sets
  `NEAREST` for all four attachments before the composite pass samples them.
- `glDrawBuffers` state is per framebuffer object. Binding another framebuffer does not "reset" it
  and the default framebuffer has its own state.
- Unwritten outputs are undefined, not zero. `single.frag` writes only location 1, so draw buffer 0
  must be `NONE` while it runs.
- Use `highp int` (or a `highp` output) when integers need more than 16 bits: the default int
  precision in fragment shaders is `mediump`.
