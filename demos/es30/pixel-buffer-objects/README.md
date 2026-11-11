# es30-pixel-buffer-objects

**API introduced in:** OpenGL ES 3.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 (one trivial fragment shader)

## Purpose

In ES 2.0 every pixel transfer goes through client memory: `glTexImage2D` copies from a pointer, `glReadPixels` copies
to a pointer, and the driver usually has to finish all pending work before `glReadPixels` can return. ES 3.0 lets the
*pointer* of these commands refer to a **buffer object** instead. A buffer bound to `PIXEL_UNPACK_BUFFER` is the source
of texture uploads; a buffer bound to `PIXEL_PACK_BUFFER` is the destination of `glReadPixels`. The data stays in
GPU-friendly memory, an upload can come from a buffer that was filled earlier (or by the GPU itself), and a read-back can
be queued and collected later. The rules are the old ones with one twist — and this demo checks the twist, the old rules
as they apply to buffers, and the errors.

## Central calls

| call | what the demo shows |
|---|---|
| `glBindBuffer(PIXEL_UNPACK_BUFFER, b)` + `glTexImage2D(…, offset)` | the last argument is a **byte offset** into `b`. `0` is offset 0, *not* "no data": with a buffer bound the texture gets the buffer's first bytes (§3.8.3) |
| `glTexSubImage2D`, `glTexImage3D` from a buffer | same rule; 3D adds `UNPACK_IMAGE_HEIGHT` and `UNPACK_SKIP_IMAGES` |
| `glPixelStorei(UNPACK_ROW_LENGTH / SKIP_PIXELS / SKIP_ROWS / ALIGNMENT, n)` | select a sub-rectangle of a *larger image in the buffer*; alignment pads rows (§3.7.2.1, eq. 3.12, 3.13) |
| `glBindBuffer(PIXEL_PACK_BUFFER, b)` + `glReadPixels(…, offset)` | the pixels are written into `b` at the byte offset, laid out by the `PACK_*` modes |
| `glMapBufferRange(PIXEL_PACK_BUFFER, …)`, `glUnmapBuffer` | read the data back; the pointer is returned "once all pending operations on that buffer have completed" (§2.10.3) |
| `glFenceSync`, `glClientWaitSync`, `glDeleteSync` | the asynchronous pattern: do not map until the read-back has finished |
| `glGetIntegerv(PIXEL_PACK_BUFFER_BINDING, PIXEL_UNPACK_BUFFER_BINDING, UNPACK_ROW_LENGTH, …)` | the binding and pixel-store state (`GetParameter` enumerators added to meta-gl, gap G1) |

## The image

The 256×256 default framebuffer shows seven textures in cells of 32×32 pixels along the bottom row (cell 0 at the left,
row 0 at the bottom). Small textures are magnified so each texel is several pixels wide; every pixel is compared.

| cell | content |
|---|---|
| 0 | the 32×32 chart (R = 8x, G = 8y, B = checker, A = 128 + x + y), `glTexImage2D` with data = offset 0 |
| 1 | a 20×13 sub-rectangle at column 5, row 3 of the chart in the buffer (`ROW_LENGTH 32, SKIP_PIXELS 5, SKIP_ROWS 3`) |
| 2 | a 20×13 sub-rectangle (`SKIP_PIXELS 2, SKIP_ROWS 8`) of a **second**, colour-inverted chart that starts at byte offset 4160 of the same buffer |
| 3 | `glTexSubImage2D`: a 14×10 patch from the chart (`SKIP 11, 2`) written at texel (6, 9) of a flat-coloured texture |
| 4 | RGB8, 5×4 (×6) taken from a 7×6 image (`ROW_LENGTH 7, SKIP 1, 1`) whose rows are padded to 24 bytes by `UNPACK_ALIGNMENT 4`; data offset 5 |
| 5 | RGBA4, 5×4 (×6), type `UNSIGNED_SHORT_4_4_4_4`, rows of 10 bytes padded to 12; data offset 6 |
| 6 | a GPU-only round trip: `glReadPixels` of cell 0 into a pack buffer, then the **same buffer** as the unpack buffer of a `glTexImage2D` |

## Key ideas

- **The pointer is an offset (§3.7.1, §3.7.2.1, §4.3.2).** With a buffer bound, "data is an offset into the pixel unpack
  buffer and the pixels are unpacked from the buffer relative to this offset". Cells 2, 4 and 5 start at offsets 4160, 5
  and 6, cell 0 at 0. The second chart is inverted so that an implementation that ignored the offset would show the first
  chart; cell 2 is compared with the inverted one.
- **Sub-rectangles of a larger image.** The first group is read at
  `offset + SKIP_PIXELS · n·s + SKIP_ROWS · k`, and each next row starts `k` bytes later, with `k = n·l` (rows of
  `l = ROW_LENGTH or width` groups of `n` elements of `s` bytes) when `s ≥ ALIGNMENT`, and
  `a/s · ⌈s·n·l / a⌉` elements otherwise (equations 3.12–3.13). The demo builds the buffers with a CPU function that
  implements these formulas independently of the GL, so GL and CPU have to agree on every row stride — 24 bytes for
  7 RGB groups, 12 for 5 RGBA4 groups, 24 for 5 RGBA groups under `PACK_ALIGNMENT 8` (equation 3.13 gives `⌈20/8⌉·8`).
- **Alignment of the offset to the type.** The offset must be a multiple of the size of the data type (`UNSIGNED_BYTE` 1,
  `UNSIGNED_SHORT_4_4_4_4` 2, `UNSIGNED_INT` 4); otherwise `INVALID_OPERATION` (§3.7.2.1, §4.3.2.4). The checks use
  offset 3 vs 4 for 4_4_4_4 uploads and offset 2 vs 4 for a 4-byte `RGBA_INTEGER`/`UNSIGNED_INT` pack into an `RGBA8UI`
  framebuffer (the pair ES 3.0 requires for unsigned-integer buffers).
- **Range checks.** If the transfer "would access memory beyond the size of the buffer", the error is
  `INVALID_OPERATION`. The demo finds the boundary exactly: a 4096-byte image from a 4096-byte buffer works, from a
  4095-byte buffer, or from offset 1, it does not; with `ROW_LENGTH 32` and `SKIP_ROWS` the footprint is
  `offset + (SKIP_ROWS + h − 1) · k + (SKIP_PIXELS + w) · groupsize`, which fits at `SKIP_ROWS 19` (byte 8208) and does not at
  20 (byte 8336) in an 8256-byte buffer. A rejected `glTexSubImage2D` or `glReadPixels` leaves the texture / the buffer
  contents alone.
- **3D uploads.** `UNPACK_IMAGE_HEIGHT` sets the number of rows per image in the buffer, `UNPACK_SKIP_IMAGES` skips whole
  images (§3.8.3); a 4×3×2 sub-volume of a 6×5×3 volume is uploaded and every texel of both layers is checked.
  `glTexImage2D` is "TexImage3D with depth 1, except that `UNPACK_SKIP_IMAGES` is ignored", also tested.
- **Reading back.** `glReadPixels` into a pack buffer writes pixels at the offset with the `PACK_*` modes
  (`PACK_ROW_LENGTH 24, SKIP_PIXELS 3, SKIP_ROWS 2` at offset 36 for a 10×7 rectangle, `PACK_ALIGNMENT 8` for 5×3). The buffer
  is mapped with `glMapBufferRange(…, MAP_READ_BIT)` and compared byte for byte with a CPU placement of a client-memory read
  of the same pixels. Only bytes the specification places pixels in are compared exactly; bytes before the offset and after
  the last pixel group must be unchanged (our reading of §4.3.2.4: the footprint of the transfer is the only memory the
  command accesses);
  padding bytes inside the footprint are not specified and are only reported (`note:`).
- **Asynchronous pattern.** A synchronous `glReadPixels` into client memory stalls the pipeline until the pixels exist.
  Into a pack buffer it can return immediately. The classic pattern keeps two buffers: in frame *k* the read-back goes to
  buffer *k mod 2*, a fence marks it (`glFenceSync`), and frame *k* + 1 maps the buffer of frame *k*. The demo runs this
  for six frames, each cleared to a different colour, waits with `glClientWaitSync(… SYNC_FLUSH_COMMANDS_BIT, 1 s)` (which must
  return `ALREADY_SIGNALED` or `CONDITION_SATISFIED`, the two outcomes the specification defines for a signalled fence)
  and compares all 256 pixels of each frame. Without the fence the pattern is still correct, because `glMapBufferRange`
  blocks until the pending read is done; the demo checks that variant too.
- **Bindings.** The pack and unpack targets have separate bindings (`PIXEL_PACK_BUFFER_BINDING`,
  `PIXEL_UNPACK_BUFFER_BINDING`); one buffer object may be bound to both at once (cell 6). Deleting a bound buffer resets
  the binding to 0 in the current context (§2.10.1). With 0 bound, `glBufferData` and `glGetBufferParameteriv` on the target
  generate `INVALID_OPERATION`.

## Validation

76 checks per run.

- **Textures.** Cells 0–3 and 6 are RGBA8, so each texture is attached to a framebuffer and read with `glReadPixels`
  and compared **exactly** with the model (no arithmetic is specified for moving a stored 8-bit value). All seven cells are
  also compared as drawn, pixel for pixel, with ±1: the shader output conversion allows one step (§2.1.6.2). RGB8 and
  RGBA4 are only compared as drawn (RGBA4 is `17·c`, since `c/15·255`).
- **Pack buffers.** Four layouts (whole 32×32 at offset 0; a 10×8 rectangle at offset 100; row length / skips / offset; alignment 8)
  plus the integer read: the mapped bytes of every pixel group equal the CPU placement of a client-memory read of the same
  pixels, and the sentinel `0xEE` is still there everywhere outside the footprint.
- **Errors.** `INVALID_OPERATION` for each over-long unpack and pack, each misaligned offset of a 2- and a 4-byte type,
  `glBufferData`/`glGetBufferParameteriv` with 0 bound, mapping an already mapped buffer, `glBufferSubData` into a mapped
  buffer, unmapping an unmapped buffer; `INVALID_VALUE` for negative row length / skip, alignment 3 and 16
  (§3.7.1), and the state unchanged afterwards. The over-long transfers have a matching success one byte (or one row) away (4095/4096, 255/256, offset 1/0,
  SKIP_ROWS 20/19, offset 3/4 and 2/4), so those checks cannot pass by always failing.
- **State.** The eight pixel-store values and the two buffer bindings read back with `glGetIntegerv`: initial (0, 4), the values
  just set, and independence of the two targets.
- **Negative controls.** (1) The 20×13 texture of cell 1 does not equal the top-left corner of the chart (it would, if
  `ROW_LENGTH`/`SKIP_*` were ignored): all 260 texels differ. (2) Cell 2 does not equal the same rectangle of the first
  chart (it would, if the offset 4160 were ignored). (3) The `PACK_ROW_LENGTH`/`SKIP` result does not equal a tightly packed
  placement: hundreds of bytes differ. (4) Failure cases such as the 4095-byte buffer have passing siblings. During
  development, forgetting the data store of the round-trip buffer (its size was 0) turned cell 6 into `INVALID_OPERATION`
  and a failing run, and forgetting the NEAREST filter of that texture turned the cell black (an incomplete texture):
  both were caught by these checks.

## Not tested, and why

- **A mapped buffer used as the source of `glTexImage2D`.** §2.10.3.2 only says such commands "*may* generate an
  INVALID_OPERATION error if all or part of the buffer object is mapped"; "only commands which explicitly describe this error
  are required to do so", and TexImage* does not. Otherwise the result is undefined and may "result in GL interruption or
  termination". The demo therefore never makes the call. The errors that *are* explicitly specified for mapped buffers
  (a second `glMapBufferRange`, `glBufferSubData` into a mapped range, `glUnmapBuffer` of an unmapped buffer) are tested.
- **Padding bytes inside a pack footprint**: not specified.
- **`MAP_UNSYNCHRONIZED_BIT` and the other mapping flags**: that is the buffer-mapping demo's subject.

## Pitfalls

- With a buffer bound, a `NULL`/`0` data argument is offset 0 and *does* read buffer contents. To allocate a texture
  without data, unbind the unpack buffer first.
- A pack buffer must have a data store (`glBufferData(PIXEL_PACK_BUFFER, size, NULL, usage)`) before `glReadPixels`;
  a new buffer has size 0, so any read is "beyond the end".
- The `PACK_*` and `UNPACK_*` modes apply to client memory and buffers alike and stay set until you reset them; the demo
  resets them after every use.
- `UNPACK_SKIP_IMAGES` and `UNPACK_IMAGE_HEIGHT` do nothing for 2D uploads.
- Mapping a pack buffer immediately after `glReadPixels` is correct but makes the application wait. Map the previous
  frame's buffer, and wait on a fence only if you must.
- The offset of a transfer must be a multiple of the element size; unaligned offsets are an error, not a silent copy.
