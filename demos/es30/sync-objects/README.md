# es30-sync-objects

**Minimum:** OpenGL ES 3.0 · **Shaders:** GLSL ES 3.00 · **Run:** `es30-sync-objects [--headless --validate]`

## Purpose

The GL command stream is asynchronous: when `glDrawArrays` returns, the GPU may not even have started. That is what makes
it fast, and it is also why the CPU needs a way to ask "has the GPU reached this point yet?". A **fence sync object**
answers that. `glFenceSync` puts a marker into the command stream; when everything before the marker has completed, the
object becomes *signaled*. The CPU can poll it, block on it (`glClientWaitSync`), or tell the *GPU* to wait for it
(`glWaitSync`) without stalling the CPU.

The demo uses the pattern that matters most in practice: a **ring of two vertex buffers that the CPU rewrites without
synchronization**. `MAP_UNSYNCHRONIZED_BIT` tells the GL not to wait for pending draws that read the buffer; the application
must do that waiting itself, and the fence is how. It then examines the sync API in isolation: properties, the legal results
of every wait, `glFlush` versus `glFinish`, an asynchronous readback guarded by a fence, object lifetime and the errors.

## What the image shows

A 4 × 4 grid of cells that partitions the whole surface exactly (column *i* spans `[i·w/4, (i+1)·w/4)`, so at sizes that are
not a multiple of 4 neighbouring cells differ by one pixel), reading order. Cell *f* is a rectangle whose position, size and color
are a function of *f* alone; the rectangle is designed on a 64 × 64 grid and scaled to the cell with integer division, so its edges
stay on whole pixels at every size (a different color for each of the 16 cells). The rectangles are drawn by 16 updates that alternate between two
buffers; every cell was drawn from the buffer contents written for *that* update:

```
for update f in 0..15:                      slot = f % 2
    if slot has a fence:                    glClientWaitSync(fence, SYNC_FLUSH_COMMANDS_BIT, 50 ms)  (repeated while TIMEOUT_EXPIRED)
                                            glDeleteSync(fence)
    glMapBufferRange(slot, MAP_WRITE_BIT | MAP_UNSYNCHRONIZED_BIT)  ->  write 4 vertices  ->  glUnmapBuffer
    glDrawArrays(TRIANGLE_STRIP, 0, 4)      // into cell f
    fence = glFenceSync(SYNC_GPU_COMMANDS_COMPLETE, 0)
```

The rectangle corners are integer pixel positions, so their NDC coordinates are small dyadic fractions: the edges lie exactly
between pixel centers and the coverage is unambiguous; the comparison is pixel-exact (colors ±1).

## Central calls

| Call | Role |
|---|---|
| `glFenceSync(SYNC_GPU_COMMANDS_COMPLETE, 0)` | insert a fence; returns the sync name (a pointer-sized `GLsync`, 0 on failure) |
| `glClientWaitSync(sync, flags, timeout_ns)` | block up to `timeout` nanoseconds; `ALREADY_SIGNALED`, `CONDITION_SATISFIED`, `TIMEOUT_EXPIRED` or `WAIT_FAILED`; `SYNC_FLUSH_COMMANDS_BIT` flushes first |
| `glWaitSync(sync, 0, TIMEOUT_IGNORED)` | make the *server* wait; the client continues at once |
| `glGetSynciv(sync, OBJECT_TYPE \| SYNC_CONDITION \| SYNC_STATUS \| SYNC_FLAGS, ...)` | properties and the signaled/unsignaled state |
| `glIsSync`, `glDeleteSync` | lifetime; the name is invalid immediately after deletion |
| `glFlush`, `glFinish` | submit the commands in finite time / wait until they are complete |
| `glGetInteger64v(MAX_SERVER_WAIT_TIMEOUT)` | how long a `glWaitSync` may block the server |

## Spec

- ES 3.0 §5.1 Flush and Finish, **§5.2** Sync Objects and Fences, **§5.2.1** Waiting for Sync Objects, §5.2.2 Signalling,
  **§6.1.8** Sync Object Queries, Table 5.1 (initial properties), Table 6.29 (`MAX_SERVER_WAIT_TIMEOUT`), §6.1.2 (conversion
  of 64-bit state), §2.10.3 `MapBufferRange` (`MAP_UNSYNCHRONIZED_BIT`), §2.5 (errors).

The rules, shortened:

- A fence is created UNSIGNALED. `SYNC_GPU_COMMANDS_COMPLETE` is satisfied when the fence command and *all preceding commands*
  have completed; then the status becomes SIGNALED and never changes back. So a later fence is never signaled before an earlier one.
- `glClientWaitSync`: `ALREADY_SIGNALED` is returned "always ... if sync was signaled, even if the value of timeout is zero".
  With timeout 0 it "does not block, but simply tests the current state" and `TIMEOUT_EXPIRED` is returned if it is not
  signaled. With a longer timeout the result may also be `CONDITION_SATISFIED` (signaled while waiting) or `TIMEOUT_EXPIRED`.
  `WAIT_FAILED` accompanies a GL error. If the fence will never be flushed to the GPU the call "may hang forever"; with
  `SYNC_FLUSH_COMMANDS_BIT` the equivalent of `glFlush` happens first.
- `glWaitSync`: `timeout` must be exactly `TIMEOUT_IGNORED` (2^64 - 1) and `flags` must be 0, otherwise `INVALID_VALUE`; the server
  waits no longer than `MAX_SERVER_WAIT_TIMEOUT`.
- `glGetSynciv`: `OBJECT_TYPE` is always `SYNC_FENCE`, `SYNC_CONDITION` is the condition, `SYNC_FLAGS` the flags (0), `SYNC_STATUS`
  `SIGNALED` or `UNSIGNALED`. At most `bufSize` integers are written and their number is returned in `length`; on error nothing is written.
- Names: `glDeleteSync(0)` is ignored; a deleted or never valid name is "not the name of a sync object" and gives `INVALID_VALUE` in
  every command that takes one, except `glIsSync`, which returns FALSE. Deleting an unsignaled fence is legal.
- `MAP_UNSYNCHRONIZED_BIT` may be combined with `MAP_WRITE_BIT` but **not** with `MAP_READ_BIT` (`INVALID_OPERATION`, §2.10.3): in ES 3.0
  a CPU read of GPU-written data cannot bypass the GL's own synchronization, which is why the readback pattern below maps
  *after* the fence with a plain read mapping.

## Shader concepts

`ring.vert/frag` draw a flat-colored rectangle from two vertex attributes. `heavy.vert` builds a triangle that covers the viewport from
`gl_VertexID` alone; `heavy.frag` runs a loop of `sin()` calls whose sum is multiplied by the uniform `u_zero` (0.0 at run time, so the
compiler cannot fold it away) and added to `u_color`: the output is exactly `u_color`, but the draw takes long enough that the fences
behind it are usually not signaled yet.

## How it is validated

- **Image.** Every pixel of all 16 cells against the model (at 256 × 256: 12088 covered, 53448 background pixels; at any size
  covered + background is exactly width × height; exact up to the ±1 color
  conversion). **Negative control:** every cell is also compared with the model of update *f + 2*, the other generation of the *same
  ring slot*; all 16 must disagree, so a cell that showed stale slot data would be caught.
- **Ring protocol.** 14 waits (updates 2..15) all end `ALREADY_SIGNALED` or `CONDITION_SATISFIED`, never `WAIT_FAILED`, and the fence
  is signaled by `glGetSynciv` afterwards; all `glMapBufferRange` / `glUnmapBuffer` calls succeed; after `glFinish` both outstanding
  fences are SIGNALED. The mix of results (and of `TIMEOUT_EXPIRED` slices) is printed as a note, not asserted: it depends on timing
  (in the reference runs radeonsi gave about half `ALREADY_SIGNALED` and half `CONDITION_SATISFIED`, llvmpipe all `CONDITION_SATISFIED`).
- **Properties.** `OBJECT_TYPE`, `SYNC_CONDITION`, `SYNC_FLAGS`; `SYNC_STATUS` is UNSIGNALED or SIGNALED right after creation (the GPU
  may be quicker than the query: Table 5.1 says UNSIGNALED, but nothing can *require* an unsignaled observation), `bufSize` 2 and 0,
  a null `length`.
- **Wait results.** Timeout 0 without and with the flush bit: only `ALREADY_SIGNALED` or `TIMEOUT_EXPIRED` are legal (the demo
  prints which one it saw; in the reference runs it was `TIMEOUT_EXPIRED` on all three drivers because the heavy draw was still
  running, but that is never required). A wait loop of 50 ms slices with the flush bit ends signaled, with only legal results on the way. Once signaled,
  timeout 0, a 1 s timeout and the largest possible timeout all return `ALREADY_SIGNALED` at once.
- **glFlush.** After `glFlush`, `glGetSynciv` is polled (no wait call): the fence becomes SIGNALED within a 10 s bound ("must complete
  in finite time") and stays so. **glFinish.** The fence created just before `glFinish` is SIGNALED on the very first query and
  `glClientWaitSync(0)` says `ALREADY_SIGNALED`: `Finish` completes "all previous commands", the fence command among them.
  **Ordering.** When a second fence (after a heavy draw) is signaled, the first is too; 32 fences created back to back are distinct,
  all signaled when the last is.
- **Asynchronous readback.** `glReadPixels` into a pixel pack buffer pre-filled with `0xA5` (negative control: the comparison fails on
  all 262144 bytes before the read), `glFenceSync`, `glFlush`, then `glClientWaitSync(timeout 0)` polled until `ALREADY_SIGNALED`
  (only `TIMEOUT_EXPIRED` allowed before that), then the buffer is mapped and every byte equals the heavy draw's color. This is the
  "poll one frame later instead of stalling" pattern.
- **glWaitSync.** `MAX_SERVER_WAIT_TIMEOUT` is non-negative and the 32-bit query returns it clamped; `glWaitSync(fence, 0, TIMEOUT_IGNORED)`
  raises no error and the draw queued behind it still executes (center pixel compared); on an already signaled fence it is legal.
- **Lifetime and errors.** `glIsSync` TRUE for a new fence, FALSE for 0 and immediately after `glDeleteSync`; deleting a fence that is
  probably unsignaled is legal; `glDeleteSync(0)` is silent. For the name 0 *and* for a deleted name: `glClientWaitSync` returns
  `WAIT_FAILED` + `INVALID_VALUE`, `glWaitSync` and `glGetSynciv` give `INVALID_VALUE` (and `glGetSynciv` writes neither `values`
  nor `length`), a second `glDeleteSync` of a deleted name gives `INVALID_VALUE`. `glWaitSync` with timeout 0, 1 s or 2^64 - 2 gives
  `INVALID_VALUE`; the same call with `TIMEOUT_IGNORED` (the positive control) does not.
- **Not exercised (meta-gl).** `glFenceSync` with a non-zero `flags` or another `condition`, `glClientWaitSync` with flag bits other than
  `SYNC_FLUSH_COMMANDS_BIT`, `glWaitSync` with non-zero `flags` and `glGetSynciv` with an unknown `pname` are defined errors, but meta-gl's
  `SyncFlag` (only `None`), `SyncCondition` (only `GpuCommandsComplete`), `SyncFlushMask` (`None`, `Commands`) and `SyncParameter`
  (four values) have no value that triggers them; they would need raw casts. The demo prints a note instead of working around it.
- **Driver deviation (XFAIL, strict elsewhere).** `glGetSynciv` with `bufSize` 0 writes nothing but reports `length` 1 on Mesa; the spec
  says "the actual number of integers replaced is returned in length", i.e. 0.

## Run

```
./build/debug/bin/es30-sync-objects --headless --validate --capture sync-objects.png
```

## Pitfalls

- A fence that is never flushed may never be signaled. `glClientWaitSync` without `SYNC_FLUSH_COMMANDS_BIT` can wait for nothing;
  when *polling* with timeout 0 the application must call `glFlush` itself once after creating the fence.
- `TIMEOUT_EXPIRED` is a normal result, not an error; `WAIT_FAILED` is the error. Never assume "timeout expired" means "the GPU is hung".
- Timeouts are in **nanoseconds** and rounded up to the implementation's accuracy; a timeout of 0 never blocks.
- `glWaitSync` does not block the CPU, only the GPU queue, and in a single context its effect is invisible: commands are in order anyway.
  It matters for work submitted from *another* context (ES 3.0 Appendix D.2).
- `MAP_UNSYNCHRONIZED_BIT` moves the responsibility for correctness from the GL to you. Without the fence wait the rewrite can race with the
  draw that reads the old contents. It cannot be used together with `MAP_READ_BIT`.
- Delete fences you have waited on; the name becomes invalid at once, so do not reuse a stale `GLsync` value (the underlying memory may be
  recycled by a later `glFenceSync`).
- Creating a fence is not free on every driver, and `glFinish` is a pipeline stall: prefer one fence per frame or per resource over `glFinish`.
- Laying the cells out as `w / 4` × `h / 4` leaves a strip of up to three pixels uncovered at sizes that are not a multiple of 4; the
  demo partitions the surface with `i·w/4 .. (i+1)·w/4` instead. Below 64 × 64 (cells under 16 pixels) it reports SKIP.
