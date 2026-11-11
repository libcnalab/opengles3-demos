# test-driver-crash-readpixels-none

Tests the error rules for `glReadPixels` when there is no color buffer to read. It also isolates
a Mesa crash on one of these rules, so the crash cannot take the test process down.

Minimum version: OpenGL ES 3.0 (`glReadBuffer`, integer color attachments). The test uses no shaders.

## The rule

ES 3.0 §4.3.2.1 (Obtaining Pixels from the Framebuffer):

> ReadPixels generates an INVALID_OPERATION error if READ_BUFFER is NONE or if the GL is using a
> framebuffer object (i.e. READ_FRAMEBUFFER_BINDING is non-zero) and the read buffer selects an
> attachment that has no image attached.

§4.3.2 adds the same two conditions for `GetIntegerv(IMPLEMENTATION_COLOR_READ_FORMAT / _TYPE)`.
§4.3.1 makes `glReadBuffer(NONE)` legal for both the default framebuffer and framebuffer objects. It
also makes `COLOR_ATTACHMENTi` legal for any i below `MAX_COLOR_ATTACHMENTS`, whether or not an image
is attached there. In ES 3.0, framebuffer completeness does not depend on the read buffer (§4.4.4.2),
so a missing read image produces INVALID_OPERATION and not INVALID_FRAMEBUFFER_OPERATION.

## What runs where

**In the test process.** These calls do not crash on any tested driver:

- A framebuffer object with a single RGBA8 renderbuffer. The initial `READ_BUFFER` is `COLOR_ATTACHMENT0`,
  and a legal read returns the cleared color. This positive control shows that the identical call
  succeeds when the read-buffer state allows it.
- `glReadBuffer(NONE)`: no error, and `READ_BUFFER` reads back NONE. `glReadPixels` must then raise
  INVALID_OPERATION and leave the sentinel-filled output buffer untouched (§2.5). Both
  implementation-read queries must also fail with INVALID_OPERATION.
- `glReadBuffer(COLOR_ATTACHMENT1)` with nothing attached at COLOR1. The framebuffer must stay complete,
  `glReadPixels` and both queries must raise INVALID_OPERATION, and the output buffer must stay untouched.
- `glReadBuffer(COLOR_ATTACHMENT0)` again: the same read succeeds and returns the cleared color.
- The default framebuffer: `READ_BUFFER` is initially BACK. With NONE the read fails with
  INVALID_OPERATION. With BACK again it returns the clear color.

**In child processes.** These cases need a fresh process because Mesa may crash:

| case | framebuffer | read | expected |
|---|---|---|---|
| `control-read` | COLOR0 = RGBA8, COLOR3 = RG32I | COLOR0 RGBA/UNSIGNED_BYTE, COLOR3 RGBA_INTEGER/INT | cleared values, (R, G, 0, 1) for RG32I; exit 0 |
| `control-fail` | same | legal read, deliberately wrong expectation | exit 1 |
| `control-signal` | same | the child raises SIGSEGV itself | killed by SIGSEGV |
| `none-mixed-rgba` | COLOR0 = RGBA8, COLOR3 = RG32I | READ_BUFFER NONE, RGBA/UNSIGNED_BYTE | INVALID_OPERATION, buffer untouched; exit 0 |
| `none-mixed-integer` | same | READ_BUFFER NONE, RGBA_INTEGER/INT | same |
| `none-integer-only` | COLOR0 = RG32I | READ_BUFFER NONE, RGBA_INTEGER/INT | same |

The three controls prove that the parent can tell a passing child, a failing child and a crashing
child apart. Without them, a broken harness could report every case as "exited normally".

## How the child process is made

The test re-executes its own binary through `posix_spawn("/proc/self/exe", ...)`. The first
argument is a hidden one, `--readpixels-none-child <case>`. `main()` recognises it before
`demo::run` parses options, then runs a separate `demo::Demo` (`ChildProbe`) with `--headless --validate`.
The child is a complete new process. It creates its own EGL display, pbuffer and context and
initialises meta-gl, exactly as every demo does, so `common/` needs no change. The child inherits the
environment (`EGL_PLATFORM`, `GALLIUM_DRIVER`, `LIBGL_ALWAYS_SOFTWARE`, `MESA_GLES_VERSION_OVERRIDE`),
so it runs on the parent's driver and API version. Its stdout and stderr go through a pipe. The
parent echoes them with a `| ` prefix and classifies the child's ending as an exit status, a signal,
or a timeout after 30 s (the child is then killed).

A plain `fork()` without exec was tried and rejected. A GL context is not specified to survive
`fork()`, and only the forking thread exists in the child: the driver's worker threads (glthread,
the llvmpipe rasterizer, ...) do not. On Mesa 25.0.7 a forked child that used the inherited context
got an answer from `glGetError` but then hung in the first `glClear`/`glReadPixels` on radeonsi,
llvmpipe and zink. Creating a second EGL display in a forked child would still start from the
parent's driver state (threads, DRM file descriptors), and the test would have to link EGL itself.

The child prints a marker line just before and just after the `glReadPixels` call. Only a death by
signal between these two markers counts as the known crash. A crash during setup, a wrong error
code or a timeout fails on every driver.

Two environment details matter for the crash:

- In the child, `ASAN_OPTIONS` gets `handle_segv=0:handle_sigbus=0:handle_abort=0`. Otherwise a crash
  in a sanitizer build (`--preset sanitize`) would become exit status 1, which cannot be told apart from FAIL.
- The parent lowers the soft `RLIMIT_CORE` to 0 around the spawn, so that the expected crash does not
  write a core dump.

## Known driver deviation

On Mesa 25.0.7 (radeonsi, llvmpipe and zink, at both es-max and es-min), all three `none-*` children die with
SIGSEGV inside `glReadPixels` instead of raising INVALID_OPERATION. This includes `none-integer-only`,
so the crash does not need a normalized attachment next to the integer one. The test records this
through `expect_or_known_deviation(..., "Mesa", ...)` and prints the signal. Any other driver gets the
strict check, and if Mesa is fixed the check passes and is labelled XPASS. On a framebuffer whose
attachments are all normalized, Mesa raises the error correctly, which the parent checks in-process.

## Run

```
./build/debug/bin/test-driver-crash-readpixels-none --headless --validate
ctest --preset debug -R test-driver-crash-readpixels-none
```

Linux only (`posix_spawn`, `/proc/self/exe`, `waitpid`). The default framebuffer is only cleared, and
the capture is a uniform (64, 128, 191) blue.

## Pitfalls

- Do not "fix" the crash by testing READ_BUFFER = NONE only on normalized framebuffers. That is
  what `es30-multiple-render-targets` does, and it is why this separate test exists.
- Each child is a new process with a new context. Objects of the parent do not exist there, so every
  child case builds its own framebuffer.
- `glClearBufferiv(COLOR, i, ...)` addresses draw buffer i, not attachment i. The mixed framebuffer
  therefore sets the draw buffers to `{COLOR0, NONE, NONE, COLOR3}` before clearing draw buffer 3.
