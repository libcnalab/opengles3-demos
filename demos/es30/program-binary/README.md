# es30-program-binary

**API introduced in:** OpenGL ES 3.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

Compiling and linking GLSL at start-up costs time. ES 3.0 lets an application save the result of a
successful link with `glGetProgramBinary` and load it in a later run with `glProgramBinary`,
skipping compilation. The binary is opaque and driver-specific. A load can legitimately fail
(after a driver update, for example), so the GLSL source stays the portable format and the
application must be able to fall back to it.

This demo saves the binary of a linked program and loads it into a **fresh** program object. The
original program renders the left half of the window and the reloaded program renders the right
half. It then checks every observable consequence that ES 3.0 §2.12.4 (ES 3.2 §7.5) defines.

## Central calls

```cpp
glProgramParameteri(prog, PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);  // before glLinkProgram
glLinkProgram(prog);
glGetIntegerv(NUM_PROGRAM_BINARY_FORMATS, &n);                         // may be 0
glGetProgramiv(prog, PROGRAM_BINARY_LENGTH, &len);
glGetProgramBinary(prog, len, &written, &format, bytes);
glProgramBinary(fresh, format, bytes, written);                        // an implicit "link"
glGetProgramiv(fresh, LINK_STATUS, &ok);                               // FALSE -> fall back to source
```

## Spec sections

- ES 3.0 §2.12.4 Program Binaries (ES 3.2 §7.5, whose explicit error lists are used where ES 3.0 relies
  on the general rules).
- ES 3.0 §2.12.3 `ProgramParameteri` and `UseProgram`; ES 3.2 §7.3 errors of `ProgramParameteri`.
- ES 3.0 §2.5 GL Errors: an enum that is not allowable gives `INVALID_ENUM`, a negative `sizei` gives
  `INVALID_VALUE`, and an erroring command has no effect on GL state or on values behind pointer arguments.
- ES 3.0 Table 6.19: `PROGRAM_BINARY_LENGTH` (initially 0) and `PROGRAM_BINARY_RETRIEVABLE_HINT` (initially FALSE).
- ES 3.0 §2.1.6.2: float to unsigned-normalized conversion (the pixel tolerance).
- GLSL ES 3.00 §4.1.3: unsigned integer multiplication wraps to the low 32 bits.

## The image

A Truchet pattern. Each square cell holds two quarter-circle arcs around opposite corners, and a
per-cell integer hash chooses one of the two orientations, so the arcs join into long meandering
curves. Arc colours run in diagonal bands of teal, sand, apricot and coral over a two-tone night-blue
background. The cell size follows the surface (about a seventh of the shorter side of a half), so the
picture fills the window at any size. The two halves are identical and are separated by a thin dark
gap: **left = program linked from source, right = program loaded from the binary**.

The shader uses only integer arithmetic on `gl_FragCoord` and the uniforms `u_origin`, `u_cell`,
`u_seed`, `u_gray`, plus the vertex uniform `u_scale`. The C++ code reproduces every pixel exactly.

If the implementation reports `NUM_PROGRAM_BINARY_FORMATS = 0`, which the spec allows, no binary
can be retrieved. The right half then shows the original program with `u_gray = true`, a grey
version of the pattern, and the demo prints why.

## What is verified

With **at least one binary format** (all Mesa drivers in the matrix report one format, `0x875F`):

| Check | Spec |
|---|---|
| Hint initially FALSE; TRUE after `glProgramParameteri(TRUE)` and a link; value 2 → `INVALID_VALUE` (ES ≥ 3.1) and the hint is unchanged | Table 6.19, ES 3.2 §7.3 |
| `glGetProgramBinary`: `length == PROGRAM_BINARY_LENGTH`, the format is in `PROGRAM_BINARY_FORMATS` | §2.12.4 |
| `glProgramBinary` into a fresh program: no error, `LINK_STATUS = TRUE` | §2.12.4 "must be successful" |
| Directly after the load every uniform reads 0 / FALSE. Negative control: the same predicate is false on the original program. | §2.12.4 "reset all uniform variables" |
| `a_pos` is at attribute 5, the binding in effect at the original link, although the fresh program had a decoy `glBindAttribLocation(2)` | §2.12.4 "input ... assignments ... are restored" |
| Active uniforms and attributes (name, size, type) and the `o_color` output location match the original | §2.12.4 |
| Both halves match the CPU reference (±1), and the right half is **bit-identical** to the left half | §2.1.6.2, §2.12.4 |
| Negative control: a different `u_seed` on the reloaded program is caught by both comparisons | — |
| Seven bad binaries (byte inverted at the start, middle or end; truncated by 1 byte or to half; 16 trailing bytes with length + 16; a single flipped bit), each loaded over a successfully loaded program **in a child process** (see below): no error, `LINK_STATUS = FALSE`, `PROGRAM_BINARY_LENGTH = 0`, and the process survives | §2.12.4 "will fail", "no error is generated", "information about a previous link ... is lost" (known Mesa and ANGLE deviations, see below) |
| In-process, a binary with an inverted first byte (a header mutation, rejected cleanly by every tested driver) fails to load; afterwards `glUseProgram` and `glGetProgramBinary` → `INVALID_OPERATION` | §2.12.3, §2.12.4 |
| A `binaryFormat` that is not listed → `INVALID_ENUM`; `length = -1` → `INVALID_VALUE`. Both calls must leave the program linked. | ES 3.2 §7.5, ES 3.0 §2.5 |
| Fallback: after a failed load of a stale binary (its header altered, as after a driver update) the program still has its two shaders attached and its attribute binding, relinks from source with `glLinkProgram`, and renders the reference image | §2.12.4 |
| `glGetProgramBinary` with `bufSize = length − 1` and with `bufSize = 0` → `INVALID_OPERATION`, and `binaryFormat` / `length` are not written | §2.12.4, §2.5 |

With **zero formats**, every `binaryFormat` value passed to `glProgramBinary` must give
`INVALID_ENUM`, because no value is in the empty list. On ES 3.2, `glGetProgramBinary` on a linked
program must give `INVALID_OPERATION`. Both halves are compared with the CPU reference, with a
negative control. No driver in the matrix reports zero formats. Mesa reports one format whenever the
gallium screen has a disk-cache object, and it keeps that object even with
`MESA_SHADER_CACHE_DISABLE=true`, so this path compiles but has not been run.

**In both cases:** a never-linked program has `PROGRAM_BINARY_LENGTH = 0`, and `glGetProgramBinary`
on it gives `INVALID_OPERATION`. The same call on a program whose link failed (vertex shader only)
also gives `INVALID_OPERATION`.

### Bad binaries in child processes

A load that must fail cleanly crashes one driver (SIGILL inside `glProgramBinary` on ANGLE), so each of the seven
mutations runs in a fresh child process on every driver. The executable re-executes itself through `/proc/self/exe` with
a hidden first argument (the pattern of `tests/negative/driver-crash-readpixels-none`). The child creates its own
context, links the pattern program from source, retrieves its binary, loads the valid binary and then the mutated one
between two printed markers, prints the error, `LINK_STATUS` and `PROGRAM_BINARY_LENGTH`, and exits with PASS or FAIL. The
parent echoes the child's output and classifies its end (exit status, signal, 30 s timeout). The documented deviations are
a death between the markers and a mutated binary that loads with `LINK_STATUS = TRUE`. Anything else, such as a wrong error,
a failing valid load or a crash during setup, is a plain FAIL on every driver.

## Known Mesa deviations (kept strict, reported as XFAIL)

All three come from Mesa's shared code (`src/mesa/main/shaderapi.c`, `program_binary.c`), so they
show up on radeonsi, llvmpipe and zink alike:

1. **An over-long `length` loads.** Mesa's header check rejects only binaries that are too short.
   The CRC covers the payload size stored in the header, so trailing bytes are ignored. §2.12.4
   requires `length` to equal the binary's length and says the load "will fail" otherwise.
2. **A `glProgramBinary` call that raises an error unlinks the program.** Mesa clears the program's
   link data *before* it checks `length < 0` and `binaryFormat`. §2.5 says the command generating
   the error "is ignored so that it has no effect on GL state".
3. **A `glGetProgramBinary` call that raises an error writes `*length = 0`.** §2.5 says "no change is
   made to these values". `binaryFormat` is left alone, as required.

## Known ANGLE deviations (kept strict, reported as XFAIL)

ANGLE 2.1.17841 (Vulkan backend, one format `0x93A6`):

1. **Bad binaries load.** A byte inverted in the middle or at the end, a single flipped bit, a length one byte short and a
   length 16 bytes too long all load with `LINK_STATUS = TRUE` (`PROGRAM_BINARY_LENGTH` 13395 instead of the 13195 bytes
   retrieved). Only the inverted first byte (the header) is rejected. §2.12.4: *"binaryFormat and binary must be those
   returned by a previous call to GetProgramBinary, and length must be the length of the program binary as returned by
   GetProgramBinary or GetProgramiv with pname PROGRAM_BINARY_LENGTH. Loading the program binary will fail, setting the
   LINK_STATUS of program to FALSE, if these conditions are not met."*
2. **A binary truncated to half its length crashes the process** (SIGILL inside `glProgramBinary`) instead of failing the
   load. Same rule; also *"If ProgramBinary fails to load a binary, no error is generated"*.
3. **`length = -1` raises no error.** ES 3.0 §2.5: *"If a negative number is provided where an argument of type sizei or
   sizeiptr is specified, the error INVALID_VALUE is generated."* (ES 3.2 §7.5 states it for `ProgramBinary` explicitly.)
   The program stays linked, as required.
4. **`glGetProgramBinary` with a too-small `bufSize` writes `*binaryFormat` and `*length = 0`** while it raises
   `INVALID_OPERATION` (§2.5: "no change is made to these values"). On the other `INVALID_OPERATION` paths (never-linked
   program, failed load) ANGLE leaves both untouched.

## Pitfalls

- **`NUM_PROGRAM_BINARY_FORMATS` may be 0.** Check it before you build a cache on binaries.
- **Treat `LINK_STATUS` after `glProgramBinary` as authoritative.** A rejected binary raises no GL
  error. Keep the GLSL sources and fall back to `glLinkProgram`. Shaders that are still attached and
  the attribute bindings survive the failed load.
- **Uniforms are reset.** A loaded program starts with every default-block uniform at 0 / FALSE
  (in ES 3.2, at its initializer). Upload them again after every load.
- **Uniform locations may differ** between the original and the reloaded program. ES 3.0 has no
  explicit uniform locations, so query them on the program you draw with.
- **Attribute and output assignments come from the binary**, not from `glBindAttribLocation` calls
  made on the new program before the load.
- The retrieval hint only takes effect at the next successful link. What the query returns between
  `glProgramParameteri` and that link is not specified; the demo prints it.

## Run

```
./build/debug/bin/es30-program-binary                       # window
./build/debug/bin/es30-program-binary --headless --validate --capture out.png
```
