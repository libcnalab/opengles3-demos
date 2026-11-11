# es20-uniforms

**API introduced in:** OpenGL ES 2.0 (uint and non-square matrix variants: ES 3.0) · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

Uniforms are a program's per-draw constants. ES 3.0 has 33 `glUniform*` and 9 `glUniformMatrix*`
entry points, and each must match the declared type exactly. This demo uses every one of them,
together with bool, struct, struct-array, partial-array and matrix-array uniforms. It then
proves three ways that the values arrived intact.

## How it works

The fragment shader (`shaders/uniforms.frag`) declares one uniform per test and divides the screen
into an 8x5 grid that partitions the whole surface: a cell is `width / 8` by `height / 5` pixels
(the size reaches the shader as `uniform ivec2 u_cell_size`), so all five rows are visible at any
aspect ratio; the minimum surface is 8x5. Cell *i* compares uniform *i* with the **exact** value the C++ code
uploaded (all values are exactly representable in binary floating point). It paints green on a
match and red otherwise. Both float and int use `highp`: a fragment shader's int precision is not
fixed, and `mediump` ints may be only 16 bits, too few for `0x80000001u`.

## What is verified

| Check | Spec |
|---|---|
| 38 cells green: the shader sees every value bit-exactly | ES 3.0 §2.12.6.1 |
| A deliberately wrong `ivec3` turns exactly its cell red (the test can fail) | — |
| `glGetUniform{f,i,ui}v` reads back 41 values, array elements one location at a time | ES 3.0 §6.1.12 |
| `glGetActiveUniform` reports arrays once as `name[0]` with size = length, and struct members as `s.a`, `sa[1].c` | ES 3.0 §2.12.6 |
| `glUniform1i` on a `float`, `glUniform4f` on a `vec3`, `count = 2` on a non-array, an unknown location, no current program: each `INVALID_OPERATION`, value unchanged | ES 3.0 §2.12.6.1 |
| Location −1 is silently ignored | ES 3.0 §2.12.6.1 |

## Concepts worth remembering

- **Exact type match.** There is no conversion: a `vec3` needs `glUniform3f[v]`, a `uint` needs
  `glUniform1ui[v]`. Only `bool`s accept any of the `i`, `ui` and `f` forms, with 0 → false.
- **Arrays.** The location of `a` equals that of `a[0]`. `glUniform*v(loc(a[k]), N, ...)` replaces
  elements k … k+N−1. Reading back requires one `glGetUniform*v` call per element.
- **Matrices are column-major.** `mat2x3` has **2 columns of 3 rows**, so `m[1][2]` is column 1,
  row 2. The upload array is 6 floats in column order with `transpose = GL_FALSE`. ES 3.0+ also accepts
  `GL_TRUE` for row-major data (§2.12.6.1); only ES 2.0 requires `GL_FALSE`.
- **Structs** have no location of their own. Each member has one, named `s.member` / `arr[i].member`.
- **Uniform state belongs to the program object.** `glUniform*` writes to the *current* program.
  ES 3.1 adds `glProgramUniform*` to target any program.

## Expected image

38 green cells (8 columns x 5 rows filling the surface; rows start at the bottom of the window), with
the two unused cells of the top row dark gray.

## Driver note

softpipe executes this shader incorrectly (every branch after the third reports a mismatch), while
radeonsi, llvmpipe and zink are correct. See `docs/KNOWN_LIMITATIONS.md`. softpipe is therefore not
part of this demo's test matrix.
