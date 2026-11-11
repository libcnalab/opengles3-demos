# es31-compute-particles

**Minimum:** OpenGL ES 3.1 · **Shaders:** GLSL ES 3.10 (`#version 310 es`) · **Run:** `es31-compute-particles [--headless --validate] [--width W --height H]`

## Purpose

This demo is a particle system that lives entirely on the GPU, the usual first real use of compute shaders. A compute
shader moves 16384 particles stored in one shader storage buffer, and the same buffer object is then bound as a vertex
buffer and drawn. The CPU sets uniforms and issues commands. It never reads or writes a particle while the demo runs.

It brings together the ES 3.1 pieces such a pipeline needs:

- a **std430 shader storage block** with a runtime-sized array of structs, and what std430 packs where;
- **`glDispatchCompute`** for one integration step per frame;
- **`glMemoryBarrier`** with the right bit for each consumer. Shader stores are not synchronized automatically with
  anything (ES 3.2 §7.11.2);
- the SSBO consumed as **instanced vertex attributes** (`glVertexAttribPointer` into the buffer, divisor 1);
- an **atomic counter** that counts respawns;
- **deterministic randomness**: a respawned particle gets its launch values from an integer hash of (particle index,
  step number), not from a clock, so every run on every driver produces the same particles.

## What the image shows

A three-nozzle sprinkler fountain over a dark ground grid, seen from about 26° above the horizon. The emitter has three
nozzles 120° apart. The CPU turns them a little every step, and a weak vortex around the vertical axis bends the
trajectories, so the three jets become **spiral arms**. Each arm leaves the emitter ice blue (fast), turns white and
orange near its apex (slow) and comes down to the ground. There the particles bounce with restitution 0.35, slide with
friction and fade to violet as their life runs out, which leaves three curved orange trails on the grid. Particles are
camera-facing quads with a soft round falloff, blended additively (`ONE, ONE`), so dense regions glow. They are
depth-tested against the grid but do not write depth. The grid fades with the distance from the fountain to give a
depth cue.

The picture fills the surface at any size. The camera backs off on portrait surfaces so that the fountain fits
across, and the billboards have a minimum size of 1.4 pixels so that a 256×256 capture stays readable. In windowed
mode the camera orbits slowly.

### Controls (also printed by `--help`)

`SPACE` pause the simulation · `a`/`d` orbit · `w`/`s` dolly · `q`/`e` raise/lower the camera · `r` reset the camera · `ESC` quit

## Central calls

| call | role |
|---|---|
| `glDispatchCompute(128, 1, 1)` | one step for 16384 particles, `local_size_x = 128` |
| `glMemoryBarrier(SHADER_STORAGE_BARRIER_BIT)` | before every step: earlier stores are visible to this step's loads, **and** this step's stores wait for the previous draw's vertex fetches (write after read) |
| `glMemoryBarrier(VERTEX_ATTRIB_ARRAY_BARRIER_BIT)` | after the step: the draw's vertex fetches see the stores |
| `glMemoryBarrier(BUFFER_UPDATE_BARRIER_BIT)` | before `glMapBufferRange`, `glCopyBufferSubData` and the `glBufferSubData` that resets the counter |
| `glBindBufferBase` / `glBindBufferRange(SHADER_STORAGE_BUFFER, 0, ...)` | the particle array; with a range, `p.length()` is the range size divided by 32 |
| `glBindBufferBase(ATOMIC_COUNTER_BUFFER, 0, ...)` | the respawn counter |
| `glVertexAttribPointer(1/2, 4, FLOAT, stride 32, offset 0/16)` + `glVertexAttribDivisor(.., 1)` | `pos.xyz, age` and `vel.xyz, life` of each particle as per-instance attributes |
| `glDrawArraysInstanced(TRIANGLE_STRIP, 0, 4, 16384)` | one billboard per particle |
| `glMapBufferRange(..., MAP_READ_BIT)` | validation read-back |
| `glGetProgramResourceiv(BUFFER_VARIABLE, OFFSET / TOP_LEVEL_ARRAY_STRIDE)` | the std430 layout as the GL reports it |

## Spec sections

- ES 3.2 §17 Compute Shaders; §7.8 Shader Buffer Variables and Shader Storage Blocks; §7.6.2.2 the std430 rules
  (a `vec3` has base alignment 16 but size 12, so the `float` after it occupies bytes 12..15; a runtime-sized array's
  `BUFFER_DATA_SIZE` counts one element).
- ES 3.2 §7.11.2 Shader Memory Access Synchronization (the barrier bits quoted above).
- ES 3.2 §7.7 Atomic Counter Buffers; §6.1.1 `BindBufferRange`; §6.3 mapping; §6.5 `CopyBufferSubData`.
- ES 3.2 §10.3 vertex arrays, `VertexAttribDivisor`, §10.5 `DrawArraysInstanced`; §15.1.5 blending.
- GLSL ES 3.10 §4.3.7 buffer variables, §4.3.9 interface blocks, §4.4.4 block layout qualifiers (std430, binding), §4.1.9 runtime-sized arrays and `.length()`; §4.4.6 atomic counter
  layout, §8.10 `atomicCounterIncrement`; §7.1.3 `gl_GlobalInvocationID`; **§4.7.1 Range and Precision**, which is
  the basis of the validation tolerance.

## Shader concepts

`simulate.comp` declares

```glsl
struct Particle { vec3 pos; float age; vec3 vel; float life; };
layout(std430, binding = 0) buffer Particles { Particle p[]; };
layout(binding = 0, offset = 0) uniform atomic_uint u_spawned;
```

so one element is 32 bytes and the C++ `struct Particle { float px, py, pz, age, vx, vy, vz, life; }` matches it
exactly. Under std140 the struct would be the same here, but this is because `float` packs after `vec3` in both
layouts. The demo states std430 because that is the SSBO layout of choice, and checks the offsets the GL reports.

Every invocation first checks `gl_GlobalInvocationID.x < p.length()`, because the last work group may extend past the
bound range. Ages and lives are integer-valued floats counted in steps, so `age += 1.0` and `age >= life` are exact.
Floats hold integers below 2^24 exactly, and comparisons are exact. A dead particle is respawned with values from a
lowbias32 integer hash. Bits 8..31 of the hash become a float in [0, 1) through `float(h >> 8) * (1.0 / 16777216.0)`,
which is exact as well. The nozzle directions are uniforms (`vec2 u_arm[3]`) computed on the CPU, because trigonometric
functions have no defined precision in GLSL ES 3.10. Everything else is `+ - * /` written out scalar by scalar, in the
same order as the CPU model.

The vertex shader expands each instance to a quad in eye space (`eye.xy += corner * half_size`), so the billboards
always face the camera. Their size never falls below `u_min_px` pixels.

## Validation (`--validate`)

All checks run on scratch buffers. The display buffer is never touched, and the capture shows the fountain.

1. **Layout (state query).** `glGetProgramResourceiv` on `p[0].pos / age / vel / life` returns OFFSET 0 / 12 / 16 / 28
   and TOP_LEVEL_ARRAY_STRIDE 32, and the block's BUFFER_DATA_SIZE is 32. The vertex attribute offsets and stride
   depend on these values.

2. **Rounding bound (how the tolerance is derived).** GLSL ES 3.10 §4.7.1 makes highp `+ - *` correctly rounded, "the
   rounding mode is not defined but must not affect the result by more than 1 ULP". `a*b+c` may be fused or not, `/` is
   within 2.5 ULP, comparisons are exact and subnormals may be flushed. Since ULP(x) ≤ 2^-23·|x|, the CPU model
   evaluates the shader's expression tree on a value type `E = {value, bound}`. Each operation propagates the input
   bounds to first order, exactly (`|x'y' − xy| ≤ |x|e_y + |y|e_x + e_x e_y`, and the analogous quotient formula), and
   then adds `k·2^-23·(|value| + propagated) + 2^-126` with k = 1, or 2.5 for division. The result bounds the distance
   of a conforming GPU result from the exact result. A fused multiply-add is never worse than the two-rounding bound.
   The comparison allows a further 1e-12 relative slack for the double arithmetic of the model. Every check prints the
   largest observed |gpu − model| / bound.

3. **Step-wise (all branches).** The developed fountain (302 steps) is copied with `glCopyBufferSubData` and stepped
   three times. After every step the whole buffer is mapped, and every particle is compared with one model step from the
   **bit-identical** previous GPU state. Positions and velocities must lie within the bound, and ages and lives must be
   equal. The atomic counter must equal the model's respawn count exactly. Respawn and bounce must both occur, typically
   about 450 respawns and 9000 bounces. A bounce decision whose `y` lies within its own rounding bound of the plane may go
   either way under §4.7.1. The model then accepts either branch, and the number of such cases is printed (it has been 0
   on every driver).

4. **Trajectory (N steps).** 4000 particles are uploaded in a CPU-defined state and run for 90 steps with
   `glBindBufferRange` covering exactly 4000 elements. 32 groups give 4096 invocations, so the 96 guard elements behind
   the range must stay bit-identical, which shows that `p.length()` = 4000. The bounce is disabled here (ground at
   −1000), so the only branch is the exact respawn decision and nothing chaotic can amplify differences. The model is
   run **independently** from the uploaded state, and its bound is propagated through all 90 steps. Every particle must
   be within it, with ages and lives equal and the atomic counter equal to the model's ~3300 respawns. The measured
   error is about 1e-5 against bounds of up to 1.3e-4.
   **Negative controls:** the same comparison must reject a model with gravity 2 % too strong (about 3980 of 4000
   particles rejected, ratios in the thousands) and a model that runs one step fewer (all 4000 rejected).

5. **Render (barrier-ordered vertex fetch).** 17 particles are uploaded 9 units *above* a 4×4 grid of targets in the
   plane z = 0 (outside the view), with a velocity that one step of `p += v·dt` brings exactly onto the targets. The
   17th particle lands on target 5. The validation camera fits the grid into 70 % of the surface at any aspect ratio. A
   precondition check requires quads of at least 4 px half size that are at least 8 px apart. Flat colour, `ONE, ONE`:
   - drawn **before** the compute step, every target pixel shows the clear colour (negative control: the probes can
     tell "no particle" from "particle");
   - after `glDispatchCompute` + `VERTEX_ATTRIB_ARRAY_BARRIER_BIT`, the centre ±2 px of every CPU-projected quad is
     clear + flat colour, target 5 shows clear + 2 × flat colour, and points 3 px outside each quad show the clear
     colour. All colours are multiples of 1/255 and their sums do not saturate, so the tolerance is ±1 (one 8-bit
     conversion per blend, ES 3.2 §2.3.5.2 / §15.1.5);
   - the stored positions read back are within 1e-4 of the targets.

There are 31 checks in total. All of them PASS on gpu (radeonsi), llvmpipe and zink, at ES 3.2 and under the ES 3.1
override, at 256×256, 800×800, 640×480 and 333×777. Surfaces smaller than 96×96 report SKIP_UNSUPPORTED.

## Pitfalls

- **No barrier, no guarantee.** Shader stores are incoherent with vertex fetching. `VERTEX_ATTRIB_ARRAY_BARRIER_BIT`
  is the bit for "the SSBO is now an ARRAY_BUFFER". `SHADER_STORAGE_BARRIER_BIT` is the bit for the next dispatch and
  also covers the write-after-read hazard against the previous draw. Before mapping you need
  `BUFFER_UPDATE_BARRIER_BIT`, not `SHADER_STORAGE`.
- **`vec3` in buffers.** It has a base alignment of 16 but a size of 12. A lone `vec3[]` array has a 16-byte stride, and
  only a following scalar fills the gap. Query the offsets instead of assuming them.
- **Partial last work group.** Dispatches are counted in groups, so guard with `.length()` (or a uniform count).
  Otherwise the extra invocations write outside the bound range, which is undefined.
- **Reading an SSBO in the vertex shader** needs `MAX_VERTEX_SHADER_STORAGE_BLOCKS > 0`, which ES 3.1 allows to be 0.
  Sourcing the data as vertex attributes works everywhere.
- **Point sprites** may be limited to 1 pixel (`ALIASED_POINT_SIZE_RANGE` only has to include 1), so the demo uses
  instanced quads.
- **Validation of float simulations.** Compare one step from identical inputs (no amplification) and short
  non-chaotic trajectories, and keep discrete decisions on exact quantities (integer ages, integer hashes). Do not use
  `sin`/`cos`/`length` in the simulated path, because their precision is undefined.
