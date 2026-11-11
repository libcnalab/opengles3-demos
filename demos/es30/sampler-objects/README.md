# es30-sampler-objects

**API introduced in:** OpenGL ES 3.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00 ·
**Run:** `es30-sampler-objects [--headless --validate]`

## Purpose

In OpenGL ES 2.0 the way a texture is *sampled* (filters, wrap modes) is part of the texture object:
two draws that want the same image with different filters must either change the texture's
parameters between the draws or keep two copies. ES 3.0 adds **sampler objects**. A sampler object
holds only the second half of a texture's state, the *sampling state*: `TEXTURE_MIN_FILTER`,
`TEXTURE_MAG_FILTER`, `TEXTURE_WRAP_S/T/R`, `TEXTURE_MIN_LOD`, `TEXTURE_MAX_LOD`,
`TEXTURE_COMPARE_MODE` and `TEXTURE_COMPARE_FUNC`. It is bound to a texture *unit*, and "when a sampler
object is bound to a texture unit, its state supersedes that of the texture object(s) bound to that
texture unit" (ES 3.0 §3.8.2). The shader is unchanged: a `sampler2D` uniform names a unit, and the unit
decides where the texels and the sampling rules come from.

This demo shows each consequence of that sentence, and verifies it by reading pixels, not by checking
that the calls return.

## What the image shows

A 4 x 4 contact sheet; every cell is 64 x 64 pixels and one draw call. In the capture row 3 is on top.
Textures: **A** is a 4x4 RGBA8 texture whose *own* state is `NEAREST` + `REPEAT`; **B** has the same texels but its own
state is `LINEAR` + `CLAMP_TO_EDGE`; **M** is 64x64 with 7 mipmap levels, each level a single flat color
(red, green, blue, yellow, magenta, cyan, white for levels 0 to 6). A, B and the split cells show
texture coordinates over [-1, 2] (three repetitions, so wrap modes and filters are all visible).

| Cell | What is bound | What you see | Spec |
|---|---|---|---|
| 0 | A, no sampler | A's own `NEAREST`/`REPEAT`: hard texels, three repeats | §3.8.2 |
| 1 | A + sampler `LINEAR`/`CLAMP_TO_EDGE` | smooth texels, the edge texels smeared outwards: the sampler wins | §3.8.2 |
| 2 | split: unit 0 - the sampler of cell 1 is **deleted** while bound; unit 1 - A with a sampler in its *initial* state | left: A's own state again, identical to cell 0. Right: black. The initial `MIN_FILTER` is `NEAREST_MIPMAP_LINEAR` and A has no mip chain, so A is incomplete on unit 1 and samples as (0, 0, 0, 1), while the same texture is complete on unit 0 | §3.8.2, §3.8.13.1 |
| 3 | B, no sampler | B's own `LINEAR`/`CLAMP_TO_EDGE` | |
| 4 | B + sampler `NEAREST`/`REPEAT` | hard texels and repeats although B itself is `LINEAR`/`CLAMP`: "vice versa" of cell 1 | §3.8.2 |
| 5 | B on unit 0 + sampler `NEAREST`/`REPEAT`, **the same B on unit 1** + sampler `LINEAR`, `MIRRORED_REPEAT`/`CLAMP_TO_EDGE` | one draw call, two looks of the same texels side by side (left tiled and blocky, right smooth, mirrored in s and clamped in t) | §3.8.1, §3.8.2 |
| 6 | A on units 0 and 1 with **one** sampler (`LINEAR`/`CLAMP`) | both halves identical | §3.8.2 |
| 7 | the same sampler after `glSamplerParameteri(MIN/MAG_FILTER, NEAREST)` | both halves changed: modifying a sampler affects every unit it is bound to | §3.8.2 |
| 8 | A + sampler `NEAREST`/`CLAMP`, then `glTexParameteri(A, MIN/MAG_FILTER, LINEAR)` | still `NEAREST`/`CLAMP`: TexParameter changes the texture, not the sampler | §3.8.2 |
| 9 | `glBindSampler(0, 0)` | A's own state, now with the `LINEAR` that cell 8 stored in the texture (`REPEAT` wrap) | §3.8.2 |
| 10 | M + sampler `NEAREST_MIPMAP_NEAREST`, default LOD range | level 1 (green) | §3.8.10.4 |
| 11 | the sampler's `MIN_LOD` = 3 (`glSamplerParameterf`) | level 3 (yellow): λ is clamped up | §3.8.10.1 eq. 3.18 |
| 12 | `MIN_LOD` = -1000, `MAX_LOD` = 0 (`glSamplerParameteri`) | level 0 (red): λ is clamped down | eq. 3.18 |
| 13 | `LINEAR_MIPMAP_LINEAR`, `MIN_LOD` = `MAX_LOD` = 1.5 (`glSamplerParameterfv`) | the 50/50 mix of levels 1 and 2 (teal) | eq. 3.24, 3.25 |
| 14 | M with its **own** `MIN_LOD` = 3 (`glTexParameterf`), no sampler | level 3 (yellow) | §3.8.7 |
| 15 | the same M plus a sampler with the default LOD range | level 1 (green): the sampler overrides the texture's LOD range | §3.8.2 |

The LOD cells use a quad that maps 128 texels onto 64 pixels in `u` and keeps `v` constant, so the
scale factor ρ is exactly 2 and λ = 1 for every pixel; there is no dependence on how an implementation
approximates ρ (§3.8.10.1 only bounds it).

## Central calls

`glGenSamplers`, `glBindSampler(unit, sampler)` (note: a unit *index*, not `GL_TEXTUREi`),
`glSamplerParameter{i,f,iv,fv}`, `glGetSamplerParameter{i,f}v`, `glIsSampler`, `glDeleteSamplers`,
and `glGetIntegerv(SAMPLER_BINDING)`. `glSamplerParameterIiv` / `Iuiv` and their getters are
OpenGL ES 3.2 (border colors) and are not used. `SamplerParameter::MaxAnisotropy` in meta-gl's enum
belongs to an extension and is not used either.

## Key ideas

- **Per unit, not per texture.** `SAMPLER_BINDING` is state of the texture unit (queried for the active
  unit). One sampler may be bound to several units, one texture to several units, and each combination
  is sampled independently: that is what cell 5 shows.
- **Unbinding gives the texture its state back.** `glBindSampler(unit, 0)` and deleting a bound sampler
  have the same effect (cells 2 and 9).
- **A sampler object exists from `glGenSamplers` on.** Unlike textures (which are created by the first
  bind), `glIsSampler` is `TRUE` for a name returned by `glGenSamplers` (§6.1.5); the name acquires
  its state on first use in `glBindSampler`, `glSamplerParameter*`, `glGetSamplerParameter*` or `glIsSampler`.
- **`glTexParameter` never touches a sampler.** It modifies the texture bound to the active unit
  (cell 8), so with a sampler bound you can change the texture and see nothing until the sampler goes away.
- **LOD clamping is part of the sampler.** The clamp of equation 3.18 can force or forbid mip levels
  without any change to the texture (cells 10 to 13), which is how streaming code limits
  detail while the high-resolution levels are not yet resident.
- **The data type of a parameter is just a convenience.** `glSamplerParameteri` on `MIN_LOD` converts the
  integer to float (§2.3.1), `glSamplerParameterf` on an enum parameter is accepted when the float
  holds an enum value.

## Validation (`--validate`)

59 checks, none of which only checks that a call returned:

- **Pixels, cells 0 to 9.** Grids of probes (about 100 to 250 per cell) are compared with a CPU implementation of ES 3.0 §3.8.10.2:
  nearest selection, bilinear weights (equation 3.22), and `REPEAT`, `MIRRORED_REPEAT` and
  `CLAMP_TO_EDGE` (Table 3.22), independently for s and t. NEAREST probes within 0.1 texel of a
  texel boundary are skipped (where an implementation's rounding is allowed to differ) and the rest must
  match exactly. LINEAR probes may differ by +-3/255: the spec leaves the precision of the filter weights
  to the implementation. For the split cells the left and right half are compared separately, each with
  its own sampler's model; the incomplete half of cell 2 must be (0, 0, 0, 255) at every probe.
- **Pixels, cells 10 to 15.** Each cell must show the expected mip level color (+-1/255); the 50/50 mix of cell 13
  gets +-3/255.
- **Negative controls.** Cells 0 and 1 (same texture, same coordinates, with and without a sampler)
  must differ in many pixels, so "the sampler was ignored" cannot pass; and cell 0's pixels must *not* match the CPU model of
  `LINEAR`/`CLAMP_TO_EDGE`, so the comparison can fail.
- **State read back.** `SAMPLER_BINDING` while a sampler is bound, after deleting it, and per unit for several
  units; `glIsSampler` for generated, deleted, zero and never generated names; the initial state of a new sampler
  (`NEAREST_MIPMAP_LINEAR`, `LINEAR`, `REPEAT` x 3, -1000, 1000, `NONE`, `LEQUAL`); a round trip of every parameter
  through each setter form `i`, `f`, `iv`, `fv`; texture and sampler state after `glTexParameter` while a sampler
  is bound.
- **Defined errors.** `glBindSampler` with a deleted or never generated name (`INVALID_OPERATION`) and with
  `unit = MAX_COMBINED_TEXTURE_IMAGE_UNITS` (`INVALID_VALUE`); `glSamplerParameter*` and `glGetSamplerParameter*` on
  sampler 0, a deleted sampler or a never generated name (`INVALID_OPERATION`); illegal enum values
  (`INVALID_ENUM`, and the parameter keeps its value). Deleting unused names and 0 is silent.

`TEXTURE_COMPARE_MODE` and `TEXTURE_COMPARE_FUNC` are only verified as state (set and read back);
their effect belongs to the depth-texture demo.

## Spec ambiguity

ES 3.0 §3.8.2 says that an illegal value for a sampler parameter generates "an error ... as specified in the description of
TexParameter*", but the ES 3.0.6 text of TexParameter does not list that error. Table 3.20 gives the legal values and
ES 3.2 §8.10 states `INVALID_ENUM`; the demo checks `INVALID_ENUM` and cites both.

## Pitfalls

- `glBindSampler` takes the unit **index** (0, 1, ...), while `glActiveTexture` takes `GL_TEXTURE0 + i`.
- Sampler uniforms (`glUniform1i`) hold unit indices; the same texture on two units is bound twice.
- A sampler that is bound to the *wrong* unit is silently ignored by shaders that sample another unit; check
  `SAMPLER_BINDING` with the right `glActiveTexture`.
- Texture completeness is judged with the filters that are *in effect* (§3.8.13.1): a sampler whose `MIN_FILTER` needs mipmaps
  (and that is the *initial* state of every new sampler!) makes a texture without a mip chain incomplete on that unit, so it samples as
  (0, 0, 0, 1), while the same texture stays complete on a unit whose sampler does not need mipmaps (cell 2). Texture M has all 7 levels.
- Implicit LOD derivatives in non-uniform control flow are undefined; `sampled.frag` samples both units
  unconditionally and selects the result afterwards.
