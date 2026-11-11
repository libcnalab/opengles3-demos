# es20-cube-map

**API introduced in:** OpenGL ES 2.0 · **Minimum context:** ES 3.0 · **Shaders:** GLSL ES 3.00

## Purpose

A cube map texture is six square 2D images, the faces of a cube, addressed by a *direction* instead of by
texture coordinates. This demo uploads six different 2×2 faces, samples them through a `samplerCube`, and
checks the rule that turns a direction into a face and a texel — ES 3.0 §3.8.9 and its Table 3.21 — against
an independent CPU implementation, together with the rules that make a cube map *complete*.

All 24 texels are different colours: each face has a hue of its own (+X red, −X cyan, +Y green, −Y magenta,
+Z blue, −Z yellow) and the four texels of a face are four brightnesses of that hue (the first row of the
uploaded data brightest). A face that is swapped, mirrored in `s` or `t`, rotated or transposed therefore
changes the colour of at least one probe.

## The image

```
y = 0 ..191   skybox: the camera sits at the cube centre and looks towards the corner where +X, +Y and -Z meet
y = 192..223  six 32x32 face tiles: +X  -X  +Y  -Y  +Z  -Z   (two empty cells)
y = 224..255  four [incomplete | complete] pairs of cube maps (the incomplete one is black)
```

Each face tile samples directions that lie on the plane of one face (Table 3.21 solved for the direction), so a
tile must show that face's 2×2 image upright and unmirrored: the first uploaded row at the top, the first texel of
the row at the left. The bottom pairs show the +X face of a cube map that is missing a face, has a face of the wrong
size, has a face of the wrong format, or needs mipmaps it does not have — black — next to a complete twin.

The rows above are for the default 256×256 surface; the layout scales with any surface. Eight tile columns
span the width (`width / 8`). Tiles are square as long as their two rows take at most half of the height;
on a wider surface they get flatter (`height / 4`), so the skybox always keeps at least half the height. The
skybox camera has a 70° vertical field of view, and the horizontal one follows the aspect ratio. On a
very wide or very tall sky rectangle each half-angle is kept between 14° and 60°, so pixels are no longer
square: the view never turns into a fisheye strip (mostly near face edges, where probes must be skipped)
or into a sliver that shows only one or two faces. The checks use the same scale factors. The single-direction
probes are drawn and read back in batches of one surface, so they work at any size.

**Minimum size.** `init()` throws `demo::Skip` unless two conditions hold. Both follow from what the
checks need, and both depend only on the layout:

1. Every face tile is at least 2×2 pixels, so **width ≥ 16 and height ≥ 8**. In a 2-pixel tile the pixel
   centres sit at s, t = 0.25 and 0.75, half a texel from the boundary. Each of the four texels then gets a
   probe, and the face-tile check requires that. A 1-pixel tile has its only pixel centre exactly on the
   texel boundary, so it would have no probe at all.
2. The CPU model of the skybox (the same Table 3.21 lookup the skybox check uses) predicts more than 100
   unambiguous pixels on each of at least three faces, and more than 60 % of the sky rectangle away from texel
   and face edges. These are the skybox check's own conditions. A sky only a few pixels high, or too
   narrow for its clamped field of view, cannot meet them.

The second condition depends on the aspect ratio, so the minimum is not a single rectangle. Some smallest
admitted sizes: 26×26 (25×25 skips), 16×35 (16×34 skips), 400×25 (400×24 skips), and 1000×10 (1000×9 skips). On llvmpipe,
a sweep of every 3rd width from 8 to 128 against every 3rd height from 7 to 127 gives only PASS or SKIP.

## Key ideas

- **One object, six images.** `glBindTexture(GL_TEXTURE_CUBE_MAP, id)` binds the whole cube map;
  `glTexImage2D` takes the face as its *target* (`GL_TEXTURE_CUBE_MAP_POSITIVE_X` … `_NEGATIVE_Z`, consecutive
  enumerators in the order +X −X +Y −Y +Z −Z). Passing `GL_TEXTURE_CUBE_MAP` itself to `glTexImage2D` is
  `INVALID_ENUM`. `GL_TEXTURE_BINDING_CUBE_MAP` is a binding of its own, independent of `GL_TEXTURE_BINDING_2D`.
- **Table 3.21.** The direction `r = (rx, ry, rz)` selects the face by its *major axis* (the component of largest
  magnitude and its sign); with `sc`, `tc`, `ma` from the table
  `s = ½ (sc/|ma| + 1)`, `t = ½ (tc/|ma| + 1)`:

  | major axis | face | sc | tc | ma |
  |---|---|---|---|---|
  | +rx | POSITIVE_X | −rz | −ry | rx |
  | −rx | NEGATIVE_X | +rz | −ry | rx |
  | +ry | POSITIVE_Y | +rx | +rz | ry |
  | −ry | NEGATIVE_Y | +rx | −rz | ry |
  | +rz | POSITIVE_Z | +rx | −ry | rz |
  | −rz | NEGATIVE_Z | −rx | −ry | rz |

  `(s, t) = (0, 0)` is the corner of the *first texel of the first row* of the uploaded data, and `t` grows with
  the row index. Only the ratios `sc/|ma|`, `tc/|ma|` matter, so the direction does not have to be normalised.
  The spec leaves ties on the major axis to the implementation ("the rule must be deterministic and depend only on
  rx ry rz"); the demo never probes a tie.
- **Seamless filtering.** In ES 3.0 (a change from ES 2.0, §3.8.9.1) wrap modes are ignored for cube maps, NEAREST
  always clamps to the face edge, and LINEAR takes missing texels from the neighbouring face. The demo samples with
  NEAREST only and stays away from face edges, so none of this is exercised.
- **Orientation.** Every face image is oriented as a viewer *outside* the cube sees it: `s` grows to the right and
  `t` downwards. For the +Z face that is `sc = rx, tc = −ry`; for +X (`sc = −rz`) the viewer stands at +X looking
  towards −X, where the right-hand side is −Z. Seen from the *inside* — a camera at the centre in a right-handed
  world, as in a skybox — each face therefore appears mirrored left-to-right relative to the uploaded image
  (the well-known "cube maps are left-handed" convention). Content authored for another API usually needs per-face
  flips; the face tiles (outside view) and the skybox (inside view) show what the GL does.
- **Completeness (§3.8.13).** A cube map is *cube complete* when the base-level images of the six faces have identical,
  positive, square dimensions and the same effective internal format; a cube map that is not cube complete is
  incomplete and samples (0, 0, 0, 1) without a GL error. It must also be mipmap complete if `TEXTURE_MIN_FILTER`
  needs mipmaps, and the *default* filter, `NEAREST_MIPMAP_LINEAR`, does: a complete-looking cube map with only
  level 0 is black until `glGenerateMipmap(GL_TEXTURE_CUBE_MAP)` creates the levels of all six faces, or the filter
  is changed. `glGenerateMipmap` on a cube map that is not cube complete is `INVALID_OPERATION` (§3.8.10.5).
  Uploading a non-square face is `INVALID_VALUE` at once (§3.8.3).
- **Shader side.** `texture(samplerCube, vec3)` takes the direction. The fragment shader builds it in three ways
  by a uniform: from the plane of a face (the face tiles), from an interpolated camera ray (the skybox), or from
  a uniform (single-pixel draws for exact directions).

## Validation

35 checks per run. Colours of NEAREST lookups in an RGBA8 texture are exact, so every comparison has tolerance 0.
Probes within 0.05 texel of a texel boundary, or whose `max(|sc|, |tc|)/|ma|` exceeds 0.98 (face edge, or a tie
on the major axis), are skipped.

- **Skybox:** every pixel that is not skipped (38 661 of 49 152, 79 %) equals the CPU lookup of its view ray. The CPU
  evaluates the very linear function that the vertex shader interpolates. At least three faces must be visible.
- **Face tiles (6 checks):** each tile equals the face's own 2×2 image, pixel by pixel.
- **Single directions** (one pixel per draw, `u_dir`):
  - eight directions worked out on paper (one per face and two oblique, non-normalised ones), against both the GL and the CPU model;
  - the 24 texel centres (face, texel) — `face_direction(f, (i + ½)/2, (j + ½)/2)` — must return exactly their own texel;
  - copies scaled by 1000 and 0.001 must give the same texel;
  - 1500 pseudo-random directions (seeded; random length) agree with the CPU model, and every one of the 24
    (face, quadrant) texels is hit at least 10 times.
- **Completeness:** four scenarios × 3 checks: the broken tile is exactly (0, 0, 0, 1) and the complete twin shows
  the face; `glGenerateMipmap(TEXTURE_CUBE_MAP)` on the incomplete cube maps → `INVALID_OPERATION`; one repairing
  call on the *same* texture object and the face appears.
- **State and errors:** `TEXTURE_BINDING_CUBE_MAP`, its independence from `TEXTURE_BINDING_2D`,
  `MAX_CUBE_MAP_TEXTURE_SIZE ≥ 2048` (the ES 3.0 minimum, Table 6.28; all three Mesa drivers report 16384), non-square face →
  `INVALID_VALUE`, a face larger than the maximum → `INVALID_VALUE`, target `TEXTURE_CUBE_MAP` for `glTexImage2D` →
  `INVALID_ENUM`, a cube map bound to `TEXTURE_2D` → `INVALID_OPERATION`.

**Negative controls.** Three deliberately wrong CPU models — `t` flipped, `s` flipped, `s` and `t` exchanged — are run
on the same random probes. The two flips match the GL at none of the 1290 probes, and the transposition at 622 of them (about half: it
cannot be told apart from the right answer on the diagonal texels). Separately, one texel of the
−Y face is overwritten with `glTexSubImage2D`: exactly the quadrant of that texel changes in the −Y tile, all other
pixels stay, so the tile comparison would fail; the texel is then restored.

All three Mesa drivers (radeonsi, llvmpipe, zink) at es-min and es-max produce *identical* captures and agree with the
model at every probe; no driver deviation was found.

## Pitfalls

- A cube map is black — silently — if one face is missing, differs in size or format, or the minification filter
  needs mipmaps that do not exist (the default!). Check the faces and `TEXTURE_MIN_FILTER` first.
- `glTexImage2D` accepts a face of the wrong size (it only rejects non-square ones); the cube map becomes incomplete.
- The direction is not normalised by the GL; scaling it changes nothing, but a zero vector is undefined.
- Table 3.21's orientation is not the orientation of most image tools or of other APIs; test it with an asymmetric
  image, as this demo does, before trusting a skybox.
- The face order is +X −X +Y −Y +Z −Z. The enumerators are consecutive in gl.xml (0x8515 … 0x851A), so
  `GL_TEXTURE_CUBE_MAP_POSITIVE_X + i` works, but the specification text does not promise it; this demo keeps an
  explicit table of the six targets.
