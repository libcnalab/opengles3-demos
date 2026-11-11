# es30-interactive-3d-scene: the capstone

**Minimum version:** OpenGL ES 3.0, GLSL ES 3.00 (`#version 300 es`). Everything here is core ES 3.0, with no
extensions.

The earlier demos each isolate one idea. This one puts them together in a small outdoor scene that you can walk
through: a meadow with a panelled stone wall, a glossy monolith on a stone plinth, a wooden crate, a beach ball,
a colonnade of six pillars, ten trees, a sky with a sun, haze towards the horizon, and real shadows. Each technique
can be switched off with a number key, so you can see what it contributes. The validation checks that each
technique produces exactly the pixels an independent CPU ray caster predicts.

```
./build/debug/bin/es30-interactive-3d-scene                      # window (SDL3)
./build/debug/bin/es30-interactive-3d-scene --width 1280 --height 720
./build/debug/bin/es30-interactive-3d-scene --headless --validate --capture out.png
```

## Controls

| input | effect |
|---|---|
| left mouse button + drag | look around (yaw and pitch; pitch is limited to ±85°) |
| W / S | walk forward / back along the horizontal view direction |
| A / D | strafe left / right |
| Q / E | down / up (the eye never goes below 0.3 units) |
| mouse wheel | field of view, 25°–90° (zoom) |
| 1 | shadows on/off |
| 2 | normal mapping on/off |
| 3 | fog on/off |
| 4 | wireframe overlay on/off |
| 5 | shadow-map inset on/off |
| 6 | Blinn-Phong ↔ Phong specular |
| R | reset the camera |

Movement speed is 4 units per second, scaled by `Env::frame_seconds`, so it does not depend on the frame rate.
When several movement keys are held, their directions are added and the sum is normalized: walking diagonally is
not faster. The field of view always spans the **shorter** side of the window, so a portrait window shows the same
width of the scene as a square one and is not squeezed.

## Where each earlier demo lives in this frame

`render()` first moves the camera (`update_camera`, the held keys × frame time). It then draws one frame in four
steps (`draw_frame`):

### Pass 1: the shadow map (es30-shadow-mapping)

The scene is drawn from the sun into a `DEPTH_COMPONENT24` texture of 2048², the **only** attachment of a
framebuffer object (`glDrawBuffers(NONE)`, `glReadBuffer(NONE)`; ES 3.0 §4.4.2, §4.4.4, §4.2.1, §4.3.1). The sun is
a directional light, so its projection is orthographic: `ortho(±24, near 15, far 85) * look_at(50·L, origin)`. That
box encloses every object, including the farthest trees. `shadow.vert` applies exactly the same instance transform
as the main pass, so the instanced pillars and trees cast shadows too. `shadow.frag` is empty: depth is written by
the fixed-function pipeline.

`glPolygonOffset(4.5, 4)` (ES 3.0 §3.6.2, eq. 3.9–3.11) pushes every stored depth away from the sun to prevent
shadow acne. The factor is derived, not guessed. The corner tap of the 3×3 PCF kernel in pass 2 filters texels up
to 2 texels per axis away from the fragment, a depth difference of up to 2(|∂z/∂x| + |∂z/∂y|) ≤ 2√2·|∇z|. Eq. 3.10
allows the implementation to approximate the slope m as max(|∂z/∂x|, |∂z/∂y|) ≥ |∇z|/√2. A factor of 4 therefore
keeps every tap of a lit plane lit. With factor 3 (the value es30-shadow-mapping uses with a smaller kernel
footprint) the validation found one tap of nine failing on a pillar face lit at n·L = 0.35.

### Pass 2: the scene from the camera (transformations, lighting, textures, normal maps, instancing, fog)

All objects are drawn with one program, `scene.vert` + `scene.frag`:

* **Transformations** (es20-transformations-camera). `u_view_proj = perspective * look_at(eye, eye + forward)`.
  The camera's forward vector comes from yaw and pitch: (cos p sin y, sin p, −cos p cos y). The model matrix places
  the prototype; the normal matrix is the inverse transpose of its upper 3×3.
* **Instancing** (es30-instancing). Attribute 5 is `a_instance = (offset.xyz, scale)` and attribute 6 is
  `a_tint`. For the pillars and the canopies both come from an instance buffer with
  `glVertexAttribDivisor(loc, 1)` (ES 3.0 §2.9, §2.9.3), and the draw is
  `glDrawArraysInstanced(TRIANGLES, 0, n, 6 or 10)`. The trunks use the *same* instance buffer but leave attribute 6
  **disabled** in their VAO. Non-instanced objects disable both. A disabled array is replaced by the *current
  generic attribute value* (ES 3.0 §2.8), which the demo sets once: `glVertexAttrib4f(5, 0,0,0,1)` (no offset,
  scale 1) and `glVertexAttrib3f(6, 1,1,1)` (no tint). Drawing with the arrays enabled leaves those current values
  unchanged (§2.9.3), and the validation reads them back afterwards. Each VAO records its own attribute
  configuration (§2.11).
* **Procedural textures** (es20-textures-2d, es20-mipmaps). Grass, wood, stone and the beach-ball pattern are
  painted on the CPU from tileable value noise. Each has a **complete mip chain built on the CPU** with a box filter,
  uploaded level by level into `glTexStorage2D` storage (§3.8.4). `glGenerateMipmap` would leave the filter to the
  implementation (§3.8.10), and the validation needs to know every level. Sampling uses
  `LINEAR_MIPMAP_LINEAR` + `REPEAT`. The ground computes its texture coordinates from the world position
  (`v_world.xz / 4`); the boxes have per-face coordinates proportional to the face size, so the texture keeps its
  scale on every box.
* **Normal mapping** (es30-normal-mapping). The wall and the plinth sample a stone normal map: raised panels with
  8-texel bevels tilted 40°. The vertex shader passes T, B = cross(N, T)·w and N; the fragment shader computes
  `n = normalize(mat3(T, B, N) * (texel * 2 − 1))`. The *geometric* normal still decides whether the face can see
  the sun at all. A face turned away from the sun gets no direct light whatever its normal map says, so bevels on
  back faces do not glow.
* **Lighting** (es30-lighting). A directional sun, Blinn-Phong `pow(max(N·H, 0), shininess)` or Phong
  `pow(max(R·V, 0), shininess)` (key 6), specular only where N·L > 0, plus a hemisphere ambient term that blends a
  ground colour into a sky colour by n.y. Lighting is done in linear space and gamma-encoded (1/2.2) at the end,
  because the default framebuffer is not sRGB.
* **Shadows** (es30-shadow-mapping). The fragment's position in the sun's clip space (`v_light`) is looked up in the
  depth texture through a `sampler2DShadow`. The texture has `TEXTURE_COMPARE_MODE = COMPARE_REF_TO_TEXTURE`,
  `COMPARE_FUNC = LEQUAL` and `LINEAR` filtering, so each `texture()` returns a filtered comparison over 2×2 texels
  (§3.8.15, GLSL ES 3.00 §8.8). A 3×3 grid of such taps widens the penumbra. Only the direct sunlight is shadowed;
  ambient light is not.
* **Fog** (es30-skybox-and-fog). ES has no fixed-function fog. The fragment shader computes
  `mix(fog colour, colour, exp(−0.016·distance))`. The fog colour is the sky's colour at the horizon, so distant
  ground dissolves into the sky. The ground ends at ±400 units, where the fog factor is below 1 %.
* **Wireframe overlay** (key 4). Core ES has no polygon mode. Every mesh is drawn **non-indexed**, so each triangle
  owns its three vertices and can carry a barycentric attribute (1,0,0), (0,1,0), (0,0,1). `b_i / fwidth(b_i)`
  (GLSL ES 3.00 §8.9) is the distance to edge i in pixels; pixels within 0.75 px of an edge get the full wire colour,
  fading to none at 1.5 px. Lines also fade where `fwidth(b)` is large (triangles only a few pixels tall), so distant
  ground and small sphere facets do not merge into solid colour. All implicit derivatives (`fwidth` and both
  texture lookups) are evaluated before any non-uniform branch.

### Pass 3: the sky (es30-skybox-and-fog)

A cube around the eye, drawn **after** the opaque scene. `mat3(u_view)` drops the translation; `gl_Position = clip.xyww`
puts every sky fragment at window depth 1.0. With `glDepthFunc(LEQUAL)` and depth writes off, the sky fills exactly
the pixels the scene did not cover. Drawing it last saves all the fragment work hidden behind objects. The cube map
(256² per face, LINEAR) holds display colours painted on the CPU: a horizon-to-zenith gradient, a glow and a disc
around the sun, a haze below the horizon. Face selection is ES 3.0 §3.8.9, table 3.21.

### Overlay: the shadow-map inset (key 5)

A scissored clear (§4.1.2, §4.2.3) draws a white frame. A viewport in the top-left corner then shows the depth
texture as grey (near the sun = dark). The **same** depth texture is bound to unit 3 with a **sampler object** whose
`COMPARE_MODE` is `NONE` and whose filter is `NEAREST`. The sampler state supersedes the texture's own comparison
state (§3.8.2), so a plain `sampler2D` can read raw depth while unit 2 keeps comparing. A non-comparing depth texture
must be sampled with NEAREST to be complete (§3.8.13).

Texture units are fixed for the whole run: 0 albedo, 1 normal map, 2 shadow map (comparing), 3 shadow map (raw,
through the sampler object), 4 sky.

## What you should see

The start view looks from the south-east over the meadow: the stone wall with three bevelled panels on the left, the
crate and the striped ball in front, the dark-blue monolith on its plinth in the middle, the colonnade on the right,
round trees around the edge. The sun stands to the right at 35° elevation, so the shadows fall to the left and away
from you: the crate and ball shadows on the grass, long striped pillar shadows. The far meadow fades into a pale
horizon under a blue sky. Key 5 adds the shadow map in the top-left corner: a gradient with the objects as darker
blobs. Key 4 draws magenta triangle edges. Looking towards the sun shows its disc and glow.

## How it is validated (`--validate`)

The CPU reference is an independent ray caster in double precision. It intersects rays analytically with boxes
(slab test in the box frame), spheres and the ground plane, and evaluates the same lighting equation. Validation sets
`pose_` and the toggles, the same state that input changes, renders, reads the default framebuffer back, and compares.
50 checks:

1. **State.** `VERTEX_ATTRIB_ARRAY_DIVISOR` of `a_instance` is 1 in the pillar VAO and 0 in the wall VAO. `a_tint` is
   disabled in the trunk VAO. `CURRENT_VERTEX_ATTRIB` 5 and 6 are still (0,0,0,1) and (1,1,1,1) after the instanced
   draws (§2.9.3, §6.1.12). The shadow map is in `COMPARE_REF_TO_TEXTURE` mode, and `SAMPLER_BINDING` of unit 3 is
   the raw-depth sampler.
2. **Fixed poses, every stable pixel.** About 8000 sample pixels in the central square of the shorter side (every
   second pixel at most), whatever the window size; the field of view spans the shorter side, so the objects get the
   same share of the samples in landscape, portrait and square windows, and the extra width or height adds periphery.
   A sample becomes a *probe* only when its colour can be predicted:
   * the hits at ±2 px see the same object and box face (no silhouette or crease under the pixel), and a box probe
     keeps ≥ 0.12 world units from the face border, because PCF taps near a crease read the neighbouring face, which
     has its own polygon offset;
   * **texture footprint.** ES 3.0 §3.8.10 lets ρ be approximated by any f with max(mu, mv) ≤ f ≤ mu + mv. From the
     texture coordinates of the 3×3 pixel neighbourhood the CPU bounds f ≤ F: an implementation may difference over
     the 2×2 quad that holds the pixel, forward or backward, and with coarse derivatives on the quad's *other* row or
     column, so ∂u/∂x is bounded by the horizontal differences of rows y−1, y, y+1 (and likewise ∂/∂y by columns).
     Only the sphere's u wraps at its seam; ground and box coordinates are unbounded (on far, grazing ground two
     neighbouring rows differ by several texture repeats, which must not be folded back to a small footprint). λ ≤ log2 F, so a trilinear lookup touches at most levels 0 … ⌊log2 F⌋ + 1, and on each level the
     bilinear 2×2 texels around (u, v). The filtered value is a convex combination of those texels whatever the
     weights, so the per-channel min/max of the 4×4 texels around (u, v) on those levels bounds the albedo. The
     shading is monotonic in the albedo (non-negative coefficients, then a monotonic fog mix and gamma), so the albedo
     range maps to a colour range. Probes whose predicted range is wider than 0.05 (texture edges) are dropped;
   * **normal map.** On normal-mapped faces the same footprint must hold one single normal-map texel value; the
     shading normal is then known exactly (a bevel or a flat panel top);
   * **shadow.** n·L ≥ 0.2 (or ≤ −0.02, where the sun cannot reach), and the CPU's occlusion towards the sun is the
     same on rings of radius 0.3 and 0.15 world units around the point (≥ 12 shadow-map texels, far wider than the
     PCF footprint). The PCF result is therefore exactly 0 or 1. Points near the edge of the sun's frustum are
     dropped;
   * **sky.** The 4×4 cube texels around the direction on one face, more than two texels from a face edge (seamless
     filtering mixes faces only within one texel).

   Poses: the **start** view (sky, lit ground, objects, instances); a **shadow** view of the monolith's shadow
   (≥ 30 lit and ≥ 30 shadowed ground probes); a **specular** view looking back along the sun's mirror direction on
   the monolith's +X face (≥ 20 monolith pixels whose Blinn-Phong highlight adds more than 20 levels); a **wall**
   close-up (≥ 20 bevel and ≥ 20 flat-panel probes, i.e. the TBN frame is right); a **sky** view facing the sun
   (≥ 200 sky probes, colour per view direction); a **fog** view over open ground.
3. **Instances.** From the side of the colonnade, the centre of each pillar's +X face is projected to the window.
   The nearest stable pixel of that instance within ±4 px must show its tinted, lit colour: six named checks plus the
   count. **Negative control:** drawing with `instanceCount` 5 removes the *last* instance. Its pixel then matches the
   CPU-predicted background (and not the pillar), and the whole image matches a CPU model without that pillar.
4. **Fog at known distances.** In the fog view, the ground pixels of the middle column nearest to 6, 15, 35 and 80
   units must match the fogged prediction. The same pixels must match the fog-free prediction after key 3. The fog
   factor exp(−0.016 d) is printed (0.91 … 0.26).
5. **Every toggle changes the image where predicted** (keys 1, 2, 3, 6, pressed through `on_key`). With the key
   pressed, *all* probes of the pose must match the CPU model with that technique flipped. In addition, every probe
   where the model predicts a change larger than 2·tolerance + 2 must fail the *unflipped* prediction. This is the
   negative control: the comparison can tell the two states apart. Counts between 256×256 and 800×800: shadows
   about 300–340, Phong 810–900, normal mapping 860–1090, fog 1950–3370 changed pixels (minimums 30, 20, 20, 30).
6. **Wireframe (key 4).** Seen from a pose high above the scene, looking steeply down, so that the 8-unit ground
   cells are at least 50 px wide even on a 200-pixel side. Probe points lie on edges: the centre of a box face (it lies
   on the face's diagonal) and the points at ¼, ½ and ¾ of every ground-cell edge. Each must be visible to the CPU ray, at least 12 px from every other edge of its
   two triangles (all right triangles, so every altitude is ≥ 17 px and `fwidth(b) ≤ 0.083` stays below the 0.12
   where lines start to fade). The pixel containing the projected point is ≤ 0.71 px from the edge, so its wire weight
   is exactly 1: it must show the wire colour, and must *not* with the overlay off. The centroids of both triangles of
   every ground cell, where they are ≥ 4 px inside the triangle, must be unchanged by the overlay.
7. **Shadow-map inset (key 5).** For a 12×12 grid of inset pixels the CPU traces the sun's ray through the
   corresponding texel and its 8 neighbours (NEAREST may pick a neighbour at a texel edge). On planar surfaces the
   grey level must lie between the shallowest analytic depth and the deepest plus the polygon offset
   (factor × the depth range across the neighbours). Those pixels differ from the inset-off image, and every probe
   outside the inset and its frame is unchanged.
8. **The controls are deterministic.** With `frame_seconds = 1/60` and `keys_held` set directly, a script runs:
   mouse look with the button held; a 300-px move with the button *released* (must do nothing); wheel +3; W for 30
   frames; D+E for 12; S+A for 9; Q for 120 (into the ground: clamped at 0.3); a huge upward look (pitch clamps at
   85°); a look down; wheel −20 (fov clamps at 90°); wheel +4. The CPU computes the end pose in closed form,
   independently of the per-frame accumulation, and the pose must agree to 1e-9. The frame rendered after the input
   must equal, pixel for pixel, the frame rendered with the predicted pose set directly. As a negative control, more
   than half the pixels differ from the start frame. R resets to the start pose.

### Tolerances

A pixel may lie at most **3 levels** (of 255) outside its predicted *range*. The range already contains the
uncertainty the spec allows (filter weights, LOD selection, texel choice). The 3 levels cover 8-bit conversion of the
result (±1), dithering, which is enabled by default (§4.1.9), and highp float evaluation of `pow`/`exp`/`normalize`
versus double. The measured worst case on radeonsi, llvmpipe, zink and ANGLE, at all four sizes, is 1.0 level. The inset check is
a sanity tolerance of the same size: the spec gives no bound for r in eq. 3.11. The shadow factor at a stable probe
is exactly 0 or 1, because all PCF comparisons agree.

The window size does not matter: the probes are a grid scaled to the window, every geometric margin is in world
units or pixels measured on the actual projection, and the validation passes at 256×256, 800×800, 640×480,
333×777, 1000×200 and 200×1000. **Minimum: a shorter side of 180 pixels**; below it the textured objects of the start
pose have fewer than 20 pixels whose filtered colour is predictable, and the demo reports SKIP.

## Pitfalls this demo ran into

* **PCF needs more bias than a single tap.** See pass 1: the bias must cover the kernel radius plus the bilinear
  footprint, and eq. 3.10 lets m be as small as |∇z|/√2.
* **Creases.** A lit point close to a box edge can be darkened by PCF taps that land on the adjacent face. That face
  was rasterized with its own (smaller) slope offset. It is visible only as a faint 1–2-texel line; the validation
  keeps away from it rather than hiding it.
* **`min()` of barycentrics in derivatives.** `fwidth(min(b.x, b.y, b.z))` changes meaning wherever a 2×2 quad
  straddles the switch between two components. On small triangles that produces speckle. Take `b_i / fwidth(b_i)`
  per component, then the minimum.
* **Derivatives in non-uniform control flow** are undefined (GLSL ES 3.00 §8.8, §8.9). Every `texture()` with an
  implicit LOD and `fwidth` runs before any per-fragment branch; the shadow lookups use a single-level texture whose
  min and mag filters are both LINEAR, so their LOD does not matter.
* **sampler2DShadow has no default precision** and samplers default to lowp in fragment shaders (GLSL ES 3.00
  §4.5.4): declare `precision highp sampler2D; precision highp sampler2DShadow; precision highp samplerCube;`.
* **The sky must be drawn after the scene** with LEQUAL. With LESS, sky fragments at exactly 1.0 fail against the
  cleared depth of 1.0 and the sky disappears.
