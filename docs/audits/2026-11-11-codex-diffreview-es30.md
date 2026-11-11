# Independent diff review

Reviewed `git diff 5bd59a6..HEAD` for the 17 assigned `demos/es30` directories. The 12 changed C++ targets built with `-j 1`; their llvmpipe ES-max and ES-min CTest runs passed (24/24). Manual `--validate` runs passed at 256×256. The 11 demos with revised layouts also passed at 1000×200 and 200×1000. All runs used surfaceless llvmpipe.

## Findings

- **BUG — `demos/es30/instancing/main.cpp:46,456`; `demos/es30/instancing/README.md:22`.** The new 60-pixel cell minimum admits 240×240, but validation fails 7 checks there (112/119): several cell color/probe checks fail because their probes land within 1 px of model edges. The README calls 240×240 the minimum, so this permitted size is not validated; raise the minimum or adjust the probe layout. The run at the stated minimum reproduced this.
- **WEAK — `demos/es30/post-processing/main.cpp:1338-1341`.** The dark-channel floor was loosened from `w*h/4` to `w*h/8` (8.33% to 4.17% of channel samples). The comment gives an observed ~6% for a 5:1 surface, but does not derive the new floor from a lower bound or spec allowance; this check can pass with substantially fewer dark samples than the cited observation. Record the basis for that floor or derive it from the expected scene coverage.

## Per-demo status

- `buffer-mapping` — ok; README clarification correctly describes synchronized mapping and the staging-copy possibility.
- `deferred-shading` — ok; changed negative-control shares use CPU-predicted scene pixels, and the README explains the empty-background dilution.
- `depth-textures` — ok; known-deviation matching is limited to a renderer string containing both `Vulkan` and `llvmpipe`; llvmpipe ES-max/min runs stayed strict and passed.
- `glsl-texture-functions` — ok; README explains the stage-specific implicit-LOD rules and the 148 fragment lookup blocks.
- `instancing` — see BUG above.
- `interactive-3d-scene` — ok; size-dependent derivative neighborhood and 180-pixel shorter-side minimum are documented; minimum-size and wide/tall runs passed.
- `lighting` — ok; portrait FOV is mirrored in the CPU camera model; wide/tall runs passed.
- `multisample-default-framebuffer` — ok; README walkthrough matches the render and resolve sequence.
- `normal-mapping` — ok; portrait FOV is shared by rendering and CPU ray validation; wide/tall runs passed.
- `post-processing` — see WEAK above; remaining checks and wide/tall runs passed.
- `shadow-mapping` — ok; README’s comparison-then-filter explanation matches the shadow sampling rule.
- `skybox-and-fog` — ok; near-plane adjustment and close-up Fresnel negative control are validated; wide/tall runs passed.
- `srgb` — ok; wording no longer overgeneralizes all 8-bit images as sRGB encoded.
- `sync-objects` — ok; exact cell partition, scaled rectangle model, and 64×64 minimum passed at the boundary and wide/tall sizes.
- `texture-3d-and-arrays` — ok; boundary probes accept either adjacent texel only for positive checks, while negative controls omit them; 128×128 minimum and wide/tall runs passed.
- `transform-feedback` — ok; geometry-derived scan counts replace fixed pixel counts, with the quarter/half bounds explained; 128×128 minimum and wide/tall runs passed.
- `uniform-buffer-objects` — ok; the README documents the revised cell sizing and 192×120 minimum; minimum-size and wide/tall runs passed.
