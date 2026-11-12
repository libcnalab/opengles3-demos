# Diff review (base `f75838c`)

## Findings

- **WEAK — `demos/es30/srgb/README.md:70`:** The README says validation performs 57 checks, but the requested llvmpipe run reports `checks=56/56`. Update the documented count to match the executable.
- **WEAK — `demos/es32/cube-map-array/README.md:31-32`:** The README says portrait viewports keep a 45° horizontal field of view, but `layout_of()` clamps the vertical half-angle tangent to 1. For aspect ratios below `tan(22.5°)` (~0.414), that clamp makes the horizontal FOV less than 45°. Describe the 90° cap as an exception to the 45° portrait rule.

## Per-demo result

- `es30-srgb` — otherwise OK; negative controls passed.
- `es30-texture-swizzle-lod` — OK; cell comparisons and negative controls passed.
- `es31-multisample-textures` — OK; sample-mask and reversed-sample negative controls passed.
- `es31-stencil-texturing-and-level-queries` — OK; validation passed.
- `es31-texture-gather` — OK; row-order and wrap negative controls passed.
- `es32-cube-map-array` — otherwise OK; tile and sphere negative controls passed.

## Runs

Configured with the supplied meta-gl source and built the debug preset using `-j 1`. Ran all six demos with `--headless --validate` on llvmpipe only. Also ran surface-size checks: sRGB 16×16, swizzle/LOD 32×32, multisample textures 64×64, stencil texturing 64×64, texture gather 40×40, and cube-map-array 80×89 and 1000×101. Every run passed.
