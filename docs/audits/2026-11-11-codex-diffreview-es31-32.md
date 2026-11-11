# Independent diff review

Built the assigned targets with `cmake --build build/diff-review -j 1`. Ran validation with `EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe` only. All ten demos PASS at the default 256 × 256 size. Minimum-size runs also PASS for all assigned demos except the finding below.

- **ok** `es31-compute-indirect-and-to-render` — default and 128 × 128 PASS; image and ByRegion comparisons remain complete, and negative controls detect the changed models.
- **ok** `es31-framebuffer-no-attachments` — default PASS; the helper-invocation negative control still bites. Derivative precision is now an observation with the cited GLSL ES 3.10 §8.13 justification.
- **ok** `es31-image-load-store` — default and 64 × 64 PASS; texel probes follow the shader's integer mapping and the minified-cell negative control still bites.
- **ok** `es31-program-interface-query` — default and 64 × 64 PASS; probes remain inside their cells and the uniform-mutation control still bites.
- **ok** `es31-program-pipelines` — default and 64 × 64 PASS; quadrant probes stay off the split at the declared minimum and the uniform controls still bite.
- **ok** `es31-ssbo-std430` — default and 128 × 128 PASS; bar probes use the calculated bar spans and the std140 / mutation controls still bite.
- **BUG** [main.cpp](demos/es32/cube-map-array/main.cpp:475) — the new Skip condition admits 80 × 89, where the 8-pixel tile rows and 65-pixel scene satisfy `kMinTile` / `kMinScene`; however, validation requires `hits[k] >= 40` for every sphere at [main.cpp](demos/es32/cube-map-array/main.cpp:1649). At 80 × 89, llvmpipe reports 33, 32, 32, 34, and 39 hits, so the declared minimum returns FAIL (78/79 checks). Raise the minimum or scale the hit threshold so every admitted size can pass. The README's claim that the minimum only requires 8-pixel tile rows and 64 scene rows does not disclose this additional constraint.
- **ok** `es32-float-targets-and-indexed-blend` — default and 128 × 128 PASS; the RGBA8 and neighbour-blend negative controls still detect their models.
- **ok** `es32-geometry-shaders` — default and 200 × 200 PASS; coverage floors scale from the reference frame area, and all four raster and transform-feedback negative controls still bite. The total-output-components link outcome is correctly an observation, with the conditional wording quoted from ES 3.2 §11.3.4.4.
- **ok** `es32-texture-buffer` — default and 128 × 96 PASS; minimum-size run compares 7,829 scene probes, sees all 64 instances, and the stale-update negative control still bites.
