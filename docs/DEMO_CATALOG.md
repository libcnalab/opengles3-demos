# Demo catalog and learning path

Status of every executable: **[generated/DEMO_STATUS.md](generated/DEMO_STATUS.md)**.
Entries marked **planned** do not exist yet; they form the implementation backlog in the order a
reader should study them. Feature IDs refer to `inventory/features.csv`.

## Stage 1 — first pixels (ES 3.0 context, GLSL ES 3.00)

| # | executable | teaches | status |
|---|---|---|---|
| 1 | `es30-hello-triangle` | VBO, VAO, shader program, `glDrawArrays`, interpolation | implemented |
| 2 | `es20-clear-viewport-scissor` | clear values, viewport mapping, scissor, color mask | implemented |
| 3 | `es20-shader-compilation` | compile/link status, info logs, deferred deletion, source/precision queries | implemented |
| 4 | `es20-vertex-attributes` | attribute formats, normalization, stride/offset, constant attributes, bind/get location | implemented |
| 5 | `es20-indexed-drawing` | index buffers, index types, `glBufferSubData` | implemented |
| 6 | `es20-primitive-topology` | points/lines/strips/fans, `gl_PointSize`, line width range | implemented |
| 7 | `es20-uniforms` | scalar/vector/matrix/array/struct uniforms, active-uniform enumeration | implemented |

## Stage 2 — the fixed pipeline stages that remain (per-fragment operations)

| # | executable | teaches | status |
|---|---|---|---|
| 8 | `es20-culling-front-face` | winding, culling, `gl_FrontFacing` | implemented |
| 9 | `es20-depth-test` | depth functions, depth mask, depth range, polygon offset | implemented |
| 10 | `es20-stencil-test` | all stencil ops/functions, separate front/back state, masks | implemented |
| 11 | `es20-blending` | blend factors, equations, constant color, separate RGB/alpha | implemented |
| 12 | `es30-blend-min-max` | MIN/MAX equations | implemented inside `es20-blending` |

## Stage 3 — textures and framebuffers

| # | executable | teaches | status |
|---|---|---|---|
| 13 | `es20-textures-2d` | texture objects/units, filtering, wrapping, unpack alignment | implemented |
| 14 | `es20-mipmaps` | mip chains, `glGenerateMipmap`, completeness | implemented |
| 15 | `es20-cube-map` | cube faces and direction lookup | implemented |
| 16 | `es20-framebuffer-objects` | render to texture, renderbuffers, completeness (incl. failures) | implemented |
| 17 | `es20-copy-tex-image` | `glCopyTexImage2D/SubImage2D` | implemented as `es20-copy-and-read-pixels` |
| 18 | `es20-read-pixels` | readback formats, pack alignment, implementation read format | implemented as `es20-copy-and-read-pixels` |
| 19 | `es30-sampler-objects` | sampler vs texture state, two samplers on one texture | implemented |
| 20 | `es30-texture-3d-and-arrays` | 3D and 2D-array textures, layer addressing | planned |
| 21 | `es30-immutable-storage` | `glTexStorage*`, immutability errors, sized formats | planned |
| 22 | `es30-integer-float-textures` | isampler/usampler, float/half textures, filterability rules | planned |
| 23 | `es30-srgb` | sRGB textures and render targets | planned |
| 24 | `es30-texture-swizzle-lod` | swizzles, base/max level, min/max LOD | planned |
| 25 | `es30-etc2-eac` | required compressed formats | planned |
| 26 | `es30-pixel-buffer-objects` | PBO pack/unpack | planned |
| 27 | `es30-multiple-render-targets` | `glDrawBuffers`, fragment output locations | implemented |
| 28 | `es30-framebuffer-blit-msaa` | read/draw FBOs, blits, multisample resolve | planned |
| 29 | `es30-clear-buffer` | `glClearBuffer*` per attachment and type | implemented inside `es30-multiple-render-targets` |
| 30 | `es30-framebuffer-layers-invalidate` | layer attachments, invalidation | planned |
| 31 | `es30-depth-textures-shadow` | depth textures, comparison sampling, shadow-map basics | planned |

## Stage 4 — ES 3.0 data paths

| # | executable | teaches | status |
|---|---|---|---|
| 32 | `es30-vertex-array-objects` | VAO switching, element buffer as VAO state | implemented |
| 33 | `es30-integer-and-packed-attributes` | `glVertexAttribIPointer`, 2_10_10_10, half float | implemented inside `es30-vertex-array-objects` |
| 34 | `es30-instancing` | divisors, instanced draws, `gl_InstanceID` | implemented |
| 35 | `es30-draw-range-primitive-restart` | `glDrawRangeElements`, primitive restart | implemented inside `es30-instancing` |
| 36 | `es30-buffer-mapping` | map flags, explicit flush, copy, 64-bit queries | planned |
| 37 | `es30-uniform-buffer-objects` | std140, ranges, introspection, sharing | planned |
| 38 | `es30-transform-feedback` | capture, modes, objects, pause/resume, render from capture | planned |
| 39 | `es30-occlusion-queries` | query objects, availability | planned |
| 40 | `es30-sync-objects` | fences, client/server waits | planned |
| 41 | `es30-program-binary` | binary retrieval/reload + recompile fallback | planned |
| 42 | `es30-flat-centroid` | flat/centroid interpolation, provoking vertex | planned |
| 43 | `diagnostics-capabilities` | version, limits, extension report | implemented |

## Stage 5 — shading compositions (ES 3.0)

`es30-lighting` (directional/point/spot, Phong/Blinn-Phong), `es30-normal-mapping`,
`es30-post-processing` (blur, Sobel, gamma), `es30-fog`, `es30-deferred-shading`,
`es30-shadow-mapping`, `es30-particles-transform-feedback` — all **planned**.

## Stage 6 — ES 3.1

`es31-compute-basics`, `es31-compute-shared-memory`, `es31-compute-indirect-dispatch`,
`es31-ssbo-std430`, `es31-image-load-store`, `es31-atomic-counters`, `es31-atomic-memory`,
`es31-indirect-draw`, `es31-separate-vertex-format`, `es31-program-pipelines`,
`es31-program-interface-query`, `es31-texture-gather`, `es31-multisample-textures`,
`es31-stencil-texturing`, `es31-sample-mask`, `es31-framebuffer-no-attachments`,
`es31-compute-particles`, `es31-compute-image-filter` — all **planned**.

## Stage 7 — ES 3.2

`es32-geometry-basics`, `es32-geometry-layered`, `es32-tessellation-triangles`,
`es32-tessellation-quads-isolines`, `es32-tessellation-lod`, `es32-texture-buffer`,
`es32-cube-map-array`, `es32-multisample-array`, `es32-border-color`, `es32-astc`,
`es32-copy-image`, `es32-float-render-targets`, `es32-indexed-blend`, `es32-advanced-blend`,
`es32-sample-shading`, `es32-interpolate-at`, `es32-base-vertex`, `es32-debug-output`,
`es32-debug-groups-labels`, `es32-robustness` — all **planned**.

## Test executables (not demos)

`glsl-compile-tests` (implemented: GLSL ES positive/negative compile and link), `test-glsl300-builtins` (implemented), `glsl310-builtins` (planned), `tests/negative/*`
(`test-metagl-contracts` implemented; more planned).
