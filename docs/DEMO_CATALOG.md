# Demo catalog and learning path

Every executable in the suite, in the order a reader should study them (the `order` field of each
`demo.json`). Live status per driver and ES version: **[generated/DEMO_STATUS.md](generated/DEMO_STATUS.md)**.
Executables marked ★ are 3D scenes, good starting points to see what the API pieces add up to. The capstone
`es30-interactive-3d-scene` combines them in one explorable scene (mouse look, WASD, technique toggles 1-6).

## Short path for a first reading

The full catalog below is complete but long. A reader new to OpenGL ES can take this route first and return to
the catalog for depth:

1. [`es30-hello-triangle`](../demos/es30/hello-triangle/README.md): context, shaders, a buffer, a VAO, one draw call.
2. [`es20-vertex-attributes`](../demos/es20-foundations/vertex-attributes/README.md) and
   [`es30-vertex-array-objects`](../demos/es30/vertex-array-objects/README.md): how vertex data reaches the shader.
3. [`es20-shader-compilation`](../demos/es20-foundations/shader-compilation/README.md) and
   [`es20-uniforms`](../demos/es20-foundations/uniforms/README.md): shader objects, programs, uniforms.
4. [`es20-transformations-camera`](../demos/es20-foundations/transformations-camera/README.md) ★: matrices and a 3D camera.
5. [`es20-textures-2d`](../demos/es20-foundations/textures-2d/README.md), then [`es30-lighting`](../demos/es30/lighting/README.md) ★.
6. [`es20-framebuffer-objects`](../demos/es20-foundations/framebuffer-objects/README.md): rendering into textures.
7. [`es30-instancing`](../demos/es30/instancing/README.md) and [`es30-transform-feedback`](../demos/es30/transform-feedback/README.md).
8. [`es31-compute-basics`](../demos/es31/compute-basics/README.md), [`es31-ssbo-std430`](../demos/es31/ssbo-std430/README.md),
   [`es31-compute-particles`](../demos/es31/compute-particles/README.md) ★ (ES 3.1).
9. [`es32-geometry-shaders`](../demos/es32/geometry-shaders/README.md) ★ and
   [`es32-tessellation-terrain`](../demos/es32/tessellation-terrain/README.md) ★ (ES 3.2).
10. [`es30-interactive-3d-scene`](../demos/es30/interactive-3d-scene/README.md) ★: the explorable capstone.

Every README has a section on how the demo is validated. Run it with `--headless --validate` and each check prints one
line (`[ok]` or `[FAIL]` with the measured and expected values).

## Stage 1–3: ES 2.0 foundations on an ES 3.x context

| executable | min ES | what it teaches |
|---|---|---|
| [`es20-clear-viewport-scissor`](../demos/es20-foundations/clear-viewport-scissor/README.md) | 3.0 | Clear, viewport and scissor |
| [`es20-shader-compilation`](../demos/es20-foundations/shader-compilation/README.md) | 3.0 | Shader and program objects |
| [`es20-vertex-attributes`](../demos/es20-foundations/vertex-attributes/README.md) | 3.0 | Vertex attributes |
| [`es20-indexed-drawing`](../demos/es20-foundations/indexed-drawing/README.md) | 3.0 | Indexed drawing |
| [`es20-primitive-topology`](../demos/es20-foundations/primitive-topology/README.md) | 3.0 | Primitive topology |
| [`es20-uniforms`](../demos/es20-foundations/uniforms/README.md) | 3.0 | Uniform variables |
| [`es20-culling-front-face`](../demos/es20-foundations/culling-front-face/README.md) | 3.0 | Face culling and front-face winding |
| [`es20-depth-test`](../demos/es20-foundations/depth-test/README.md) | 3.0 | Depth buffer test |
| [`es20-stencil-test`](../demos/es20-foundations/stencil-test/README.md) | 3.0 | Stencil buffer test |
| [`es20-blending`](../demos/es20-foundations/blending/README.md) | 3.0 | Blending |
| [`es20-textures-2d`](../demos/es20-foundations/textures-2d/README.md) | 3.0 | 2D textures: filtering, wrapping, formats, units |
| [`es20-framebuffer-objects`](../demos/es20-foundations/framebuffer-objects/README.md) | 3.0 | Framebuffer objects: render to texture, renderbuffers, completeness |
| [`es20-mipmaps`](../demos/es20-foundations/mipmaps/README.md) | 3.0 | Mipmaps: level of detail, the four mipmap filters, glGenerateMipmap, completeness |
| [`es20-copy-and-read-pixels`](../demos/es20-foundations/copy-and-read-pixels/README.md) | 3.0 | Copy and read pixels: glReadPixels, glCopyTexImage2D, glCopyTexSubImage2D |
| [`es20-cube-map`](../demos/es20-foundations/cube-map/README.md) | 3.0 | Cube map textures: face selection (Table 3.21) and completeness |
| [`es20-transformations-camera`](../demos/es20-foundations/transformations-camera/README.md) ★ | 3.0 | Transformations and camera |
| [`es20-state-queries-errors`](../demos/es20-foundations/state-queries-errors/README.md) | 3.0 | State queries, conversions and GL errors |

## Stage 4: OpenGL ES 3.0

| executable | min ES | what it teaches |
|---|---|---|
| [`es30-interactive-3d-scene`](../demos/es30/interactive-3d-scene/README.md) ★ | 3.0 | Interactive 3D scene (capstone: camera, lighting, textures, normal mapping, shadows, instancing, skybox, fog) |
| [`es30-hello-triangle`](../demos/es30/hello-triangle/README.md) | 3.0 | Hello Triangle |
| [`es30-vertex-array-objects`](../demos/es30/vertex-array-objects/README.md) | 3.0 | Vertex array objects, integer and packed attributes |
| [`es30-instancing`](../demos/es30/instancing/README.md) | 3.0 | Instancing, divisors, primitive restart |
| [`es30-multiple-render-targets`](../demos/es30/multiple-render-targets/README.md) | 3.0 | Multiple render targets: glDrawBuffers, glClearBuffer*, integer targets |
| [`es30-sampler-objects`](../demos/es30/sampler-objects/README.md) | 3.0 | Sampler objects: sampling state separate from textures |
| [`es30-integer-float-textures`](../demos/es30/integer-float-textures/README.md) | 3.0 | Integer, float and packed-float textures |
| [`es30-texture-3d-and-arrays`](../demos/es30/texture-3d-and-arrays/README.md) | 3.0 | 3D textures and 2D array textures |
| [`es30-texture-swizzle-lod`](../demos/es30/texture-swizzle-lod/README.md) | 3.0 | Texture swizzle and shader-chosen level of detail |
| [`es30-immutable-storage`](../demos/es30/immutable-storage/README.md) | 3.0 | Immutable texture storage and sized formats |
| [`es30-pixel-buffer-objects`](../demos/es30/pixel-buffer-objects/README.md) | 3.0 | Pixel buffer objects: texture uploads and read-backs through buffer objects |
| [`es30-srgb`](../demos/es30/srgb/README.md) | 3.0 | sRGB textures and sRGB render targets |
| [`es30-etc2-eac`](../demos/es30/etc2-eac/README.md) | 3.0 | ETC2 / EAC compressed textures |
| [`es30-glsl-texture-functions`](../demos/es30/glsl-texture-functions/README.md) | 3.0 | GLSL ES 3.00 texture lookup and derivative functions |
| [`es30-framebuffer-blit-msaa`](../demos/es30/framebuffer-blit-msaa/README.md) | 3.0 | Framebuffer blits and multisample resolve |
| [`es30-flat-centroid`](../demos/es30/flat-centroid/README.md) | 3.0 | Flat shading, centroid and sample coverage |
| [`es30-multisample-default-framebuffer`](../demos/es30/multisample-default-framebuffer/README.md) | 3.0 | Multisampled default framebuffer |
| [`es30-framebuffer-invalidate`](../demos/es30/framebuffer-invalidate/README.md) | 3.0 | Framebuffer invalidation and attachment queries |
| [`es30-buffer-mapping`](../demos/es30/buffer-mapping/README.md) | 3.0 | Buffer mapping and copying |
| [`es30-uniform-buffer-objects`](../demos/es30/uniform-buffer-objects/README.md) | 3.0 | Uniform buffer objects |
| [`es30-transform-feedback`](../demos/es30/transform-feedback/README.md) | 3.0 | Transform feedback: capture, objects, pause/resume, queries |
| [`es30-program-binary`](../demos/es30/program-binary/README.md) | 3.0 | Program binaries |
| [`es30-sync-objects`](../demos/es30/sync-objects/README.md) | 3.0 | Sync objects: fences, client and server waits, flush and finish |
| [`es30-normal-mapping`](../demos/es30/normal-mapping/README.md) ★ | 3.0 | Normal mapping: tangent space, specular map, mip chains |
| [`es30-deferred-shading`](../demos/es30/deferred-shading/README.md) ★ | 3.0 | Deferred shading: G-buffer (MRT) + 48 moving point lights |
| [`es30-lighting`](../demos/es30/lighting/README.md) ★ | 3.0 | Lighting: directional, point, spot; Phong vs Blinn-Phong |
| [`es30-skybox-and-fog`](../demos/es30/skybox-and-fog/README.md) ★ | 3.0 | Skybox, environment mapping and fog |
| [`es30-occlusion-queries`](../demos/es30/occlusion-queries/README.md) ★ | 3.0 | Occlusion queries: culling hidden towers behind a city block |
| [`es30-shadow-mapping`](../demos/es30/shadow-mapping/README.md) ★ | 3.0 | Shadow Mapping |
| [`es30-depth-textures`](../demos/es30/depth-textures/README.md) | 3.0 | Depth Textures |
| [`es30-post-processing`](../demos/es30/post-processing/README.md) ★ | 3.0 | Post-processing: gamma, blur, Sobel, bloom, depth of field |

## Stage 5: OpenGL ES 3.1

| executable | min ES | what it teaches |
|---|---|---|
| [`es31-compute-basics`](../demos/es31/compute-basics/README.md) | 3.1 | Compute shaders: dispatch, built-ins, limits |
| [`es31-compute-shared-memory`](../demos/es31/compute-shared-memory/README.md) | 3.1 | Compute shaders: shared memory and barriers |
| [`es31-ssbo-std430`](../demos/es31/ssbo-std430/README.md) | 3.1 | Shader storage buffers and std430 |
| [`es31-atomics`](../demos/es31/atomics/README.md) | 3.1 | Atomic counters and atomic memory functions |
| [`es31-image-load-store`](../demos/es31/image-load-store/README.md) | 3.1 | Image load/store |
| [`es31-compute-indirect-and-to-render`](../demos/es31/compute-indirect-and-to-render/README.md) | 3.1 | Indirect dispatch and compute to render |
| [`es31-compute-particles`](../demos/es31/compute-particles/README.md) ★ | 3.1 | Compute particles: an SSBO fountain simulated on the GPU |
| [`es31-stencil-texturing-and-level-queries`](../demos/es31/stencil-texturing-and-level-queries/README.md) | 3.1 | Stencil texturing and texture level queries |
| [`es31-texture-gather`](../demos/es31/texture-gather/README.md) | 3.1 | textureGather and textureGatherOffset |
| [`es31-framebuffer-no-attachments`](../demos/es31/framebuffer-no-attachments/README.md) | 3.1 | Framebuffer without attachments |
| [`es31-multisample-textures`](../demos/es31/multisample-textures/README.md) | 3.1 | Multisample textures and the sample mask |
| [`es31-indirect-draw`](../demos/es31/indirect-draw/README.md) | 3.1 | Indirect drawing |
| [`es31-separate-vertex-format`](../demos/es31/separate-vertex-format/README.md) | 3.1 | Separate vertex format and vertex buffer bindings |
| [`es31-program-pipelines`](../demos/es31/program-pipelines/README.md) | 3.1 | Separable programs and program pipelines |
| [`es31-program-interface-query`](../demos/es31/program-interface-query/README.md) | 3.1 | Program interface queries |

## Stage 6: OpenGL ES 3.2

| executable | min ES | what it teaches |
|---|---|---|
| [`es32-float-targets-and-indexed-blend`](../demos/es32/float-targets-and-indexed-blend/README.md) | 3.2 | Float render targets and indexed blending |
| [`es32-geometry-shaders`](../demos/es32/geometry-shaders/README.md) ★ | 3.2 | Geometry shaders: amplification, adjacency, invocations |
| [`es32-layered-rendering`](../demos/es32/layered-rendering/README.md) ★ | 3.2 | Layered rendering: gl_Layer, cube map in one pass |
| [`es32-debug-output`](../demos/es32/debug-output/README.md) | 3.2 | Debug output: messages, groups and labels |
| [`es32-tessellation-domains`](../demos/es32/tessellation-domains/README.md) | 3.2 | Tessellation domains, spacings and winding |
| [`es32-tessellation-terrain`](../demos/es32/tessellation-terrain/README.md) ★ | 3.2 | Tessellated terrain with dynamic LOD |
| [`es32-base-vertex-and-robustness`](../demos/es32/base-vertex-and-robustness/README.md) | 3.2 | Base vertex drawing and robustness |
| [`es32-texture-buffer`](../demos/es32/texture-buffer/README.md) | 3.2 | Buffer textures |
| [`es32-cube-map-array`](../demos/es32/cube-map-array/README.md) | 3.2 | Cube map array textures |
| [`es32-multisample-array-and-border-color`](../demos/es32/multisample-array-and-border-color/README.md) | 3.2 | Multisample array textures and the border color |
| [`es32-astc-and-copy-image`](../demos/es32/astc-and-copy-image/README.md) | 3.2 | ASTC textures and glCopyImageSubData |
| [`es32-advanced-blend-and-sample-shading`](../demos/es32/advanced-blend-and-sample-shading/README.md) | 3.2 | Advanced blend equations and sample shading |

## Diagnostics

| executable | min ES | what it teaches |
|---|---|---|
| [`diagnostics-capabilities`](../demos/diagnostics/capabilities/README.md) | 3.0 | Capabilities, limits and extensions |

## Test executables (not demos)

| executable | min ES | what it checks |
|---|---|---|
| [`test-glsl-language`](../tests/glsl/glsl-language/README.md) | 3.0 | GLSL ES language rules at run time |
| [`test-glsl300-builtins`](../tests/glsl/glsl300-builtins/README.md) | 3.0 | GLSL ES 3.00 built-in functions (numeric) |
| [`test-glsl310-builtins`](../tests/glsl/glsl310-builtins/README.md) | 3.1 | GLSL ES 3.10 built-in functions and arrays of arrays |
| [`test-glsl320-builtins`](../tests/glsl/glsl320-builtins/README.md) | 3.2 | GLSL ES 3.20 image atomics, fma and precise |
| [`test-driver-crash-readpixels-none`](../tests/negative/driver-crash-readpixels-none/README.md) | 3.0 | glReadPixels with READ_BUFFER = NONE (crash-isolated) |
| [`test-metagl-contracts`](../tests/negative/metagl-contracts/README.md) | 3.0 | meta-gl contracts |
| [`glsl-compile-tests`](../shader-tests/README.md) | 3.0 | GLSL ES compile and link tests |
