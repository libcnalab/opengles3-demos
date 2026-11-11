# Demo backlog (ordered)

The order follows `docs/DEMO_CATALOG.md`. Each line is one delegable task: directory → executable,
followed by feature IDs. Claim an item by moving it to "In progress" with the agent and branch.

## In progress
- GLSL language-rule coverage (Claude subagent); Codex from 22:05: items 18, 20, MSAA default framebuffer; then 62

## Done
All Stage 1–4 foundation demos listed as implemented in DEMO_CATALOG.md (merged 2026-11-11).
Batch 3 (2026-11-11): items 1, 2, 3, 4, 5, 6 (later), 7, 8, 9, 11, 12, 13, 15 merged.
Remaining ES 3.0 items for Codex: 18, 20.
Batch 4 (2026-11-11): ES 3.1 items 21-34 merged (14 demos). Item 35 (glsl310 built-ins) open.
Batch 5/6: 3D items 40, 41, 42, 43, 44, 45 and ES 3.2 items 50-61, es30-skybox-and-fog (3D), es31-compute-particles (3D), 55, 58, 61 merged.

## Next (ES 3.0)
1. `es30/texture-3d-and-arrays` → ES30.TEXTURE.3D, ES30.TEXTURE.2D_ARRAY, ES30.FBO.LAYER
2. `es30/immutable-storage` → ES30.TEXTURE.IMMUTABLE, ES30.TEXTURE.SIZED_FORMATS, ES30.TEXTURE.FORMAT_VALIDATION
3. `es30/integer-float-textures` → ES30.TEXTURE.INTEGER, ES30.TEXTURE.FLOAT, ES30.TEXTURE.SHARED_EXPONENT, ES30.TEXTURE.COMPLETENESS_INTEGER
4. `es30/srgb` → ES30.TEXTURE.SRGB, ES30.FBO.SRGB
5. `es30/texture-swizzle-lod` → ES30.TEXTURE.SWIZZLE, ES30.TEXTURE.LOD_PARAMS
6. `es30/etc2-eac` → ES30.TEXTURE.ETC2_EAC, ES30.TEXTURE.COMPRESSED_QUERY (encode tiny blocks by hand from the spec appendix)
7. `es30/pixel-buffer-objects` → ES30.BUFFER.PBO_PACK, ES30.BUFFER.PBO_UNPACK, ES30.TEXTURE.PIXEL_STORE
8. `es30/framebuffer-blit-msaa` → ES30.FBO.BLIT, ES30.FBO.MSAA_RENDERBUFFER, ES30.FBO.READ_DRAW, ES30.INTERNALFORMAT.QUERY
9. `es30/framebuffer-invalidate` → ES30.FBO.INVALIDATE, ES30.FBO.ATTACHMENT_QUERY
10. `es30/depth-textures-shadow` → ES30.TEXTURE.DEPTH, ES30.TEXTURE.DEPTH_STENCIL; GLSL300.FN.TEXTURE.SHADOW
11. `es30/buffer-mapping` → ES30.BUFFER.MAP_RANGE, ES30.BUFFER.MAP_FLAGS, ES30.BUFFER.FLUSH_EXPLICIT, ES30.BUFFER.COPY, ES30.BUFFER.PARAM64, ES30.BUFFER.INVALID_RANGES
12. `es30/uniform-buffer-objects` → ES30.UBO.*
13. `es30/transform-feedback` → ES30.TF.MODES, ES30.TF.OBJECTS, ES30.TF.PAUSE_RESUME, ES30.TF.RENDER_CAPTURED, ES30.TF.QUERY, ES30.TF.ERRORS
14. `es30/occlusion-queries` → ES30.QUERY.OCCLUSION, ES30.QUERY.OBJECTS
15. `es30/sync-objects` → ES30.SYNC.FENCE, ES30.SYNC.ERRORS, ES20.FLUSH_FINISH
16. `es30/program-binary` → ES30.PROGRAM.BINARY
17. `es30/flat-centroid` → ES30.RASTER.FLAT, ES30.RASTER.CENTROID, ES30.MULTISAMPLE.DEFAULT, ES20.SAMPLE_COVERAGE
18. `es20-foundations/state-queries-errors` → ES20.STATE.QUERIES, ES20.ERRORS.*, ES20.HINT, ES20.DITHER, ES30.STATE.INDEXED
19. `es30/glsl-texture-functions` → GLSL300.FN.TEXTURE.* (textureLod/Grad/Offset/texelFetch/textureSize/Proj), GLSL300.FN.DERIVATIVES

20. `tests/negative/driver-crash-readpixels-none` → run the Mesa crash case (READ_BUFFER NONE + integer attachment) in a child process; report crash as a known deviation, PASS when the driver raises INVALID_OPERATION

## ES 3.2 (min_es "3.2", GLSL 320 es; softpipe runs automatically and must report SKIP_UNSUPPORTED)
50. `es32/geometry-shaders` → ES32.GEOMETRY.BASIC, ES32.GEOMETRY.AMPLIFICATION, ES32.GEOMETRY.ADJACENCY; GLSL320.GEOMETRY.LAYOUT, GLSL320.GEOMETRY.BUILTINS, GLSL320.INTERFACE_BLOCKS_IO
51. `es32/layered-rendering` → ES32.GEOMETRY.LAYERED (gl_Layer, glFramebufferTexture)
52. `es32/tessellation-domains` → ES32.TESS.TRIANGLES, ES32.TESS.QUADS, ES32.TESS.ISOLINES, ES32.TESS.SPACING_WINDING, ES32.TESS.PATCH_PARAM; GLSL320.TESS.CONTROL, GLSL320.TESS.EVAL
53. `es32/tessellation-terrain` → dynamic LOD 3D terrain, ES32.TESS.LIMITS, ES32.PRIMITIVE_BOUNDING_BOX; GLSL320.PRIMITIVE_BOUNDING_BOX
54. `es32/texture-buffer` → ES32.TEXTURE.BUFFER; GLSL320.TEXTURE_BUFFER_FETCH (uses meta-gl BufferTarget::TextureBuffer, GetParameter::TextureBufferOffsetAlignment)
55. `es32/cube-map-array` → ES32.TEXTURE.CUBE_ARRAY; GLSL320.TYPES.NEW_SAMPLERS
56. `es32/multisample-array-and-border-color` → ES32.TEXTURE.MS_ARRAY, ES32.TEXTURE.BORDER_COLOR
57. `es32/astc-and-copy-image` → ES32.TEXTURE.ASTC, ES32.TEXTURE.COPY_IMAGE
58. `es32/float-targets-and-indexed-blend` → ES32.FBO.FLOAT_TARGETS, ES32.BLEND.INDEXED
59. `es32/advanced-blend-and-sample-shading` → ES32.BLEND.ADVANCED, ES32.SAMPLE.SHADING, ES32.SAMPLE.INTERPOLATION; GLSL320.SAMPLE_VARS, GLSL320.FN.INTERPOLATE_AT, GLSL320.BLEND_EQUATION_LAYOUT
60. `es32/base-vertex-and-robustness` → ES32.DRAW.BASE_VERTEX, ES32.ROBUST.RESET_STATUS, ES32.ROBUST.READN, ES32.ROBUST.BUFFER_ACCESS, ES32.CONTEXT.FLAGS
61. `es32/debug-output` → ES32.DEBUG.OUTPUT, ES32.DEBUG.GROUPS, ES32.DEBUG.LABELS, ES32.DEBUG.POINTER
62. `tests/glsl/glsl320-builtins` → GLSL320.FN.IMAGE_ATOMIC, GLSL320.FN.FMA, GLSL320.PRECISE (numeric, compute + SSBO)

## 3D compositions (prioritized by the owner on 2026-11-11: "I don't see 3D demos")
Real 3D scenes: perspective camera, depth buffer, animation in windowed mode (time = frame/60),
keyboard camera control (Demo::on_key), and deterministic validation at a fixed frame (CPU-projected
points, analytic lighting at chosen pixels). In progress (Claude subagents, batch 5):
40. `es20-foundations/transformations-camera` → ES20.CAMERA.MVP, ES20.COORDS.SYSTEMS, ES20.CLIP.VOLUME
41. `es30/lighting` → ES20.LIGHTING.SHADER (directional/point/spot, Phong vs Blinn-Phong, normal matrix)
42. `es30/normal-mapping` → normal + specular maps, tangent space
43. `es30/shadow-mapping` → ES30.TEXTURE.DEPTH, depth FBO, sampler2DShadow + PCF; GLSL300.FN.TEXTURE.SHADOW
44. `es30/deferred-shading` → G-buffer via MRT, light accumulation
45. `es30/post-processing` → render-to-texture 3D scene + blur, Sobel, gamma
Later: fog, particles (transform feedback), compute particles (ES 3.1), skybox/environment mapping.

## ES 3.1 (min_es "3.1", GLSL 310 es; cite ES 3.2 numbering)
21. `es31/compute-basics` → ES31.COMPUTE.DISPATCH, ES31.COMPUTE.LIMITS; GLSL310.COMPUTE.LAYOUT, GLSL310.COMPUTE.BUILTINS
22. `es31/compute-shared-memory` → ES31.COMPUTE.SHARED, ES31.BARRIER.MEMORY; GLSL310.SHARED, GLSL310.BARRIERS
23. `es31/ssbo-std430` → ES31.SSBO.BASIC, ES31.SSBO.STD430, ES31.SSBO.LIMITS; GLSL310.SSBO
24. `es31/atomics` → ES31.ATOMIC.COUNTERS, ES31.ATOMIC.MEMORY; GLSL310.ATOMIC_COUNTERS, GLSL310.FN.ATOMIC_MEMORY
25. `es31/image-load-store` → ES31.IMAGE.LOAD_STORE, ES31.IMAGE.ACCESS, ES31.IMAGE.QUERY, ES31.STATE.BOOLEANI; GLSL310.IMAGES, GLSL310.FN.IMAGE, GLSL310.MEMORY_QUALIFIERS
26. `es31/compute-indirect-and-to-render` → ES31.COMPUTE.INDIRECT, ES31.COMPUTE.TO_RENDER, ES31.BARRIER.BY_REGION
27. `es31/indirect-draw` → ES31.DRAW.INDIRECT
28. `es31/separate-vertex-format` → ES31.VERTEX.SEPARATE_FORMAT
29. `es31/program-pipelines` → ES31.PIPELINE.OBJECTS, ES31.PIPELINE.CREATE_SHADER_PROGRAM, ES31.PIPELINE.VALIDATE; GLSL310.SEPARABLE_IO
30. `es31/program-interface-query` → ES31.PROGRAM.INTERFACE_QUERY, ES31.UNIFORM.EXPLICIT_LOCATION; GLSL310.LAYOUT.UNIFORM_LOCATION, GLSL310.LAYOUT.BINDING
31. `es31/texture-gather` → ES31.TEXTURE.GATHER; GLSL310.FN.TEXTURE_GATHER
32. `es31/multisample-textures` → ES31.TEXTURE.MULTISAMPLE, ES31.SAMPLE_MASK; GLSL310.FN.TEXTURE_MS
33. `es31/stencil-texturing-and-level-queries` → ES31.TEXTURE.STENCIL_TEXTURING, ES31.TEXTURE.LEVEL_QUERY
34. `es31/framebuffer-no-attachments` → ES31.FBO.NO_ATTACHMENTS; GLSL310.EARLY_FRAGMENT_TESTS, GLSL310.HELPER_INVOCATION
35. `tests/glsl/glsl310-builtins` → GLSL310.FN.INTEGER, GLSL310.FN.FLOAT_EXT, GLSL310.FN.PACKING_8, GLSL310.ARRAYS_OF_ARRAYS (numeric, via compute + SSBO)

## Then: ES 3.2 (see DEMO_CATALOG.md stage 7)
