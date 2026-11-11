# Demo backlog (ordered)

The order follows `docs/DEMO_CATALOG.md`. Each line is one delegable task: directory → executable,
followed by feature IDs. Claim an item by moving it to "In progress" with the agent and branch.

## In progress
(none)

## Done
All Stage 1–4 foundation demos listed as implemented in DEMO_CATALOG.md (merged 2026-11-11).

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

## Then: compositions (ES 3.0)
lighting, normal-mapping, post-processing (blur/Sobel/gamma), fog, deferred-shading, shadow-mapping, particles-transform-feedback.

## Then: ES 3.1 and ES 3.2 (see DEMO_CATALOG.md stages 6–7)
