# Educational review: ES 3.0 group B

Reviewed the 16 assigned demos against their READMEs, `main.cpp`, shaders and `demo.json`. No files were changed except this report.

## `multisample-default-framebuffer`

**CONFUSING** — The README is technically careful, but its 229-line treatment (especially validation details at lines 85–184) obscures the learner's first path through the demo. Add a short “Read this first” sequence pointing to the sample request, framebuffer queries, cube draw, and resolve; keep the exhaustive conformance analysis as reference material. The opening and source comments are accurate and useful.

## `normal-mapping`

**good** — The opening explains the visual effect and distinguishes the tangent-space normal and specular maps; the scene description and API path make the technique discoverable. Shader comments explain the tangent-space work rather than merely narrating syntax.

## `occlusion-queries`

**good** — Clearly introduces the query result, exact versus conservative variants, then connects them to a visible city-block culling example. Good conceptual ordering from basic query semantics to application use.

## `pixel-buffer-objects`

**good** — Strong motivation contrasts client pointers with buffer offsets and immediately names pack/unpack targets. The concise central-call table is an effective entry point for learners.

## `post-processing`

**CONFUSING** — The opening promises five different effects and CPU recomputation of every pass, while the very broad scene description follows immediately. Add a brief pass-order diagram/list with input and output target for each stage, and identify the simplest pass to inspect first; this helps a learner trace the GL path through the many FBOs. No specific factual mismatch found in the opening checked against the implementation.

## `program-binary`

**good** — Purpose, portability caveat, and left/right comparison clearly explain both why binaries matter and what the demo demonstrates. The central-call snippet starts with the critical retrievable-hint-before-link ordering.

## `sampler-objects`

**good** — The opening contrasts texture-owned state with sampler-owned state and directly explains texture-unit precedence. This is a clear mental model before the implementation details.

## `shadow-mapping`

**CONFUSING** — The README's “built the way every game engine builds them” at line 7 overstates one common approach; soften to “a common shadow-map approach.” The two-pass opening is a strong teaching structure. Explain near the shadow lookup that depth comparison and filtering are separate steps (and that linear filtering averages comparison results), since the opening introduces both together.

## `skybox-and-fog`

**good** — The first paragraph previews the three cube-map uses and the depth trick, then distinguishes shader fog from nonexistent fixed-function ES fog. The visual concepts are tied to specific shader operations.

## `srgb`

**WRONG** — README line 5 says 8-bit images are stored in sRGB. Many images are linear or use other encodings; sRGB is common, not universal. Revise to “Many 8-bit color images use sRGB encoding.” The subsequent separation of texture decoding and framebuffer encoding is clear and technically specific.

## `sync-objects`

**CONFUSING** — The opening says the two-buffer ring is rewritten “without synchronization,” then explains waiting on fences; this reads as contradictory. Say “without implicit synchronization” or “without relying on the driver to wait during mapping.” The explanation of `MAP_UNSYNCHRONIZED_BIT` and application-owned fence waits is otherwise a useful central example.

## `texture-3d-and-arrays`

**CONFUSING** — The explanation that an array layer coordinate is rounded to the nearest layer and clamped at README lines 13–14 is likely to mislead: GLSL ES 3.00 array texture lookup uses the layer coordinate as a layer selection with integer conversion semantics; clarify the spec rule and avoid describing it as ordinary normalized-coordinate filtering. The 3D versus array contrast is otherwise an excellent entry.

## `texture-swizzle-lod`

**good** — The purpose makes the shared theme explicit and points to the prerequisite `es20-mipmaps` demo while delimiting what this demo adds. The contact-sheet description provides a practical route through the examples.

## `transform-feedback`

**good** — The first paragraph explains when transform feedback is useful and why it matters before ES 3.1 compute; the four-capture overview prepares the reader for the implementation. The explicit second-pass framing is especially helpful.

## `uniform-buffer-objects`

**CONFUSING** — The README is an unusually comprehensive reference, but a learner meets layout calculation, introspection, binding, updates, limits, errors, and validation in one long sequence. Add a minimal “read this first” route through one std140 block upload and draw, then point to the later sections for packed layout and error cases. The conceptual distinction between block-to-binding-point and binding-point-to-buffer is clearly stated.

## `vertex-array-objects`

**good** — The purpose leads with the VAO state model and immediately previews the four experiments. Its image table, shader explanation, and pitfalls reinforce the same key distinction between captured `ARRAY_BUFFER` and VAO-owned element binding.
