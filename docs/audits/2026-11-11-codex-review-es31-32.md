# Educational review report

Static review of the assigned ES 3.1 and ES 3.2 demos. No builds or runtime checks were performed. I compared README descriptions and stated validation with the corresponding `main.cpp`, `demo.json`, and shader sources.

## ES 3.1

- **atomics — good.** The opening identifies the two atomic families, and the image guide and order-independent checks explain both what to observe and why invocation order must not be asserted. Shader and API topics are easy to locate by filename and section.
- **compute-basics — good.** Starts with the dispatch/work-group mental model, explains the panels, and points directly to built-ins and barrier calls. The CPU model is explicitly framed as a validation reference.
- **compute-indirect-and-to-render — good.** The three use cases are distinguished immediately, and the README explains the buffer roles and consumer-specific barriers that are central to understanding the code.
- **compute-particles — good.** The GPU-resident update-and-draw flow is clear from the first paragraph; the README covers layout, synchronization, controls, and the visible particle scene.
- **compute-shared-memory — good.** Gives the work-group cooperation model before describing reduction and scan, and explains the difference between execution and memory barriers where it matters.
- **framebuffer-no-attachments — good.** Clearly states why attachmentless rasterization is useful and what fragment side effects are measured. The README makes the expected and intentionally unconstrained cases explicit.
- **image-load-store — good.** Introduces image units in contrast to sampling/render targets, then explains access qualifiers, formats, and synchronization. Shader examples are identified by source filename.
- **indirect-draw — good.** The README makes the CPU-written versus GPU-written command path understandable and explains the indirect command layout and required visibility barrier.
- **multisample-textures — good.** Clearly distinguishes per-sample access from resolved sampling and explains what the image and sample-mask checks demonstrate.
- **program-interface-query — good.** The purpose and resource-query workflow are explicit, with the queried interfaces and their relation to linked shader resources described in the README.
- **program-pipelines — good.** Explains separable programs and pipeline assembly before the examples, with useful attention to interface matching and uniform ownership.
- **separate-vertex-format — good.** The distinction between attribute format and buffer binding is presented directly, including integer attributes and the independent binding/offset state.
- **ssbo-std430 — good.** Gives a useful contrast between `std430` and `std140`, and the layout/runtime-array examples connect queried offsets to the rendered output.
- **stencil-texturing-and-level-queries — good.** Identifies the otherwise easy-to-miss stencil sampling path and the level-query distinctions; shader filenames and expected visual result are documented.
- **texture-gather — good.** Explains that gather returns the four neighboring texels for one selected component and describes how offsets change the footprint.

## ES 3.2

- **advanced-blend-and-sample-shading — good.** Separates advanced blend equations from per-sample execution and describes the output panels and sample-level validation.
- **astc-and-copy-image — good.** The README introduces compressed ASTC storage and image-to-image copying as separate concepts, then describes their concrete visual and readback checks.
- **base-vertex-and-robustness — good.** Clearly frames base-vertex indexing and robust buffer access as separate ES 3.2 topics and explains the expected bounds behavior.
- **cube-map-array — good.** Describes the cube-array layer convention and sampling path, with the shader-side indexing and scene shown in the README.
- **debug-output — good.** Explains message control, callbacks, groups, and labels as a debugging workflow and documents the observable output.
- **float-targets-and-indexed-blend — good.** The purpose distinguishes floating-point render targets from per-draw-buffer blend state; the visual panels give a concrete guide to what each contributes.
- **geometry-shaders — good.** The README establishes the new pipeline stage, then explains the demonstrations and the relevant output ordering/primitive rules. The geometry shader source names are easy to find.
- **layered-rendering — good.** The opening explains whole-level attachments and `gl_Layer`; its four-panel description gives concrete examples of cube, array, and 3D texture rendering.
- **multisample-array-and-border-color — good.** Treats multisample arrays and border color as two explicit topics, explains per-sample access, and documents format-dependent conversion and clamping.
- **tessellation-domains — good.** Clearly identifies the three domains and spacing modes, and explains why transform feedback plus a CPU model is used to teach the primitive generator.
- **tessellation-terrain — good.** Leads with the terrain pipeline and explains edge-consistent LOD, displacement, and bounding boxes, with controls and validation details available in the README.
- **texture-buffer — good.** The opening distinguishes buffer textures from uniforms and ordinary textures, then connects format rules and range attachment to a scene using per-instance data.

## Cross-demo notes

The assigned READMEs consistently provide a clear purpose, expected result, API/spec references, and validation explanation. The main GL path remains discoverable through named shader files and central-call sections even in validation-heavy examples. No specific README-to-source or README-to-`demo.json` factual mismatch was established by this static pass. Prerequisite/next-demo links are not consistently explicit; the catalog supplies the overall order, but adding a short “Read next” pointer would help learners in the more specialized ES 3.1/3.2 topics.
