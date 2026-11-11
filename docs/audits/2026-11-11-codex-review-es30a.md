# Educational review report

Scope: the 15 assigned `demos/es30` demos. Review is based on each README, `main.cpp`, `demo.json`, and shader sources. No build or files beyond this report were changed.

## buffer-mapping

- **CONFUSING** — `README.md:9-10` says mapping “avoids a copy” and describes streaming while earlier draws are still reading. Mapping does not guarantee zero-copy, and the ordinary map waits for pending operations; `MAP_UNSYNCHRONIZED_BIT` is safe only when the application has established that the range is not in use. Reword this as a possible implementation benefit and direct readers to the synchronization caveat at lines 52-54 and 97-98.

## deferred-shading

- **CONFUSING** — `README.md:8-9` says lighting cost no longer depends on how many objects overlap a pixel. Deferred shading removes repeated *lighting* work from opaque overdraw, but geometry rasterization and G-buffer writes still depend on the objects and fragments that overlap. Qualify the claim to make clear which cost is reduced.
- **POLISH** — `main.cpp` starts with substantial geometry and CPU reference machinery before the pass setup. Add a short “read this first” pointer in `README.md` to the geometry-pass and lighting-pass sections in `render()` so learners can find the visible GL path without traversing validation helpers first.

## depth-textures

Good. The format table, image description, and explanation of the separate depth comparison and stencil paths give a learner a useful sequence from storage to sampling.

## etc2-eac

Good. It explains why the data is hand-encoded, pairs it with a separate decoder, and points directly to the relevant appendix and image-update calls.

## flat-centroid

Good. The three ideas are introduced before the panel guide, and the README explicitly relates the multisample behavior to observable checks and documented driver limits.

## framebuffer-blit-msaa

Good. The README separates read and draw framebuffer roles, explains resolve and blit rules, and calls out the ES 3.1-only multisample shader path.

## framebuffer-invalidate

Good. It carefully distinguishes undefined invalidated contents from guarantees the demo can validate, a key learning point for this API.

## glsl-texture-functions

- **POLISH** — The opening claim at `README.md:14-18` is a large coverage promise. Add a link to the shader/source section that enumerates the tested overload families, and clarify that derivative-dependent forms are exercised only in stages where GLSL ES permits them. The distinction is present later, but is easy to miss before the reader encounters the exhaustive list.

## hello-triangle

Good. The opening says exactly what the learner should follow, and the call-by-call table makes the pipeline easy to locate.

## immutable-storage

Good. The README makes the important distinction between immutable dimensions/format and mutable contents/parameters, then gives concrete format-transfer examples.

## instancing

Good. The purpose and cell guide make both instanced attributes and the related primitive-restart section discoverable, with a CPU model explaining how the output is checked.

## integer-float-textures

Good. The README clearly contrasts integer, float, and packed formats and explains the sampler and filtering constraints that determine whether each texture can be sampled.

## interactive-3d-scene

Good. The controls are listed before the pass walkthrough, and each pass points back to the prerequisite demo that teaches its technique.

## lighting

Good. The paired scene comparisons isolate the lighting concepts, and the README connects the shader equations to specific controls and CPU-checked probes.

## multiple-render-targets

Good. The output-to-attachment table is a clear entry point, and the subsequent sections explain routing, typed clears, integer reads, and validation in a useful order.
