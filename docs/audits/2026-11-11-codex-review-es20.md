# Educational review report

Reviewed the assigned group against `docs/DEMO_CATALOG.md`, each README, `main.cpp`, `demo.json`, and shader sources. All checked shader files use `#version 300 es`; metadata names and `min_es: 3.0` match the catalog. No builds or file changes were made beyond this report.

## diagnostics-capabilities — good

The opening explains both the learner-facing report and its self-check role; the expected four-bar image gives a useful visual anchor. The version-dependent limit rules and renderer caveat are unusually clear. The catalog presents this after the foundations, which is appropriate for a diagnostic reference.

## es20-blending — good

The first paragraph defines source/destination colors and names the concepts taught. The contact sheet and compositing scene are explained concretely, and the pitfalls address alpha and draw order. The README, implementation, metadata, and GLSL 3.00 shaders agree.

## es20-clear-viewport-scissor — good

The opening identifies each operation and foregrounds the important distinction between clear, scissor, and viewport. The labeled panel map makes the extensive validation approachable; indirect depth/stencil observation is explained accurately. It is correctly the catalog's first foundation demo.

## es20-copy-and-read-pixels — good

The purpose clearly distinguishes client-memory reads from texture copies, and explains lower-left origin and pixel layout. The unique test chart makes the checks intuitive; implementation and metadata cover the described operations. The catalog places it after texturing and FBOs, giving useful prerequisites.

## es20-cube-map — good

The purpose says what a cube map stores and how sampling addresses it, while naming the face-selection rule and completeness. The face/texel image explanation supports the lesson, and shader versions and metadata align.

## es20-culling-front-face — good

The opening correctly starts from signed area in window coordinates and connects it to both culling and `gl_FrontFacing`. The rule sequence and examples make state effects easy to follow; README and implementation agree.

## es20-depth-test — good

The opening distinguishes several commonly confused causes of depth errors and tells the learner how the demo exposes depth through color. The isolated experiments and spec references are clear and consistent with the implementation.

## es20-framebuffer-objects — good

The first paragraph defines FBOs through attachments and contrasts textures with renderbuffers. The README explains completeness failures and visible use of each attachment; the coverage and metadata match the implementation.

## es20-indexed-drawing — good

The entry explains the core purpose of indices before listing buffer behavior and errors. The concrete mesh dimensions and visual layout help orient the learner; shader version and metadata agree.

## es20-mipmaps — good

The first paragraph explains both independent filter decisions and level of detail. Hand-built levels make the sampling behavior inspectable, and the README carefully distinguishes guaranteed sampling behavior from implementation-dependent filtering. The catalog places it after 2D textures.

## es20-primitive-topology — good

The opening explains that draw mode changes the interpretation of the same vertex sequence, then previews points, lines, and triangles. It explicitly calls out implementation-dependent point/line ranges and validates against spec-derived behavior. The catalog provides indexed drawing as a nearby prerequisite.

## es20-shader-compilation — good

The entry highlights non-obvious shader/program lifetime and error behavior, and its simple two-program image is well motivated. The table connects each call to the rule being taught; the implementation and shader version agree.

## es20-state-queries-errors — POLISH

The opening accurately frames queries and GL errors, and the dashboard is described. However, the README says the experiments run in `init()` (line 24), while `main.cpp` runs the queries there and `validate()` records the results and checks the image; a short pointer to the query sequence in `init()` would help readers find the technique in a large implementation. Consider adding a “read this first” pointer to the relevant `init()` block and distinguishing setup/query execution from result validation in the image description.

## es20-stencil-test — good

The purpose names the main stencil concepts, including value/write masks and the three outcomes. The readout method is an especially effective explanation of why the screen can reveal a buffer that `glReadPixels` cannot return. Metadata and shaders agree.

## es20-textures-2d — good

The opening states the scope and the CPU sampling reference, while the cell map gives an immediate guide to the image. Its discussion of units, sampler uniforms, filtering, and formats matches the implementation and metadata.

## es20-transformations-camera — good

The entry establishes this as the 3D path and presents the coordinate chain immediately. The perspective/orthographic comparison and near-plane cut are strong visual teaching choices; controls and shader versions are documented and match the implementation.

## es20-uniforms — good

The opening clearly states the exact-type rule and breadth of uniform variants. The cell-based shader check and readback/enumeration checks are easy to understand, and the softpipe caveat is explicit. Metadata and shader versions match.

## es20-vertex-attributes — POLISH

The purpose is a clear and unusually complete entry to generic vertex attributes. The README accurately documents conversions, VAO 0 client arrays, and queries. The CPU reference and error cases occupy substantial implementation space; add a short “read this first” pointer to the attribute setup/draw path in `main.cpp` so learners can locate the GL technique before exploring validation. The catalog's earlier indexed-drawing entry is a suitable prerequisite.
