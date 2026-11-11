# Independent diff review

Reviewed `git diff 5bd59a6..HEAD` for the assigned directories. Built the changed targets with `-j 1` against `/rv/data/development/github.com/libcna/meta-gl`, and ran validation with `EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe`.

## Findings

- **BUG — `demos/es20-foundations/cube-map/README.md:36-38`, `main.cpp:306-308`.** The new README says “every size passes” and “There is no minimum size,” but `tile_w = width / 8` and `tile_h = min(tile_w, height / 4)` produce zero or one-pixel face tiles on small surfaces, with no `Skip` guard. At 8×32, six face-tile checks and the `glTexSubImage2D` negative control fail with 0 pixels compared/changed; at 16×16 the skybox multi-face check fails. This conflicts with the documented no-minimum claim and leaves checks non-vacuous only at larger sizes such as the documented 32×32 and 16×400 examples. Add an honest minimum/skip or make the checks meaningful at these dimensions, and update the README.

- **BUG — `demos/es20-foundations/mipmaps/main.cpp:652-654`, `README.md:101-102`.** The new surface-derived layout divides dimensions into 8 columns and 15 rows without a minimum-size check or `Skip`. Small windows create zero-pixel cells: 1×1 fails 19 of 64 checks, including a negative control reporting 0 of 0 pixels rejected; 16×16 also fails three checks. The README describes the 32×17 default cell but does not state a supported minimum. Guard and document the minimum needed for meaningful per-cell and negative-control checks, or adapt those checks so small surfaces remain valid.

## Changed demos and tests

- `demos/es20-foundations/blending` — **ok.** Cluster scale and half-axis placement match the stated 192×192 minimum; llvmpipe passes 117/117 at 192×192 and 117/117 at the default size, including the painter-order negative check.
- `demos/es20-foundations/clear-viewport-scissor` — **ok.** Exact 64-pixel panels retain their dimensions and the README lists the fitting layouts; passes 78/78 at 512×128 and 128×512.
- `demos/es20-foundations/copy-and-read-pixels` — **ok.** Out-of-window bounds now use independent framebuffer width and height; passes 103/103 at 333×777, and the output confirms the alignment, row-length, and shifted-copy negative controls still reject.
- `demos/es20-foundations/cube-map` — see **BUG** above.
- `demos/es20-foundations/indexed-drawing` — **ok.** The 4×4 aspect-aware tile layout and 160×160 minimum are documented; passes 65/65 at that minimum.
- `demos/es20-foundations/mipmaps` — see **BUG** above.
- `demos/es20-foundations/primitive-topology` — **ok.** Exact cell layout variants and minimum surfaces are documented; passes 73/73 at 512×128.
- `demos/es20-foundations/stencil-test` — **ok.** Independently rounded stretched design edges and the 64×64 minimum are documented; passes 199/199 at 64×64 and 333×777.
- `demos/es20-foundations/textures-2d` — **ok.** The revised LINEAR check uses the documented convex-weight texel range and keeps its mutation check; passes 34/34 at the default size, with 55/64 shifted-sample rejections. Its pre-existing probe-count threshold also fails below the dimensions needed to collect 11 probes; this diff did not add that threshold or change the layout.
- `demos/es20-foundations/uniforms` — **ok.** The 8×5 cell dimensions and 8×5 minimum match the shader division; passes 25/25 at 8×5, including both mutation probes.
- `demos/es20-foundations/vertex-attributes` — **ok.** README additions accurately identify the basic shader, VBO/VAO setup, interleaved layout, and draw/validation paths; passes 126/126 at the default size.
- `tests/glsl/glsl300-builtins` — **ok.** The signed division operands are exactly divisible, avoiding the undefined rounding case cited in the source; passes 42/42.
- `tests/glsl/glsl-language` — **ok.** The removed API claim for printed-only indexed limits and the removed undefined-behavior feature align with the checks; passes 89/89, with documented Mesa XFAIL cases and live negative controls.
- `tests/negative/metagl-contracts` — **ok.** The updated ES 3.0 transpose semantics match the run-time contract checks; passes 19/19, including forked invalid-input deaths and the context-loss cycle.

