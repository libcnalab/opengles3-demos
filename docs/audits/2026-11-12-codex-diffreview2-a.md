# Diff review (`f75838c..HEAD`)

- `demos/es20-foundations/clear-viewport-scissor` — ok; integer-scaled geometry and CPU model retain the exact checks, and panel N remains non-vacuously below the window.
- `demos/es20-foundations/copy-and-read-pixels` — ok; the integer magnification keeps texel probes non-vacuous, and the README documents the computed minimum cell size.
- `demos/es20-foundations/framebuffer-objects` — ok; integer magnification preserves the blit checks and negative controls; the minimum layout keeps each 64×64 target at least 1:1.
- `demos/es20-foundations/primitive-topology` — ok; the per-primitive scaling rules preserve pixel-center rasterization checks, and the README describes minimum supported layouts.
- `demos/es30/framebuffer-blit-msaa` — WEAK: `demos/es30/framebuffer-blit-msaa/main.cpp:232` accepts either neighboring source pixel within 0.05 pixel of an edge. The README calls this implementation-dependent rounding but gives no normative spec quote or bound for that band; this can accept a wrong NEAREST result away from an exact tie.
- `demos/es30/framebuffer-invalidate` — WEAK: `demos/es30/framebuffer-invalidate/main.cpp:163` applies the same unsubstantiated 0.05-pixel neighbor allowance; its README likewise claims implementation-dependent rounding without a normative spec quote or bound.
- `demos/es30/immutable-storage` — ok; layout selection and the 16×16 minimum support at least 100 probes away from texel boundaries, as documented, and negative decode controls remain active.
- `demos/es30/integer-float-textures` — ok; the 16×16 minimum preserves the two-pixel split in the above-2²⁴ panel, and all 78 validation checks pass.
- `demos/es30/multiple-render-targets` — ok; quarter-window panels remain non-vacuous at the stated minimum, and both attachment/model negative controls remain active.
- `demos/es30/pixel-buffer-objects` — ok; the layout minimum fits each source texture at 1:1 or larger, and pack/unpack negative controls remain active.
- `demos/es30/sampler-objects` — ok; the 32×32 cell minimum retains the fixed probe counts and split-cell probes, and sampler override negative controls remain active.

Validation: built the 11 targets with `-j 1`; all 33 selected llvmpipe CTest cases passed (ES max, ES min, and 333×777 size sweep); direct headless validation on llvmpipe passed for all 11 demos, including their negative controls.
