# Brief: make fixed-size pictures fill the window

A contact sheet of every demo at 640x400 (llvmpipe) showed about 16 demos drawing their picture in a fixed-size
block in the lower-left corner. Validation passes, but the DEMO_BRIEF "Window size" rule is that the picture must fill
the surface at any size.

Ways to fix it, in order of preference. Pick per demo and never weaken a check:
1. **Image/texture display cells:** cells that show textures, readbacks or swatches are laid out relative to
   env.width and env.height (partition the surface exactly). Sample with NEAREST and integer texel mapping, and accept
   either neighbouring texel at boundaries. See es32-layered-rendering and es31-image-load-store for the pattern.
2. **Pixel-exact experiments** (viewport, scissor, rasterization rules, ReadPixels geometry, blit rectangles): apply
   an integer scale factor k = max(1, floor(min(width / W0, height / H0))) to every pixel-unit quantity of the
   design: viewports, scissor boxes, blit rectangles, ReadPixels rectangles, pixel-unit geometry. Scale the CPU model
   identically and centre the scaled design. Rasterization of scaled triangles changes edge pixels, so either derive
   probes from the scaled geometry or keep the original margin-based probe rules.
2b. **Offscreen design, scaled presentation** (approved in es30-texture-swizzle-lod): if the design depends on exact
   pixel geometry and the demo does not test the default framebuffer itself, render the design at its original
   size into an offscreen texture, run the existing checks on that texture unchanged, then present it with an
   integer-mapped stretch. Add a check that every window pixel equals the texel it must show.
3. If neither works without weakening a check, keep the fixed design, add a clear README note explaining why, and
   report it. Do not fake it.

Then:
- verify with `tools/sweep_sizes.sh build/debug 256x256 333x777 1000x200 200x1000 640x400 800x800` (llvmpipe; it lists
  only failures) and `ctest --preset debug -R '<exe>'`;
- view a 640x400 and an 800x800 capture of each demo;
- update visual_review and the README layout description.

CPU limit `-j 2`; llvmpipe first; never run anything known to fault the GPU.
