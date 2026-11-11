New demo: `demos/es30/multisample-default-framebuffer/`, executable `es30-multisample-default-framebuffer`,
min_es 3.0. Feature: ES30.MULTISAMPLE.DEFAULT (check inventory/features.csv). Set `demo::Info::samples = 4`
so the runtime requests a 4-sample default framebuffer. NOTE (docs/KNOWN_LIMITATIONS.md): on llvmpipe and
zink Mesa yields a 0x0 multisampled pbuffer, and the runtime reports BLOCKED_ENVIRONMENT there — that is
expected; on radeonsi (gpu) it works. Do not work around it.
- Query SAMPLE_BUFFERS = 1 and SAMPLES ≥ 1 on the default framebuffer; glGetMultisamplefv is ES 3.1 —
  use it only when the context is ≥ 3.1.
- Render anti-aliased geometry (a rotating 3D wireframe/solid scene) and verify: edges show intermediate
  resolved values; interiors are exact; ReadPixels from the multisampled default framebuffer resolves
  (ES 3.0 §4.3.2: reading from a multisample default framebuffer is allowed and returns resolved values —
  verify the exact text); glBlitFramebuffer from the default framebuffer into a single-sample FBO with
  matching size works and matches ReadPixels; glBlitFramebuffer with mismatched rectangles from a
  multisample read framebuffer → INVALID_OPERATION.
- Resolve values at edges are implementation-dependent — assert only bounds/monotonicity the spec defines;
  print measured coverage.
