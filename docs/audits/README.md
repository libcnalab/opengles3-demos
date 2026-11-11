# Independent audits

Reports from agents that did not write the code they audited. Each report is followed by an integration step:
every finding is checked against the spec text, fixed or rejected with a reason, and the result is recorded below.

## 2026-11-11: Codex phase 6 audit (AUDIT_BRIEF.md), 6 groups, 19 findings

| report | findings | disposition |
|---|---|---|
| [es20](2026-11-11-codex-es20.md) | 3 WEAK (textures-2d LINEAR precision; mipmaps weight/LOD precision, constant-image preservation) | accepted; fixed by demoting to observations (precision subagent) |
| [es30a](2026-11-11-codex-es30a.md) | 1 WEAK (depth-textures "Vulkan" match too broad) | accepted; narrowed to lavapipe-backed renderers |
| [es30b](2026-11-11-codex-es30b.md) | 4 BUG (post-processing, sync-objects, texture-3d-and-arrays, transform-feedback fail at 333x777) | accepted; size fixes (window-size subagent) |
| [es31](2026-11-11-codex-es31.md) | 1 WEAK (framebuffer-no-attachments derivative tolerance) | accepted; observation or justified bound |
| [es32](2026-11-11-codex-es32.md) | 2 BUG (cube-map-array at 333x777; float-targets fixed 256x256 viewport), 1 WEAK (geometry-shaders §11.3.4.4 is conditional) | accepted; size fixes; the conditional link rule becomes an observation and its KNOWN_LIMITATIONS row goes |
| [tests](2026-11-11-codex-tests.md) | 3 BUG (undefined -17/5; ES 2.0 transpose text; UNDEFINED_BEHAVIOR claimed), 3 WEAK (unverified API claims) | accepted; fixed in 75a64c8 |
