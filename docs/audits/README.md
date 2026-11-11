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

## 2026-11-11: Codex educational review (EDUCATION_REVIEW_BRIEF.md), 4 groups

Most demos were rated "good". Actionable items:
- **Accepted:** 1 WRONG (srgb: "8-bit images are stored in sRGB") and 9 CONFUSING/POLISH: buffer-mapping zero-copy
  wording; deferred-shading cost claim and read-first pointer; multisample-default-framebuffer, uniform-buffer-objects
  and vertex-attributes read-first routes; post-processing pass list; shadow-mapping wording and comparison-vs-filter
  note; sync-objects "without synchronization"; glsl-texture-functions overview pointer.
- **Rejected:** texture-3d-and-arrays "layer rounding" — the README matches ES 3.0 §3.8.10.2,
  `l = clamp(⌊r + 0.5⌋, 0, dt − 1)`.

Reports: [es20](2026-11-11-codex-review-es20.md), [es30a](2026-11-11-codex-review-es30a.md),
[es30b](2026-11-11-codex-review-es30b.md), [es31/es32](2026-11-11-codex-review-es31-32.md).

## 2026-11-11 (late): Codex independent review of the post-audit changes (DIFF_REVIEW_BRIEF.md)

Scope: `git diff 5bd59a6..HEAD` for the 41 demo/test directories changed by the fixing agents after the audit. Most
changes were confirmed: no unjustified loosening; thresholds are derived from the geometry; negative controls still
bite. Findings, all accepted:
- **BUG, minimum sizes:**
  - es20-cube-map and es20-mipmaps have no Skip guard, and tiny surfaces FAIL;
  - es30-instancing fails at its documented 240x240 minimum;
  - es32-cube-map-array fails at its admitted 80x89.
- **WEAK:** es30-post-processing's dark-channel floor was lowered from w*h/4 to w*h/8 on an observation, not a
  derivation.

Reports: [es20+tests](2026-11-11-codex-diffreview-es20-tests.md), [es30](2026-11-11-codex-diffreview-es30.md),
[es31/es32](2026-11-11-codex-diffreview-es31-32.md).
