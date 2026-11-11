# meta-gl dependency

opengles3-demos calls OpenGL ES only through [meta-gl](../../../libcna/meta-gl). This file records the exact
meta-gl revision the suite needs and what happens with the public one. Measured 2026-11-11.

## Required revision

| | revision | content |
|---|---|---|
| **public `develop`** | `795bcdc` (`v0.4.1-2-g795bcdc`, "ai policy v2") | upstream; contains AI Contribution Policy v2 |
| **required** | branch `opengles3-demos/enum-gaps` at `3bb7411` (pushed to `origin`) | `develop` + 5 commits (6 files, +123/−13) |

The five commits implement the fixes for gaps G1-G8 (details, spec evidence and motivation are in
[META_GL_GAPS.md](META_GL_GAPS.md)):

| commit | gaps | adds |
|---|---|---|
| `146e171` | G1, G2, G3 | `GetParameter` pixel-store, hint and enable-flag pnames; `BufferTarget::TextureBuffer`; `TextureBufferOffsetAlignment` |
| `4021048` | G4, G5 | `glGetFramebufferAttachmentParameteriv` overload for default-framebuffer attachments (`DefaultFramebufferQueryAttachment`) |
| `0a7766c` | G6 | `TextureLevelParameter::Samples`, `FixedSampleLocations` |
| `a451376` | G7 | `TextureLevelParameter` buffer-texture data-store queries |
| `3bb7411` | G8 | `glUniformMatrix<…>` helpers accept `transpose = GL_TRUE` (ES 3.0+) instead of terminating |

The changes are additive, except G8. G8 relaxes a contract: the helpers used to call `std::terminate` on
`GL_TRUE`. That termination was wrong for ES 3.0+, and no dependent relied on it (easy-gl, cna-engine and CNA were
searched).

**Policy status.** meta-gl's committed `AGENTS.md` (AI Contribution Policy v2) forbids AI-authored
contributions. The owner explicitly authorized these changes, and on 2026-11-11 also pushing the branch. It lives
at `origin/opengles3-demos/enum-gaps`
(<https://github.com/libcna/meta-gl/tree/opengles3-demos/enum-gaps>). `develop` and `main` are untouched.
Whether and how the branch is merged is the owner's decision.

## Building against the public `develop` (measured)

The suite was configured against an unmodified export of `develop`
(`git archive develop`, `-DOGLES3_METAGL_SOURCE_DIR=<export>`):

- **Build:** 65 of 82 executables compile, 17 do not, all because of the missing G1-G4/G6/G7 enumerators and
  overloads:
  diagnostics-capabilities, es20-copy-and-read-pixels, es20-mipmaps, es20-state-queries-errors, es20-textures-2d,
  es30-flat-centroid, es30-framebuffer-invalidate, es30-multisample-default-framebuffer, es30-pixel-buffer-objects,
  es30-post-processing, es30-srgb, es30-texture-3d-and-arrays, es30-transform-feedback,
  es31-stencil-texturing-and-level-queries, es32-multisample-array-and-border-color, es32-texture-buffer,
  test-glsl320-builtins.
- **Run (the 65 that build):** 370 of 376 runs pass. The 6 failures are all `test-metagl-contracts`: it asserts
  that a row-major `glUniformMatrix<4>(…, GL_TRUE)` upload works, and the unmodified helper terminates on it (G8).

The suite therefore **requires** the branch. Without it, the 17 executables above cannot be built and
`test-metagl-contracts` fails. Nothing in opengles3-demos hides this: there are no local copies of the missing
enumerators and no raw-cast substitutes.

## How to build

```sh
# default: ../../libcna/meta-gl must have opengles3-demos/enum-gaps checked out
git -C ../../libcna/meta-gl fetch origin && git -C ../../libcna/meta-gl checkout opengles3-demos/enum-gaps
cmake --preset debug

# or point at any meta-gl source tree explicitly
cmake --preset debug -DOGLES3_METAGL_SOURCE_DIR=/path/to/meta-gl
```

Every result file records `metagl_version` (meta-gl's own version string, `0.4.1` on both revisions). The version
string does not tell the two revisions apart, so the revision is recorded here.
