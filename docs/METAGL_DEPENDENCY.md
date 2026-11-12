# meta-gl dependency

opengles3-demos calls OpenGL ES only through [meta-gl](https://github.com/libcna/meta-gl) (sibling checkout
`../../libcna/meta-gl`). This file records the exact meta-gl revision the suite needs and what happens with older
ones. Measured 2026-11-11; release-candidate state 2026-11-12.

## Required revision

| | revision | content |
|---|---|---|
| **tested and required** | `develop` at `333d777fadbe53613a9c2eead358ffbfe6a2ef33` (`v0.4.1-9-g333d777`) | the 7 commits below (G1-G8 fixes; O8 tests; O12 docs) |
| later `develop` commits | expected to work (the changes are additive) but **not tested** | — |
| `main` / tag `v0.4.1` (`dd3b300`) | does **not** contain the fixes | configure stops with an error |
| work branch | `opengles3-demos/enum-gaps` at `333d777` | identical to `develop` since the merge on 2026-11-12 |
| before the merge | `develop` at `795bcdc` (`v0.4.1-2-g795bcdc`, "ai policy v2") | lacks the fixes; see the measurement below |

The commits implement the fixes for gaps G1-G8 (details, spec evidence and motivation are in
[META_GL_GAPS.md](META_GL_GAPS.md)):

| commit | gaps | adds |
|---|---|---|
| `146e171` | G1, G2, G3 | `GetParameter` pixel-store, hint and enable-flag pnames; `BufferTarget::TextureBuffer`; `TextureBufferOffsetAlignment` |
| `4021048` | G4, G5 | `glGetFramebufferAttachmentParameteriv` overload for default-framebuffer attachments (`DefaultFramebufferQueryAttachment`) |
| `0a7766c` | G6 | `TextureLevelParameter::Samples`, `FixedSampleLocations` |
| `a451376` | G7 | `TextureLevelParameter` buffer-texture data-store queries |
| `3bb7411` | G8 | `glUniformMatrix<…>` helpers accept `transpose = GL_TRUE` (ES 3.0+) instead of terminating |
| `de86b9e` | O8 | test-only: release-contract death tests run under an initialized mock context (no effect on the suite) |
| `333d777` | O12 | docs-only: the invalid-input contract does not cover bitfields (forwarded; driver reports INVALID_VALUE) |

The changes are additive, except G8. G8 relaxes a contract: the helpers used to call `std::terminate` on
`GL_TRUE`. That termination was wrong for ES 3.0+, and no dependent relied on it (easy-gl, cna-engine and CNA were
searched).

**Policy status.** meta-gl's committed `AGENTS.md` (AI Contribution Policy v2) forbids AI-authored
contributions. The seven commits were written by an AI agent under an explicit grant of the owner (2026-11-11),
pushed as `origin/opengles3-demos/enum-gaps`, and on the owner's request merged into `develop` (fast-forward
`795bcdc..333d777`, 2026-11-12). `main` is untouched. Since the final stabilization directive (2026-11-12) agents
make no further meta-gl changes.

That leaves a decision for the meta-gl maintainer, outside this repository: either keep the commits under a
recorded exception to the policy, or have a human review and re-author them. Either way the dependency stays
reproducible: commit `333d777` exists on the public `develop`, and a human-authored equivalent that keeps the same
public names (listed in the table above) would build the suite unchanged. A meta-gl release tag (for example
`v0.5.0`) containing them would let this file pin a tag instead of a branch commit.

**Configure-time check.** `CMakeLists.txt` (`OGLES3_METAGL_REQUIRED_COMMIT`) runs
`git merge-base --is-ancestor 333d777… HEAD` in the meta-gl checkout and stops with an error when the commit is
missing. A source export without `.git` cannot be checked; configure then prints a note and continues.

## Building against the pre-merge `develop` `795bcdc` (measured)

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

The suite therefore **requires** `develop` at `333d777` (or a later commit that contains it). Without it, the 17 executables above cannot be built and
`test-metagl-contracts` fails. Nothing in opengles3-demos hides this: there are no local copies of the missing
enumerators and no raw-cast substitutes.

## How to build

```sh
# default: ../../libcna/meta-gl at the tested commit
git clone https://github.com/libcna/meta-gl ../../libcna/meta-gl
git -C ../../libcna/meta-gl checkout 333d777fadbe53613a9c2eead358ffbfe6a2ef33
cmake --preset debug

# or point at any meta-gl source tree explicitly
cmake --preset debug -DOGLES3_METAGL_SOURCE_DIR=/path/to/meta-gl
```

Every result file records `metagl_version` (meta-gl's own version string, `0.4.1` on all these revisions). The
version string does not tell the revisions apart, so the commit is recorded here, in the configure output, in
`docs/generated/COVERAGE_SUMMARY.md` and in the evidence manifests (`docs/evidence/`).
