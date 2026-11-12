# Note for the meta-gl maintainer: the G1–G8 changes and a meta-gl release

Prepared 2026-11-12 for the owner, as input to a decision only the owner can make. No agent has changed meta-gl
since the final stabilization directive, and none will (AGENTS.md rule 2). Details and spec evidence:
[META_GL_GAPS.md](META_GL_GAPS.md); the dependency itself: [METAGL_DEPENDENCY.md](METAGL_DEPENDENCY.md).

## 1. Why the demos need G1–G8

opengles3-demos calls OpenGL ES only through meta-gl's typed API, never through raw enums or casts. Eight gaps in
meta-gl `v0.4.1` had no typed spelling for core ES functionality the demos must exercise:

| gap | missing in v0.4.1 | needed by |
|---|---|---|
| G1–G3 | `GetParameter` pnames (pixel store, hints, enable flags), `BufferTarget::TextureBuffer`, `TextureBufferOffsetAlignment` | diagnostics-capabilities, es20 state-query and texture executables, es32-texture-buffer, test-glsl320-builtins |
| G4, G5 | `glGetFramebufferAttachmentParameteriv` for default-framebuffer attachments (`BACK`, `DEPTH`, `STENCIL`) | es30-multisample-default-framebuffer, es30-framebuffer-invalidate, es30-srgb, es30-post-processing, diagnostics-capabilities |
| G6 | `TextureLevelParameter::Samples`, `FixedSampleLocations` | es31-stencil-texturing-and-level-queries, es32-multisample-array-and-border-color |
| G7 | buffer-texture level queries (`TEXTURE_BUFFER_DATA_STORE_BINDING/OFFSET/SIZE`) | es32-texture-buffer |
| G8 | `glUniformMatrix*` with `transpose = GL_TRUE` terminated, although ES 3.0+ allows it | es20-transformations-camera, test-metagl-contracts |

Measured against the pre-fix `develop` (`795bcdc`): 17 of 82 executables do not compile, and
`test-metagl-contracts` fails 6 runs (METAGL_DEPENDENCY.md). The suite contains no local copies or raw-cast
substitutes, so it cannot be built against `v0.4.1`.

## 2. What exists

Seven commits on meta-gl, `795bcdc..333d777` (fast-forward):

| commit | content |
|---|---|
| `146e171` | G1, G2, G3 (enumerators) |
| `4021048` | G4, G5 (default-framebuffer attachment query overload) |
| `0a7766c` | G6 (enumerators) |
| `a451376` | G7 (enumerators) |
| `3bb7411` | G8 (contract relaxed for ES 3.0+) |
| `de86b9e` | O8: test-only change (death tests under an initialized mock context) |
| `333d777` | O12: documentation only |

**Authorship.** The commits were written by an AI agent (Claude Code) on 2026-11-11 under the owner's explicit
authorization, and merged into `develop` at the owner's request on 2026-11-12. Their git author field shows the
owner's name and e-mail, because the agent committed with the repository's configured identity; they carry no AI
attribution trailer. They are therefore **not** human-authored, whatever the metadata says.

## 3. Where they are, and whether they are public

- `origin/develop` = `333d777fadbe53613a9c2eead358ffbfe6a2ef33` on <https://github.com/libcna/meta-gl>, also
  `origin/opengles3-demos/enum-gaps` (checked with `git ls-remote` on 2026-11-12). The exact revision is publicly
  reachable, so the suite's dependency is reproducible.
- `main` and the latest tag `v0.4.1` are `dd3b300` and do not contain them. **No tag contains `333d777`.**
- The local checkout `../../libcna/meta-gl` currently shows `AGENTS.md` and `CLAUDE.md` as deleted in its working
  tree (uncommitted). This session did not make that change and did not touch it. It does not affect the build
  (the configure check reads commits, not the working tree), but the owner may want to look at it.

## 4. What an approved meta-gl release needs

1. A decision on the policy question in section 5.
2. Depending on it, either nothing more, or a human-written equivalent of the seven commits (section 5, option B).
   The demos depend only on the public names (enumerators `GetParameter::*`, `BufferTarget::TextureBuffer`,
   `TextureLevelParameter::Samples`/`FixedSampleLocations`/buffer-texture queries, the
   `DefaultFramebufferQueryAttachment` overload of `glGetFramebufferAttachmentParameteriv`, and `GL_TRUE`
   acceptance in `glUniformMatrix*`). A re-implementation that keeps these names builds the suite unchanged.
3. A release tag (for example `v0.5.0`) on `main` or `develop` containing that state. opengles3-demos then changes
   `OGLES3_METAGL_REQUIRED_COMMIT` (CMakeLists.txt) and METAGL_DEPENDENCY.md to the tag's commit and re-runs its
   matrices; nothing else in the suite changes.

## 5. The decision that remains with the owner

meta-gl's committed `AGENTS.md` (AI Contribution Policy v2) says that AI must not be used to modify anything in
the project, including source, tests, documentation and commit messages. The seven commits contradict that policy,
although the owner authorized them. Options:

- **A. Keep them under a recorded exception.** Record in meta-gl (by a human) that these seven commits were made
  by an AI agent with the owner's authorization before the policy was enforced for this work, and tag a release.
  Fastest; the suite is already validated against exactly this code.
- **B. Re-implement by a human.** Revert or replace `795bcdc..333d777` with human-written changes that keep the
  public names in section 4, then tag. The suite rebuilds unchanged and its matrices are re-run against the new
  commit.
- **C. Leave as is.** The suite keeps pinning the `develop` commit `333d777`. Reproducible, but the policy
  conflict stays open, and there is no tagged meta-gl release to point 1.0.0 at.

Agents do not choose between these. Until the owner decides, opengles3-demos continues to require
`333d777fadbe53613a9c2eead358ffbfe6a2ef33`.
