# Independent review: 1.0.0-rc.2

Reviewed `v1.0.0-rc.1..8b5a50f` against `docs/agent-briefs/RC2_REVIEW_BRIEF.md`. No compilation or executable runs were performed.

## Findings

### HIGH — Headless zink runs are not pinned to a software Vulkan device

**Location:** [`cmake/OglesDemo.cmake:112`](cmake/OglesDemo.cmake:112) and [`cmake/OglesDemo.cmake:114`](cmake/OglesDemo.cmake:114) (driver configuration); the stated default is [`CMakeLists.txt:25`](CMakeLists.txt:25).

**Evidence:** For zink, the shared driver branch sets `LIBGL_ALWAYS_SOFTWARE=1` and `GALLIUM_DRIVER=zink`, but adds `VK_ICD_FILENAMES` and `VK_DRIVER_FILES` only inside `if(launch AND driver STREQUAL "zink")`. `launch` is empty for the ordinary headless runs. The nearby project comment describes the default zink driver as “on lavapipe,” but those runs do not select an ICD. In contrast, ANGLE runs always receive the configured lavapipe ICD, and windowed zink runs explicitly receive it.

**Impact:** On a host where the Vulkan loader exposes a hardware ICD, the default headless zink test may select that device. `LIBGL_ALWAYS_SOFTWARE=1` does not pin zink's Vulkan ICD. This leaves a route for zink test work to reach a hardware GPU despite the project's documented software-only zink setup; the repository records a prior GPU fault for an unsafe Vulkan-backed test. The rc.2 evidence confirms that this particular host selected llvmpipe for zink, but does not guarantee selection on other hosts.

**Recommended resolution:** Apply the configured lavapipe ICD variables to every zink run, including headless runs, or otherwise explicitly select and verify a software Vulkan physical device before registering/running zink tests. Keep hardware Vulkan opt-in and document the opt-in boundary.

## Review checks

- The Clang edits inspected are explicit float-to-double conversions at existing promotion sites. In the cases where the conversion wraps arithmetic, the arithmetic remains inside the cast (for example, the edge differences remain float operations); no changed tolerance or overload-selection defect was found in the reviewed diff.
- The multisample demo still requests four samples and checks `SAMPLE_BUFFERS == 1`, `SAMPLES >= 1`, per-sample behavior, and the multisample-specific error cases on the default framebuffer. A single-sample window would fail those assertions. The CMake fallback without SDL3 or `xvfb-run` remains headless; the recorded headless result is `BLOCKED_ENVIRONMENT`. The coverage generator was not changed.
- The runtime resize swap occurs before the validated last-frame render, so that validation uses the resized surface. The lighting sweep omits only near-terminator sphere samples; its numeric tolerance is unchanged, and surfaces below the stated minimum are reported as `SKIP_UNSUPPORTED`.
- Evidence reconciliation: all five rc.2 `results.csv` files have 701 or 395 rows, matching their manifests and summary counts; each manifest records source commit `2ac69b744a4d945834d3a045c9fedd9cf7834cb9` and meta-gl `333d777fadbe53613a9c2eead358ffbfe6a2ef33`. The rc.2 archive SHA-256 is `3e7a4522079c0d15472a77fc2c49a2f0ee7f17cdbdcbf65978a1f610c6309d6b`, matching the documented checksum file and release docs. The rc.1 archive remains present with SHA-256 `2d96c3325bbc6c62be8545f135dbe5f01f0e54050f09d7f10045b38b2409cba7`, matching the recorded value.
- The opt-in GPU-fault case is documented as disabled by default and gated by `OGLES3_RUN_GPU_FAULT_CASES=1`; no default registration or unconditional invocation was found.

## Verdict

There is one unresolved HIGH release concern: headless zink runs do not pin their Vulkan device to lavapipe, so the software-only GPU safety claim is not guaranteed on other hosts. Pin the ICD for all zink runs before treating that safety claim as resolved. No other actionable defect was identified in this focused review.
