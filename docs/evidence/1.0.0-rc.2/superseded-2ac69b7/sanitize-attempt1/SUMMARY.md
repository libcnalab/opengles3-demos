# Evidence: sanitize

- opengles3-demos `2ac69b744a4d945834d3a045c9fedd9cf7834cb9` (v1.0.0-rc.1-6-g2ac69b7)
- meta-gl `333d777fadbe53613a9c2eead358ffbfe6a2ef33` (v0.4.1-9-g333d777)
- build `Debug`, c++ (Debian 14.2.0-19) 14.2.0, cmake version 3.31.6
- run 2026-11-12T20:13:31+01:00 .. 2026-11-12T20:27:30+01:00; `ctest --test-dir /tmp/claude-1000/-rv-data-development-github-com-libcnalab-opengles3-demos/7a63c4aa-c5ab-40b2-9623-c84bee68a371/scratchpad/clean2/libcnalab/opengles3-demos/build/sanitize -j 3 --output-on-failure --output-junit /tmp/claude-1000/-rv-data-development-github-com-libcnalab-opengles3-demos/7a63c4aa-c5ab-40b2-9623-c84bee68a371/scratchpad/evidence-rc2/sanitize/junit.xml`; ctest exit code 8
- registered tests: 701; CTest status: fail 1, notrun 26, run 674
- outcomes: CRASHED 1, N/A (no result JSON by design) 1, PASS 673, SKIP_UNSUPPORTED 26
- kernel log read; amdgpu fault/timeout/reset lines during the run: 0

## Drivers seen

| GL_RENDERER | GL_VERSION | GLSL | runs |
|---|---|---|---|
| AMD Radeon 780M (radeonsi, phoenix, LLVM 19.1.7, DRM 3.61, 6.12.111+deb13-amd64) | OpenGL ES 3.0 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.00 | 54 |
| AMD Radeon 780M (radeonsi, phoenix, LLVM 19.1.7, DRM 3.61, 6.12.111+deb13-amd64) | OpenGL ES 3.1 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.10 | 16 |
| AMD Radeon 780M (radeonsi, phoenix, LLVM 19.1.7, DRM 3.61, 6.12.111+deb13-amd64) | OpenGL ES 3.2 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.20 | 83 |
| ANGLE (Mesa, Vulkan 1.4.305 (llvmpipe (LLVM 19.1.7 256 bits) (0x00000000)), llvmpipe-0.0.1) | OpenGL ES 3.0.0 (ANGLE 2.1.17841 git hash: 5fa2e61da947) | OpenGL ES GLSL ES 3.00 (ANGLE 2.1.17841 git hash: 5fa2e61da947) | 54 |
| ANGLE (Mesa, Vulkan 1.4.305 (llvmpipe (LLVM 19.1.7 256 bits) (0x00000000)), llvmpipe-0.0.1) | OpenGL ES 3.1.0 (ANGLE 2.1.17841 git hash: 5fa2e61da947) | OpenGL ES GLSL ES 3.10 (ANGLE 2.1.17841 git hash: 5fa2e61da947) | 83 |
| llvmpipe (LLVM 19.1.7, 256 bits) | OpenGL ES 3.0 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.00 | 54 |
| llvmpipe (LLVM 19.1.7, 256 bits) | OpenGL ES 3.1 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.10 | 16 |
| llvmpipe (LLVM 19.1.7, 256 bits) | OpenGL ES 3.2 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.20 | 165 |
| softpipe | OpenGL ES 3.0 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.00 | 3 |
| softpipe | OpenGL ES 3.1 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.10 | 18 |
| zink Vulkan 1.4(llvmpipe (LLVM 19.1.7, 256 bits) (MESA_LLVMPIPE)) | OpenGL ES 3.0 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.00 | 54 |
| zink Vulkan 1.4(llvmpipe (LLVM 19.1.7, 256 bits) (MESA_LLVMPIPE)) | OpenGL ES 3.1 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.10 | 16 |
| zink Vulkan 1.4(llvmpipe (LLVM 19.1.7, 256 bits) (MESA_LLVMPIPE)) | OpenGL ES 3.2 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.20 | 83 |

## Runs that did not PASS

| test | ctest | outcome | reason |
|---|---|---|---|
| es30-deferred-shading:llvmpipe:size-333x777 | fail | CRASHED | the process ended before the run completed (crash, abort or timeout) |
| es32-advanced-blend-and-sample-shading:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-advanced-blend-and-sample-shading:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-astc-and-copy-image:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-astc-and-copy-image:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-base-vertex-and-robustness:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-base-vertex-and-robustness:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-cube-map-array:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-cube-map-array:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-debug-output:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-debug-output:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-float-targets-and-indexed-blend:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-float-targets-and-indexed-blend:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-geometry-shaders:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-geometry-shaders:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-layered-rendering:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-layered-rendering:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-multisample-array-and-border-color:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-multisample-array-and-border-color:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-tessellation-domains:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-tessellation-domains:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-tessellation-terrain:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-tessellation-terrain:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-texture-buffer:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| es32-texture-buffer:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| test-glsl320-builtins:angle:native | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| test-glsl320-builtins:softpipe:es-max | notrun | SKIP_UNSUPPORTED | requires OpenGL ES 3.2, context is 3.1 |
| unit-deviation-matcher | run | N/A (no result JSON by design) |  |
