# Evidence: sanitize

- opengles3-demos `2544bad7e0605cb7728594f89279ee5cee6a77e3` (2544bad)
- meta-gl `333d777fadbe53613a9c2eead358ffbfe6a2ef33` (v0.4.1-9-g333d777)
- build `Debug`, c++ (Debian 14.2.0-19) 14.2.0, cmake version 3.31.6
- run 2026-11-12T07:02:26+01:00 .. 2026-11-12T07:09:29+01:00; `ctest --test-dir /tmp/claude-1000/-rv-data-development-github-com-libcnalab-opengles3-demos/316edffb-ec61-4c46-b5e6-4271e88847bb/scratchpad/clean/libcnalab/opengles3-demos/build/sanitize -j 2 --output-on-failure --output-junit /tmp/claude-1000/-rv-data-development-github-com-libcnalab-opengles3-demos/316edffb-ec61-4c46-b5e6-4271e88847bb/scratchpad/clean/final/sanitize/junit.xml`; ctest exit code 0
- registered tests: 701; CTest status: notrun 31, run 670
- outcomes: BLOCKED_ENVIRONMENT 5, N/A (no result JSON by design) 1, PASS 669, SKIP_UNSUPPORTED 26
- amdgpu fault/timeout/reset lines in the kernel log during the run: 0

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
| llvmpipe (LLVM 19.1.7, 256 bits) | OpenGL ES 3.2 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.20 | 166 |
| softpipe | OpenGL ES 3.0 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.00 | 3 |
| softpipe | OpenGL ES 3.1 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.10 | 18 |
| zink Vulkan 1.4(llvmpipe (LLVM 19.1.7, 256 bits) (MESA_LLVMPIPE)) | OpenGL ES 3.0 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.00 | 54 |
| zink Vulkan 1.4(llvmpipe (LLVM 19.1.7, 256 bits) (MESA_LLVMPIPE)) | OpenGL ES 3.1 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.10 | 16 |
| zink Vulkan 1.4(llvmpipe (LLVM 19.1.7, 256 bits) (MESA_LLVMPIPE)) | OpenGL ES 3.2 Mesa 25.0.7-2+deb13u1 | OpenGL ES GLSL ES 3.20 | 83 |

## Runs that did not PASS

| test | ctest | outcome | reason |
|---|---|---|---|
| es30-multisample-default-framebuffer:llvmpipe:es-max | notrun | BLOCKED_ENVIRONMENT | default framebuffer is 0x0, requested 256x256 pbuffer (samples=4) |
| es30-multisample-default-framebuffer:llvmpipe:es-min | notrun | BLOCKED_ENVIRONMENT | default framebuffer is 0x0, requested 256x256 pbuffer (samples=4) |
| es30-multisample-default-framebuffer:llvmpipe:size-333x777 | notrun | BLOCKED_ENVIRONMENT | default framebuffer is 0x0, requested 333x777 pbuffer (samples=4) |
| es30-multisample-default-framebuffer:zink:es-max | notrun | BLOCKED_ENVIRONMENT | default framebuffer is 0x0, requested 256x256 pbuffer (samples=4) |
| es30-multisample-default-framebuffer:zink:es-min | notrun | BLOCKED_ENVIRONMENT | default framebuffer is 0x0, requested 256x256 pbuffer (samples=4) |
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
