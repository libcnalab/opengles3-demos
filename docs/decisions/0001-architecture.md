# ADR 0001 — Architecture of the suite (2026-11-11)

**Independent executables + tiny runtime.** Each demo is its own executable with a four-method
`demo::Demo` interface. The shared runtime (`common/`) handles only CLI, native context creation,
meta-gl initialization, the frame loop, and result reporting. Rejected: a single launcher app,
since the brief requires independent programs, and a renderer abstraction, which would hide the API.

**Headless = EGL pbuffer, not an FBO.** A pbuffer gives a real default framebuffer (FBO 0) with
depth/stencil/MSAA, so demos are written exactly as for a window, and default-framebuffer behavior
(e.g. `glInvalidateFramebuffer` with `GL_COLOR`, `glReadBuffer(GL_BACK)`) can be tested.

**No libGLESv2 link + `GL_GLES_PROTOTYPES=0`.** Direct `gl*` calls cannot compile or link; all calls
go through meta-gl. EGL's `eglGetProcAddress` (EGL_KHR_get_all_proc_addresses) or
`SDL_GL_GetProcAddress` supplies core entry points.

**Minimum-version runs via `MESA_GLES_VERSION_OVERRIDE` + a native ES 3.1 driver.** Mesa always
creates its highest ES version, so requesting 3.0 proves nothing. The override lowers the reported
API/GLSL version but (as measured on 2026-11-11) not the compiler's accepted `#version`s. Therefore
softpipe, which natively implements ES 3.1, is part of the matrix to provide genuine version-rejection
evidence. Amended the same day after the shader tests exposed the override's limits.

**Shaders are files loaded at runtime from the demo source directory** (`Info::source_dir`), so
readers can edit and rerun without rebuilding. Shaders for negative tests live in `shader-tests/`.

**Manifests (`demo.json`) drive CMake (name, min_es, labels) and coverage**, so a demo is added by
creating one directory. This is what makes parallel work by several agents conflict-free.
