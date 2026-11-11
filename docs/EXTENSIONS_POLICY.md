# Extensions policy

## Scope

The mandatory scope is **core** OpenGL ES 3.0, 3.1, 3.2 and GLSL ES 3.00, 3.10, 3.20.
Extensions are secondary:

- **Allowed:** the diagnostics executable (`diagnostics-capabilities`) enumerates extensions with
  the ES 3.x mechanism — `glGetIntegerv(GL_NUM_EXTENSIONS)` + `glGetStringi(GL_EXTENSIONS, i)`,
  which is what `metagl::HasExtension()` uses — and reports implementation limits and optional
  capabilities. (The legacy single `glGetString(GL_EXTENSIONS)` string is not used.)
- **Not allowed:** using an extension inside a core demo, using an extension to replace a missing
  core feature, adding vendor-specific entry points, or counting extension entry points in the
  358-function core inventory.
- When a feature was **promoted to core** (e.g. KHR_debug → ES 3.2, OES_texture_buffer → ES 3.2,
  KHR_texture_compression_astc_ldr → ES 3.2), it is covered through the core version only.

## Future optional extension module (not implemented)

If extension coverage is ever added it must:

1. live in `demos/ext/<extension-name>/`, with `"category": "ext"` and `"extension": "GL_..."` in
   `demo.json`; core coverage statistics ignore that category;
2. check `metagl::HasExtension()` and `metagl::IsFunctionAvailable()` and report
   `SKIP_UNSUPPORTED` when absent;
3. load extension entry points only through meta-gl (meta-gl currently wraps no extension-only
   entry points, so any such demo first needs a meta-gl addition);
4. prioritize portable Khronos/EXT extensions with practical value — candidates:
   `GL_KHR_debug` on ES 3.0/3.1 contexts, `GL_EXT_texture_filter_anisotropic`,
   `GL_EXT_disjoint_timer_query`, `GL_EXT_color_buffer_float` on ES 3.0/3.1,
   `GL_OES_texture_float_linear`.
