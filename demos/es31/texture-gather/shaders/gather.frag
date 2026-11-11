#version 310 es
// textureGather / textureGatherOffset (GLSL ES 3.10 section 8.9.3).
//
// main.cpp injects #define lines right after the #version line to build one program per variant:
//   GSAMPLER   the sampler declaration, e.g. "highp isampler2D" (never rely on a default precision: GLSL ES 3.10
//              section 4.7.4 gives most sampler types none)
//   GVEC4      the result type: vec4 / ivec4 / uvec4
//   TARGET_ARRAY | TARGET_CUBE   the coordinate form (default: a plain 2D coordinate)
//   SHADOW     the sampler is a shadow sampler: the call takes a reference depth and returns comparison results
//   COMP       the constant component argument (omitted: the call has no comp argument and selects x)
//   OFFSET     the constant texel offset of textureGatherOffset (omitted: textureGather)
//
// Every fragment of the viewport asks for one 2 x 2 footprint. The fragment at window pixel (a, b) uses the texture
// coordinate (a, b) / 8 on an 8 x 8 texture: in texel units that is the point (a, b), the corner shared by the
// texels a-1 and a in s and b-1 and b in t. The footprint of the LINEAR filter around that point is exactly
// i0 = a-1, j0 = b-1, i1 = a, j1 = b, and the point lies 0.5 texel away from every texel centre, so no rounding
// can move it to another footprint.
precision highp float;
precision highp int;

uniform GSAMPLER u_tex;
uniform float u_size;   // texels per edge (8)
uniform float u_ref;    // reference depth of the shadow samplers
uniform int u_layer;    // array layer, or cube map face (0..5 = +X -X +Y -Y +Z -Z)
uniform int u_comp;     // only used by the shaders that must be REJECTED: comp and offset must be constants here

out GVEC4 o_value;

// The inverse of table 8.20 of the ES 3.2 specification: a direction whose major axis is +-1 and whose
// face coordinates are sc = 2s - 1, tc = 2t - 1.
vec3 cube_direction(int face, vec2 st)
{
    float sc = st.x * 2.0 - 1.0;
    float tc = st.y * 2.0 - 1.0;
    if (face == 0) return vec3(1.0, -tc, -sc);    // +X: sc = -rz, tc = -ry
    if (face == 1) return vec3(-1.0, -tc, sc);    // -X: sc =  rz, tc = -ry
    if (face == 2) return vec3(sc, 1.0, tc);      // +Y: sc =  rx, tc =  rz
    if (face == 3) return vec3(sc, -1.0, -tc);    // -Y: sc =  rx, tc = -rz
    if (face == 4) return vec3(sc, -tc, 1.0);     // +Z: sc =  rx, tc = -ry
    return vec3(-sc, -tc, -1.0);                  // -Z: sc = -rx, tc = -ry
}

void main()
{
    vec2 st = vec2(ivec2(gl_FragCoord.xy)) / u_size;
#if defined(TARGET_ARRAY)
    vec3 coord = vec3(st, float(u_layer));
#elif defined(TARGET_CUBE)
    vec3 coord = cube_direction(u_layer, st);
#else
    vec2 coord = st;
#endif

#if defined(SHADOW)
  #if defined(OFFSET)
    o_value = textureGatherOffset(u_tex, coord, u_ref, OFFSET);
  #else
    o_value = textureGather(u_tex, coord, u_ref);
  #endif
#elif defined(COMP) && defined(OFFSET)
    o_value = textureGatherOffset(u_tex, coord, OFFSET, COMP);
#elif defined(OFFSET)
    o_value = textureGatherOffset(u_tex, coord, OFFSET);
#elif defined(COMP)
    o_value = textureGather(u_tex, coord, COMP);
#else
    o_value = textureGather(u_tex, coord);
#endif
}
