#version 300 es
// One GLSL ES 3.00 §8.8 / §8.9 lookup per case, run over one 4x4 block of an RGBA32UI target.
//
// main.cpp inserts "#define OFF_MIN (n)" and "#define OFF_MAX (n)" after the #version line: the values of
// MIN_PROGRAM_TEXEL_OFFSET / MAX_PROGRAM_TEXEL_OFFSET queried from the context (an offset must be a constant expression).
// It also defines GROUP (0..5) and compiles one program per group, each holding one sampler family: a single switch
// over all ~80 cases crashed llvmpipe and zink on the first draw (deep recursion inside the driver).
//
// Every case number below is mirrored by the CPU model in main.cpp. Floating-point results are written as their raw bit
// patterns (floatBitsToUint), integer results unchanged, so the CPU sees exactly what the lookup returned.
//
// The coordinate P is an affine function of the window position:  P = u_p + u_ddx * f.x + u_ddy * f.y, where f is the
// fragment centre relative to the block origin (0.5 .. 3.5). All inputs are short dyadic fractions, so P is exact in
// float arithmetic and its implicit screen-space derivatives are exactly u_ddx and u_ddy.
//
// The switch is on a uniform, so every lookup that needs implicit derivatives runs in uniform control flow (§8.8).
precision highp float;
precision highp int;

uniform highp sampler2D u_t2d;            // unit 0: RGBA8 32x16, 6 levels
uniform highp sampler3D u_t3d;            // unit 1: RGBA8 16x8x8, 5 levels
uniform highp sampler2DArray u_tarr;      // unit 2: RGBA8 16x16, 4 layers, 5 levels
uniform highp samplerCube u_tcube;        // unit 3: RGBA8 16x16 faces, 5 levels
uniform highp usampler2D u_tu2d;          // unit 4: RGBA32UI 16x16, 5 levels
uniform highp isampler3D u_ti3d;          // unit 5: RGBA32I 8x8x4, 4 levels
uniform highp usamplerCube u_tucube;      // unit 6: RGBA8UI 8x8 faces, 4 levels
uniform highp sampler2DShadow u_s2d;      // unit 7: DEPTH_COMPONENT32F 16x16, 5 levels, LEQUAL
uniform highp samplerCubeShadow u_scube;  // unit 8: DEPTH_COMPONENT32F 8x8 faces, 4 levels, LEQUAL
uniform highp sampler2DArrayShadow u_sarr;// unit 9: DEPTH_COMPONENT32F 8x8, 3 layers, 4 levels, LEQUAL

uniform int u_case;
uniform vec2 u_origin;   // window coordinates of the lower-left corner of the block
uniform vec4 u_p;        // P at the block origin
uniform vec4 u_ddx;      // dP per pixel in window x
uniform vec4 u_ddy;      // dP per pixel in window y
uniform vec4 u_gx;       // explicit dPdx (textureGrad)
uniform vec4 u_gy;       // explicit dPdy
uniform float u_lod;     // explicit lod, or the bias of an implicit-LOD lookup
uniform ivec4 u_ip;      // texelFetch: (i, j, k or layer, lod) at the block origin; textureSize: (sampler, lod, -, -)
uniform vec2 u_ref[16];  // depth-comparison references per pixel of the block: (just below, just above) the expected texel

in vec4 v_lin;           // affine in window coordinates (set up by probe.vert), for the derivative cases

out uvec4 o_result;

uvec4 bits(vec4 v) { return floatBitsToUint(v); }
uvec4 pair(float lo, float hi) { return floatBitsToUint(vec4(lo, hi, 0.0, 0.0)); }
uvec4 sz(ivec2 s) { return uvec4(ivec4(s, 1, 0)); }
uvec4 sz(ivec3 s) { return uvec4(ivec4(s, 0)); }

void main()
{
    vec2 f = gl_FragCoord.xy - u_origin;
    ivec2 pix = ivec2(f);
    vec4 P = u_p + u_ddx * f.x + u_ddy * f.y;
    vec2 ref = u_ref[pix.y * 4 + pix.x];
    vec4 Q = P;  // a projective coordinate (s q, t q, ...) with the comparison reference in place for the shadow forms
    ivec2 fetch = u_ip.xy + pix;

    uvec4 r = uvec4(0u);
    switch (u_case)
    {
#if GROUP == 0
    // ---- sampler2D (RGBA8) ------------------------------------------------------------------------------------
    case 1:  r = bits(texture(u_t2d, P.xy)); break;
    case 2:  r = bits(texture(u_t2d, P.xy, u_lod)); break;
    case 3:  r = bits(textureProj(u_t2d, P.xyw)); break;
    case 4:  r = bits(textureProj(u_t2d, P, u_lod)); break;  // vec4 form: P.z is ignored
    case 5:  r = bits(textureLod(u_t2d, P.xy, u_lod)); break;
    case 6:  r = bits(textureOffset(u_t2d, P.xy, ivec2(OFF_MAX, OFF_MIN))); break;
    case 7:  r = bits(textureOffset(u_t2d, P.xy, ivec2(-3, 2), u_lod)); break;
    case 8:  r = bits(textureLodOffset(u_t2d, P.xy, u_lod, ivec2(3, -2))); break;
    case 9:  r = bits(textureGrad(u_t2d, P.xy, u_gx.xy, u_gy.xy)); break;
    case 10: r = bits(textureGradOffset(u_t2d, P.xy, u_gx.xy, u_gy.xy, ivec2(-1, 2))); break;
    case 11: r = bits(textureProjLod(u_t2d, P.xyw, u_lod)); break;
    case 12: r = bits(textureProjLod(u_t2d, P, u_lod)); break;
    case 13: r = bits(textureProjOffset(u_t2d, P, ivec2(2, 1))); break;
    case 14: r = bits(textureProjLodOffset(u_t2d, P.xyw, u_lod, ivec2(-2, -1))); break;
    case 15: r = bits(textureProjGrad(u_t2d, P.xyw, u_gx.xy, u_gy.xy)); break;
    case 16: r = bits(textureProjGradOffset(u_t2d, P, u_gx.xy, u_gy.xy, ivec2(1, -1))); break;
    case 17: r = bits(texelFetch(u_t2d, fetch, u_ip.w)); break;
    case 18: r = bits(texelFetchOffset(u_t2d, fetch, u_ip.w, ivec2(-5, 3))); break;

#elif GROUP == 1
    // ---- sampler3D (RGBA8) ------------------------------------------------------------------------------------
    case 20: r = bits(texture(u_t3d, P.xyz, u_lod)); break;
    case 21: r = bits(textureLod(u_t3d, P.xyz, u_lod)); break;
    case 22: r = bits(textureOffset(u_t3d, P.xyz, ivec3(-2, 1, -1))); break;
    case 23: r = bits(textureLodOffset(u_t3d, P.xyz, u_lod, ivec3(1, -1, 2))); break;
    case 24: r = bits(textureGrad(u_t3d, P.xyz, u_gx.xyz, u_gy.xyz)); break;
    case 25: r = bits(textureGradOffset(u_t3d, P.xyz, u_gx.xyz, u_gy.xyz, ivec3(1, 0, -1))); break;
    case 26: r = bits(textureProj(u_t3d, P)); break;
    case 27: r = bits(textureProjLod(u_t3d, P, u_lod)); break;
    case 28: r = bits(textureProjOffset(u_t3d, P, ivec3(0, 1, 1))); break;
    case 29: r = bits(textureProjLodOffset(u_t3d, P, u_lod, ivec3(-1, 0, 1))); break;
    case 30: r = bits(textureProjGrad(u_t3d, P, u_gx.xyz, u_gy.xyz)); break;
    case 31: r = bits(textureProjGradOffset(u_t3d, P, u_gx.xyz, u_gy.xyz, ivec3(2, -1, 0))); break;
    case 32: r = bits(texelFetch(u_t3d, ivec3(fetch, u_ip.z), u_ip.w)); break;
    case 33: r = bits(texelFetchOffset(u_t3d, ivec3(fetch, u_ip.z), u_ip.w, ivec3(-1, 2, 1))); break;

#elif GROUP == 2
    // ---- sampler2DArray (RGBA8); the layer is P.z, never offset, never part of the level of detail ---------------
    case 40: r = bits(texture(u_tarr, P.xyz, u_lod)); break;
    case 41: r = bits(textureLod(u_tarr, P.xyz, u_lod)); break;
    case 42: r = bits(textureOffset(u_tarr, P.xyz, ivec2(2, -1))); break;
    case 43: r = bits(textureLodOffset(u_tarr, P.xyz, u_lod, ivec2(-1, 1))); break;
    case 44: r = bits(textureGrad(u_tarr, P.xyz, u_gx.xy, u_gy.xy)); break;
    case 45: r = bits(textureGradOffset(u_tarr, P.xyz, u_gx.xy, u_gy.xy, ivec2(1, 1))); break;
    case 46: r = bits(texelFetch(u_tarr, ivec3(fetch, u_ip.z), u_ip.w)); break;
    case 47: r = bits(texelFetchOffset(u_tarr, ivec3(fetch, u_ip.z), u_ip.w, ivec2(2, -2))); break;

    // ---- samplerCube (RGBA8) ----------------------------------------------------------------------------------
    case 50: r = bits(texture(u_tcube, P.xyz, u_lod)); break;
    case 51: r = bits(textureLod(u_tcube, P.xyz, u_lod)); break;
    case 52: r = bits(textureGrad(u_tcube, P.xyz, u_gx.xyz, u_gy.xyz)); break;
    case 53: r = bits(texture(u_tcube, P.xyz)); break;

#elif GROUP == 3
    // ---- integer samplers -------------------------------------------------------------------------------------
    case 60: r = texelFetch(u_tu2d, fetch, u_ip.w); break;
    case 61: r = textureLod(u_tu2d, P.xy, u_lod); break;
    case 62: r = textureOffset(u_tu2d, P.xy, ivec2(2, -3), u_lod); break;
    case 63: r = textureGrad(u_tu2d, P.xy, u_gx.xy, u_gy.xy); break;
    case 64: r = textureProj(u_tu2d, P.xyw); break;
    case 65: r = uvec4(texelFetchOffset(u_ti3d, ivec3(fetch, u_ip.z), u_ip.w, ivec3(-1, 2, 1))); break;
    case 66: r = uvec4(textureLodOffset(u_ti3d, P.xyz, u_lod, ivec3(1, -1, 1))); break;
    case 67: r = uvec4(textureProjLod(u_ti3d, P, u_lod)); break;
    case 68: r = uvec4(texture(u_ti3d, P.xyz, u_lod)); break;
    case 69: r = textureLod(u_tucube, P.xyz, u_lod); break;
    case 70: r = texture(u_tucube, P.xyz, u_lod); break;

#elif GROUP == 4
    // ---- shadow samplers: two comparisons per pixel, reference just below and just above the expected depth ------
    case 80: r = pair(texture(u_s2d, vec3(P.xy, ref.x), u_lod), texture(u_s2d, vec3(P.xy, ref.y), u_lod)); break;
    case 81: r = pair(textureLod(u_s2d, vec3(P.xy, ref.x), u_lod), textureLod(u_s2d, vec3(P.xy, ref.y), u_lod)); break;
    case 82: r = pair(textureOffset(u_s2d, vec3(P.xy, ref.x), ivec2(-2, 3)), textureOffset(u_s2d, vec3(P.xy, ref.y), ivec2(-2, 3))); break;
    case 83: r = pair(textureLodOffset(u_s2d, vec3(P.xy, ref.x), u_lod, ivec2(1, -2)),
                      textureLodOffset(u_s2d, vec3(P.xy, ref.y), u_lod, ivec2(1, -2))); break;
    case 84: r = pair(textureGrad(u_s2d, vec3(P.xy, ref.x), u_gx.xy, u_gy.xy), textureGrad(u_s2d, vec3(P.xy, ref.y), u_gx.xy, u_gy.xy)); break;
    case 85: r = pair(textureGradOffset(u_s2d, vec3(P.xy, ref.x), u_gx.xy, u_gy.xy, ivec2(-1, -1)),
                      textureGradOffset(u_s2d, vec3(P.xy, ref.y), u_gx.xy, u_gy.xy, ivec2(-1, -1))); break;
    // Projective shadow forms: the reference is the third component and is divided by q as well (P' = P / P.w).
    case 86: r = pair(textureProj(u_s2d, vec4(Q.xy, ref.x * Q.w, Q.w)), textureProj(u_s2d, vec4(Q.xy, ref.y * Q.w, Q.w))); break;
    case 87: r = pair(textureProjLod(u_s2d, vec4(Q.xy, ref.x * Q.w, Q.w), u_lod), textureProjLod(u_s2d, vec4(Q.xy, ref.y * Q.w, Q.w), u_lod)); break;
    case 88: r = pair(textureProjOffset(u_s2d, vec4(Q.xy, ref.x * Q.w, Q.w), ivec2(1, 2)),
                      textureProjOffset(u_s2d, vec4(Q.xy, ref.y * Q.w, Q.w), ivec2(1, 2))); break;
    case 89: r = pair(textureProjGrad(u_s2d, vec4(Q.xy, ref.x * Q.w, Q.w), u_gx.xy, u_gy.xy),
                      textureProjGrad(u_s2d, vec4(Q.xy, ref.y * Q.w, Q.w), u_gx.xy, u_gy.xy)); break;
    case 90: r = pair(textureProjLodOffset(u_s2d, vec4(Q.xy, ref.x * Q.w, Q.w), u_lod, ivec2(-1, 1)),
                      textureProjLodOffset(u_s2d, vec4(Q.xy, ref.y * Q.w, Q.w), u_lod, ivec2(-1, 1))); break;
    case 91: r = pair(textureProjGradOffset(u_s2d, vec4(Q.xy, ref.x * Q.w, Q.w), u_gx.xy, u_gy.xy, ivec2(2, -1)),
                      textureProjGradOffset(u_s2d, vec4(Q.xy, ref.y * Q.w, Q.w), u_gx.xy, u_gy.xy, ivec2(2, -1))); break;
    case 92: r = pair(texture(u_scube, vec4(P.xyz, ref.x), u_lod), texture(u_scube, vec4(P.xyz, ref.y), u_lod)); break;
    case 93: r = pair(textureGrad(u_scube, vec4(P.xyz, ref.x), u_gx.xyz, u_gy.xyz), textureGrad(u_scube, vec4(P.xyz, ref.y), u_gx.xyz, u_gy.xyz)); break;
    case 94: r = pair(texture(u_sarr, vec4(P.xyz, ref.x)), texture(u_sarr, vec4(P.xyz, ref.y))); break;
    case 95: r = pair(textureGrad(u_sarr, vec4(P.xyz, ref.x), u_gx.xy, u_gy.xy), textureGrad(u_sarr, vec4(P.xyz, ref.y), u_gx.xy, u_gy.xy)); break;
    case 96: r = pair(textureGradOffset(u_sarr, vec4(P.xyz, ref.x), u_gx.xy, u_gy.xy, ivec2(1, -1)),
                      textureGradOffset(u_sarr, vec4(P.xyz, ref.y), u_gx.xy, u_gy.xy, ivec2(1, -1))); break;

#elif GROUP == 5
    // ---- textureSize for every sampler type; u_ip.x selects the sampler, u_ip.y is the lod ------------------------
    case 100:
        switch (u_ip.x)
        {
        case 0: r = sz(textureSize(u_t2d, u_ip.y)); break;
        case 1: r = sz(textureSize(u_t3d, u_ip.y)); break;
        case 2: r = sz(textureSize(u_tarr, u_ip.y)); break;
        case 3: r = sz(textureSize(u_tcube, u_ip.y)); break;
        case 4: r = sz(textureSize(u_tu2d, u_ip.y)); break;
        case 5: r = sz(textureSize(u_ti3d, u_ip.y)); break;
        case 6: r = sz(textureSize(u_tucube, u_ip.y)); break;
        case 7: r = sz(textureSize(u_s2d, u_ip.y)); break;
        case 8: r = sz(textureSize(u_scube, u_ip.y)); break;
        case 9: r = sz(textureSize(u_sarr, u_ip.y)); break;
        }
        break;

    // ---- §8.9 derivatives of functions that are affine in the window coordinates ---------------------------------
    case 110: r = bits(dFdx(vec4(3.0 * gl_FragCoord.x - 5.0 * gl_FragCoord.y + 7.0, -2.0 * gl_FragCoord.x + 0.5 * gl_FragCoord.y,
                                 0.75 * gl_FragCoord.x + 0.75 * gl_FragCoord.y, 0.25 * gl_FragCoord.y))); break;
    case 111: r = bits(dFdy(vec4(3.0 * gl_FragCoord.x - 5.0 * gl_FragCoord.y + 7.0, -2.0 * gl_FragCoord.x + 0.5 * gl_FragCoord.y,
                                 0.75 * gl_FragCoord.x + 0.75 * gl_FragCoord.y, 0.25 * gl_FragCoord.y))); break;
    case 112: r = bits(fwidth(vec4(3.0 * gl_FragCoord.x - 5.0 * gl_FragCoord.y + 7.0, -2.0 * gl_FragCoord.x + 0.5 * gl_FragCoord.y,
                                   0.75 * gl_FragCoord.x + 0.75 * gl_FragCoord.y, 0.25 * gl_FragCoord.y))); break;
    case 113: r = bits(vec4(dFdx(v_lin.xy), dFdy(v_lin.xy))); break;                   // an interpolated input
    case 114: r = bits(vec4(dFdx(P.x), dFdy(P.y), fwidth(P.x + P.y), 0.0)); break;   // the uniform-built P itself
    // Not linear: x^2 and y^2. The value depends on the differencing method and is only printed, never asserted.
    case 115: r = bits(vec4(dFdx(gl_FragCoord.x * gl_FragCoord.x), dFdy(gl_FragCoord.y * gl_FragCoord.y), gl_FragCoord.xy)); break;
#endif
    }
    o_result = r;
}
