#version 300 es
// Texture lookups in the VERTEX stage, captured with transform feedback (one point per case, gl_VertexID = case).
//
// ES 3.0 §2.12.9.4: a vertex shader has no derivatives, so there is no automatic level selection. A lookup with an
// explicit lod uses lambda_base = lod; one without (texture, textureOffset, textureProj) uses lambda_base = 0, i.e. the
// base level. The bias parameter does not exist in this stage (GLSL ES 3.00 §8.8).
precision highp float;
precision highp int;

uniform highp sampler2D u_t2d;
uniform highp sampler3D u_t3d;
uniform highp sampler2DArray u_tarr;
uniform highp samplerCube u_tcube;
uniform highp usampler2D u_tu2d;
uniform highp isampler3D u_ti3d;
uniform highp sampler2DShadow u_s2d;

const int kCases = 13;
uniform vec4 u_vp[kCases];    // P per case
uniform float u_vlod[kCases]; // explicit lod per case
uniform ivec4 u_vip[kCases];  // texelFetch coordinate (i, j, k, lod), or textureSize lod in .w
uniform vec2 u_vref[kCases];  // depth references for the shadow case

flat out uvec4 v_result;

void main()
{
    int c = gl_VertexID;
    vec4 P = u_vp[c];
    float lod = u_vlod[c];
    ivec4 ip = u_vip[c];
    uvec4 r = uvec4(0u);
    switch (c)
    {
    case 0:  r = floatBitsToUint(texture(u_t2d, P.xy)); break;
    case 1:  r = floatBitsToUint(textureLod(u_t2d, P.xy, lod)); break;
    case 2:  r = floatBitsToUint(textureOffset(u_t2d, P.xy, ivec2(2, -1))); break;
    case 3:  r = floatBitsToUint(textureProj(u_t2d, P.xyw)); break;
    case 4:  r = floatBitsToUint(textureProjLod(u_t3d, P, lod)); break;
    case 5:  r = floatBitsToUint(textureLodOffset(u_tarr, P.xyz, lod, ivec2(-1, 1))); break;
    case 6:  r = floatBitsToUint(texelFetch(u_t3d, ip.xyz, ip.w)); break;
    case 7:  r = floatBitsToUint(texelFetchOffset(u_t2d, ip.xy, ip.w, ivec2(-3, 2))); break;
    case 8:  r = uvec4(ivec4(textureSize(u_tcube, ip.w), textureSize(u_tarr, ip.w).xz)); break;
    case 9:  r = floatBitsToUint(textureLod(u_tcube, P.xyz, lod)); break;
    case 10: r = floatBitsToUint(vec4(textureLod(u_s2d, vec3(P.xy, u_vref[c].x), lod), textureLod(u_s2d, vec3(P.xy, u_vref[c].y), lod), 0.0, 0.0)); break;
    case 11: r = textureLod(u_tu2d, P.xy, lod); break;
    case 12: r = uvec4(texelFetch(u_ti3d, ip.xyz, ip.w)); break;
    }
    v_result = r;
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
}
