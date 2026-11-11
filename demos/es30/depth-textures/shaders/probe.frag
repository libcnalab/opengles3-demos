#version 300 es
// Read-back probe. Fragment i of an N x 1 RGBA32UI target performs lookup i with coordinate
// u_coord[i].xyz and reference u_ref[i], through the sampler selected by u_mode.
// The result is stored bit for bit with floatBitsToUint, so the CPU sees the exact float the
// shader received. RGBA32UI is color-renderable in core ES 3.0 (Table 3.13); R32F is not.
precision highp float;
precision highp int;
// sampler2D and samplerCube default to lowp in the fragment language and the array and shadow
// types have no default at all (GLSL ES 3.00 §4.5.4): declare highp so depth keeps its precision.
precision highp sampler2D;
precision highp sampler2DArray;
precision highp samplerCube;
precision highp sampler2DShadow;
precision highp sampler2DArrayShadow;
precision highp samplerCubeShadow;

uniform sampler2D u_raw2d;                    // unit 0: compare mode NONE, r = Dt, color (Dt, 0, 0, 1)
uniform sampler2DShadow u_shadow2d;           // unit 1: Dref = P.z
uniform sampler2DArrayShadow u_shadow_array;  // unit 2: layer = P.z, Dref = P.w
uniform samplerCubeShadow u_shadow_cube;      // unit 3: direction = P.xyz, Dref = P.w
uniform sampler2DArray u_raw_array;           // unit 4
uniform samplerCube u_raw_cube;               // unit 5

uniform int u_mode;
uniform vec4 u_coord[64];
uniform float u_ref[64];

out uvec4 o_bits;

void main()
{
    int i = clamp(int(gl_FragCoord.x), 0, 63);
    vec4 c = u_coord[i];
    float ref = u_ref[i];
    vec4 r = vec4(0.0);
    if (u_mode == 0)
        r = texture(u_raw2d, c.xy);
    else if (u_mode == 1)
        r.x = texture(u_shadow2d, vec3(c.xy, ref));
    else if (u_mode == 2)
        r.x = texture(u_shadow_array, vec4(c.xyz, ref));
    else if (u_mode == 3)
        r.x = texture(u_shadow_cube, vec4(c.xyz, ref));
    else if (u_mode == 4)
        r = texture(u_raw_array, c.xyz);
    else
        r = texture(u_raw_cube, c.xyz);
    o_bits = floatBitsToUint(r);
}
