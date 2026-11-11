#version 320 es
// Shows a 4 x 4 texture with texture coordinates from -0.5 to 1.5, so the texture covers the middle
// half of the quad and the rest is outside [0, 1]: with CLAMP_TO_BORDER that frame is the border color.
// main.cpp inserts the defines after the #version line:
//   SAMPLER    sampler2D | isampler2D | sampler2DShadow
//   KIND_FLOAT | KIND_INT | KIND_SHADOW
precision highp float;
precision highp int;

uniform highp SAMPLER u_tex;
uniform highp float u_ref;      // depth reference of the shadow lookup

in highp vec2 v_uv;

layout(location = 0) out vec4 o_color;

void main()
{
#if defined(KIND_SHADOW)
    o_color = vec4(vec3(texture(u_tex, vec3(v_uv, u_ref))), 1.0);    // 1 = pass, 0 = fail
#elif defined(KIND_INT)
    o_color = vec4(vec3(texture(u_tex, v_uv).rgb + 128) / 255.0, 1.0);  // RGBA8I -128..127 -> 0..255
#else
    o_color = vec4(texture(u_tex, v_uv).rgb, 1.0);
#endif
}
