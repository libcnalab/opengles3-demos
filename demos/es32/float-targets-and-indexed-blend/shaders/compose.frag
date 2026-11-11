#version 320 es
// Reads one texture texel per window pixel (texelFetch, no filtering: R32F is not filterable) and shows it.
//   mode 0: HDR scene. Left half: what a clamping RGBA8 target would hold, min(hdr, 1). Right half: Reinhard
//           tone mapping m / (1 + m) of exposure * hdr. Two grey columns mark the split.
//   mode 1: signed map s / (1 + |s|) * 0.5 + 0.5 for values of any sign and size (0 maps to 0.5).
//   mode 2: the texel as it is (an RGBA8 texture).
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_tex;
uniform int u_mode;
uniform ivec2 u_origin;  // window position of the region's lower left corner
uniform ivec2 u_size;    // region size in pixels
uniform float u_exposure;

layout(location = 0) out vec4 o_color;

void main()
{
    ivec2 local = ivec2(gl_FragCoord.xy) - u_origin;
    ivec2 texel = local * textureSize(u_tex, 0) / u_size;
    vec4 c = texelFetch(u_tex, texel, 0);

    if (u_mode == 0)
    {
        if (local.x == u_size.x / 2 - 1 || local.x == u_size.x / 2)
        {
            o_color = vec4(vec3(0.85), 1.0);
            return;
        }
        if (local.x < u_size.x / 2)
            o_color = vec4(clamp(c.rgb, 0.0, 1.0), 1.0);
        else
        {
            vec3 m = c.rgb * u_exposure;
            o_color = vec4(m / (1.0 + m), 1.0);
        }
    }
    else if (u_mode == 1)
    {
        o_color = vec4(c.rgb / (1.0 + abs(c.rgb)) * 0.5 + 0.5, 1.0);
    }
    else
    {
        o_color = vec4(c.rgb, 1.0);
    }
}
