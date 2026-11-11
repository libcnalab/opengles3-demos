#version 320 es
// Shows one slice of the 3D texture, texel for texel.
precision highp float;
precision highp sampler3D;

layout(binding = 0) uniform sampler3D u_volume;
layout(location = 0) uniform ivec2 u_origin;
layout(location = 1) uniform int u_slice;

out vec4 frag_color;

void main()
{
    frag_color = texelFetch(u_volume, ivec3(ivec2(gl_FragCoord.xy) - u_origin, u_slice), 0);
}
