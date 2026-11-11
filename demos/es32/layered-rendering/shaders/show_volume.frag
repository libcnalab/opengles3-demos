#version 320 es
// Shows one slice of the 3D texture on a tile of any size. The third coordinate is the centre of slice k: (k + 0.5) / depth,
// where the depth comes from textureSize(); the sampler object is NEAREST.
precision highp float;
precision highp sampler3D;

layout(binding = 0) uniform sampler3D u_volume;
layout(location = 0) uniform int u_slice;

in vec2 v_ndc;
out vec4 frag_color;

void main()
{
    float w = (float(u_slice) + 0.5) / float(textureSize(u_volume, 0).z);
    frag_color = texture(u_volume, vec3(v_ndc * 0.5 + 0.5, w));
}
