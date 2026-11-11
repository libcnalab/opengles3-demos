#version 300 es
// Program 1, fragment stage. The block declaration must be identical to the one in scene.vert
// (same members, types, order and layout qualifier; ES 3.0 section 2.12.6.2), otherwise the link fails.
precision highp float;
precision highp int;

layout(std140) uniform Palette
{
    vec4  color[3];
    vec2  offset;
    float gain;
    int   pick;
} pal;

out vec4 o_color;

void main()
{
    // pal.pick selects one of the three colors with a dynamically uniform index.
    o_color = vec4(pal.color[pal.pick].rgb * pal.gain, 1.0);
}
