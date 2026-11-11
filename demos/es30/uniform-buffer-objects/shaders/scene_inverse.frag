#version 300 es
// Program 2: the SAME Palette block as program 1 (it can be backed by the same buffer object), but here
// declared WITHOUT an instance name, so the members are plain global names (color, gain, pick). The
// GL reports the members as "color[0]", "gain", ... without a block prefix. Only this stage uses the block.
// std140 makes the layout identical in both programs, whatever subset of members each one reads.
precision highp float;
precision highp int;

layout(std140) uniform Palette
{
    vec4  color[3];
    vec2  offset;
    float gain;
    int   pick;
};

in vec2 v_pos;
out vec4 o_color;

void main()
{
    o_color = vec4(vec3(1.0) - color[pick].rgb * gain, 1.0);   // the inverse of what program 1 draws
}
