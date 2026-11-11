#version 300 es
// Program 3: an ARRAY of uniform blocks. tints[0] and tints[1] are two instances of the block Tint; the GL
// lists them as two separate blocks "Tint[0]" and "Tint[1]", each with its own binding point, while the member
// is listed once as "Tint.color". Instances are indexed with constant expressions only (GLSL ES 3.00 4.3.7).
precision highp float;

layout(std140) uniform Tint
{
    vec4 color;
} tints[2];

in vec2 v_pos;
out vec4 o_color;

void main()
{
    o_color = v_pos.x < 0.0 ? tints[0].color : tints[1].color;
}
