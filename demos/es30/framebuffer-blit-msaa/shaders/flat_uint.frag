#version 300 es
// One unsigned integer value. Used for the triangle in the RGBA8UI multisample texture (ES 3.1+ contexts):
// integer outputs are written without conversion, so every covered sample holds exactly u_value.
precision highp float;
precision highp int;
uniform uvec4 u_value;
out uvec4 o_value;

void main()
{
    o_value = u_value;
}
