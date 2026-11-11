#version 300 es
// Float-attribute program used by every cell except the integer row.
//
//   location 0: vec2 position (NDC)
//   location 1: vec4 "attribute under test"
//
// The attribute is what the GL delivers after converting the stored format (normalized ubyte,
// snorm short, 2_10_10_10 packed, half float, ...). u_scale / u_bias map it to a displayable
// color: color = a_attr * u_scale + u_bias. Plain cells use scale 1, bias 0.
//
// When the attribute-1 array is *disabled*, a_attr is the "current generic attribute value"
// set by glVertexAttrib4f (the shader reads it exactly like array data).
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec4 a_attr;

uniform vec4 u_scale;
uniform vec4 u_bias;

out vec4 v_color;

void main()
{
    v_color = a_attr * u_scale + u_bias;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
