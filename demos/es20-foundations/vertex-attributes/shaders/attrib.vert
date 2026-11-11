#version 300 es
// Vertex shader used by the "format", "layout", "constant" and "client array" sections.
//
// Both inputs have an explicit location (GLSL ES 3.00 4.3.8.1), so the C++ code can talk about
// generic attribute slots 0 and 1 without asking the program. The second input is a vec4 on
// purpose: an array with fewer than four components per vertex still delivers a complete vec4,
// because the GL fills the missing components with (0, 0, 0, 1) (ES 3.0 2.9.1).
layout(location = 0) in vec2 a_position;  // clip-space x, y
layout(location = 1) in vec4 a_value;     // the attribute whose conversion is being examined

// out = a_value * u_gain + u_bias, per component. The gain/bias pair only maps the value that
// arrives in the shader into the displayable range [0, 1]; it never touches the conversion
// that the GL performed on the way in.
uniform vec4 u_gain;
uniform vec4 u_bias;

out vec4 v_color;

void main()
{
    v_color = a_value * u_gain + u_bias;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
