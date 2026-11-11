#version 300 es
// Pure integer attributes set with glVertexAttribIFormat (ES 3.2 section 10.3.1): the values reach the shader
// unconverted. The cell is three horizontal bands (two triangles = six vertices each); a band turns green only when
// the integer the shader received equals the one the CPU expects, otherwise red.
//
//   band 0  uvec2 a_cell   UNSIGNED_BYTE x 2, zero-extended
//   band 1  int   a_big    INT; 16777217 = 2^24 + 1 cannot be stored in a 32-bit float, so a conversion to float
//                          would round it to 16777216 and the band would turn red
//   band 2  int   a_neg    SHORT; -3 must be sign-extended (a zero-extended value would be 65533)
layout(location = 0) in vec2 a_pos;
layout(location = 3) in uvec2 a_cell;
layout(location = 4) in int a_big;
layout(location = 5) in int a_neg;

uniform uvec2 u_expect_cell;
uniform int u_expect_big;
uniform int u_expect_neg;

flat out vec4 v_color;

void main()
{
    int band = gl_VertexID / 6;
    bool ok = band == 0 ? a_cell == u_expect_cell : (band == 1 ? a_big == u_expect_big : a_neg == u_expect_neg);
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_color = ok ? vec4(0.0, 0.8, 0.2, 1.0) : vec4(0.9, 0.1, 0.1, 1.0);
}
