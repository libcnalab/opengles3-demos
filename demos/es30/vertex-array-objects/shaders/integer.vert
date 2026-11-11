#version 300 es
// Integer-attribute program. ivec4 / uvec4 inputs must be fed by glVertexAttribIPointer or
// glVertexAttribI4* (ES 3.0 section 2.9: any other source leaves the values undefined). The
// values travel to the fragment shader unchanged as `flat` outputs: integer varyings cannot be
// interpolated, so GLSL ES 3.00 requires `flat`.
layout(location = 0) in vec2 a_pos;
layout(location = 1) in ivec4 a_i;
layout(location = 2) in uvec4 a_u;

flat out ivec4 v_i;
flat out uvec4 v_u;
out vec2 v_cell;  // 0..1 across the quad (the quad covers the whole cell)

void main()
{
    v_i = a_i;
    v_u = a_u;
    v_cell = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
