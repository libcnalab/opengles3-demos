#version 300 es
// Every input is read and contributes to an output that the fragment shader consumes, so all of
// them are active attributes - except a_unused. The inputs cover the classes of types that
// glGetActiveAttrib can return for a GLSL ES 3.00 vertex input (booleans, arrays and structures
// are not allowed there, GLSL ES 3.00 4.3.4). The mat2 occupies TWO consecutive attribute slots.
in float a_weight;
in vec2  a_position;
in vec3  a_tint;
in vec4  a_value;
in mat2  a_basis;
in ivec2 a_cell;
in uint  a_id;
in vec3  a_unused;

out vec4 v_color;

void main()
{
    vec2 p = a_basis * a_position;
    v_color = vec4(a_tint, a_weight) * a_value + vec4(p, float(a_cell.x), float(a_id));
    gl_Position = vec4(p, 0.0, 1.0);
}
