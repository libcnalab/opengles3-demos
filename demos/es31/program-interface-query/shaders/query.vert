#version 310 es
// Vertex shader of the "graphics" program that the demo introspects with glGetProgramInterfaceiv and
// glGetProgramResource*. Every declaration below is a deliberate test subject; main.cpp keeps a table of
// what ES 3.2 section 7.3.1 says each query must return for it.
//
//   PROGRAM_INPUT                a_position (explicit location 0), a_color (explicit location 3),
//                                a_cell (no location: assigned by the linker), gl_VertexID (built-in)
//   UNIFORM                      u_tint (explicit location 4, also declared in the fragment shader) and
//                                the members of block Params
//   UNIFORM_BLOCK                Params (std140, layout(binding = 3))
//   TRANSFORM_FEEDBACK_VARYING   v_color, v_w (whole array), gl_Position - see glTransformFeedbackVaryings

layout(location = 0) in vec3 a_position;   // xy = position inside the cell, (0..1)^2; z unused (0)
layout(location = 3) in vec4 a_color;
in ivec2 a_cell;                            // x = cell number 0..7, y = a second integer

layout(location = 4) uniform vec4 u_tint;

// std140 layout (ES 3.2 section 7.6.2.2), offsets worked out by hand in main.cpp:
//   scale 0 | basis 16 (3 columns, stride 16) | skew 64 (row_major, 3 rows, stride 16) | k 112 (stride 16) | offset 144
layout(std140, binding = 3) uniform Params
{
    vec4 scale;
    mat3 basis;
    layout(row_major) mat2x3 skew;
    float k[2];
    vec3 offset;
} params;

out vec4 v_color;       // read by the fragment shader and captured by transform feedback
out float v_w[2];       // captured as a whole array
out vec2 v_uv;
out vec3 v_ubo;
flat out int v_cell;

void main()
{
    // Every member of Params contributes, so every member is active. The sum is what the "block Params"
    // cell of the picture shows; the CPU computes the same sum from the values it packed.
    v_ubo = params.scale.rgb + params.basis[1]
          + vec3(params.skew[0][1], params.skew[1][0], params.skew[1][2])
          + vec3(params.k[0], params.k[1], 0.0) + params.offset;

    v_color = a_color * vec4(u_tint.rgb, 1.0);
    v_w[0] = float(gl_VertexID);
    v_w[1] = float(a_cell.y);
    v_cell = a_cell.x;
    v_uv = a_position.xy;

    // Cell n of a 4 x 2 grid that covers the upper half of the window: column n & 3, row n >> 2.
    vec2 origin = vec2(float(a_cell.x & 3), float(a_cell.x >> 2));
    gl_Position = vec4(-1.0 + (origin.x + a_position.x) * 0.5, (origin.y + a_position.y) * 0.5, a_position.z, 1.0);
}
