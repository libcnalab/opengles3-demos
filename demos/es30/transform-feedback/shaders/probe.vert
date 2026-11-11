#version 300 es
// A shader whose outputs show what transform feedback records and in which order: integer outputs
// (gl_VertexID, gl_InstanceID), an array output and the built-in gl_Position. All are captured in
// INTERLEAVED_ATTRIBS mode in the order of the glTransformFeedbackVaryings list:
//   v_vertex (int, 4 B) | v_instance (int, 4 B) | v_trail[0] (vec2, 8 B) | v_trail[1] (vec2, 8 B) | gl_Position (vec4, 16 B)
layout(location = 0) in vec2 a_pos;

flat out int v_vertex;
flat out int v_instance;
out vec2 v_trail[2];

void main()
{
    v_vertex = gl_VertexID;
    v_instance = gl_InstanceID;
    v_trail[0] = a_pos;
    v_trail[1] = a_pos * 2.0;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
