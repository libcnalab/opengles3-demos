#version 320 es
// One vertex shader for every draw of the demo. All meshes live in ONE vertex buffer; the
// *BaseVertex draw commands select a mesh by adding basevertex to every index (ES 3.2 section 10.5).
//
//   a_position, a_color  per-vertex data, fetched at element index = index + basevertex
//   a_index              the absolute position of this vertex in the shared buffer (stored on the CPU)
//   a_instance           per-instance data (divisor 1): xy = offset in NDC, z = colour tint. An instanced
//                        attribute is fetched at floor(instance / divisor) + baseinstance; basevertex does
//                        NOT apply to it, and baseinstance is always zero in OpenGL ES.
//
// u_mode 0: colour = a_color * tint.
// u_mode 1: vertex-ID check. ES 3.2 section 10.5 defines the vertex ID (gl_VertexID) of an element as
//           the index read from the element array PLUS basevertex, so it must equal a_index. Green when it
//           does, red when it does not.
layout(location = 0) in highp vec2 a_position;
layout(location = 1) in highp vec3 a_color;
layout(location = 2) in highp int a_index;
layout(location = 3) in highp vec3 a_instance;

uniform int u_mode;

out highp vec3 v_color;

void main()
{
    gl_Position = vec4(a_position + a_instance.xy, 0.0, 1.0);
    if (u_mode == 1)
        v_color = (gl_VertexID == a_index) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    else
        v_color = a_color * a_instance.z;
}
