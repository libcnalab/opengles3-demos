#version 300 es
// Vertex shader shared by every indirect draw of the demo.
//
//   location 0  a_pos    position in "shape space" (per vertex, divisor 0)
//   location 1  a_color  color of the instance (the divisor is VAO state: 1, or 2 in one VAO)
//
// The indirect commands decide WHICH vertices are fetched (first / firstIndex / baseVertex),
// HOW MANY (count) and HOW OFTEN (instanceCount). The uniforms below only place the result in the
// viewport, so the same command can be drawn into different places.
//
//   u_mode == 0  the position comes from the attribute array a_pos
//   u_mode == 1  the position is computed from gl_VertexID alone (a point on a circle), so the pixels
//                prove the exact vertex ID the GL delivers: first + i for DrawArraysIndirect and
//                baseVertex + index for DrawElementsIndirect (ES 3.2 section 10.5)
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_color;

uniform int u_mode;
uniform vec2 u_scale;   // shape space -> NDC of the viewport
uniform vec2 u_origin;  // NDC position of instance 0
uniform vec2 u_pitch;   // NDC distance between consecutive instances (gl_InstanceID)

flat out vec3 v_color;

void main()
{
    vec2 p = a_pos;
    if (u_mode == 1)
    {
        float angle = float(gl_VertexID) * 1.3;
        p = 0.8 * vec2(cos(angle), sin(angle));
    }
    gl_Position = vec4(p * u_scale + u_origin + float(gl_InstanceID) * u_pitch, 0.0, 1.0);
    v_color = a_color;
}
