#version 300 es
// A circle as a GL_TRIANGLE_FAN: vertex 0 is the center, vertices 1..kSegments+1 walk around the
// rim counter-clockwise (the last one repeats the first). Draw with kSegments + 2 vertices.
//
//   u_circle = (center.x, center.y, radius) in normalized device coordinates
const int kSegments = 48;
uniform vec3 u_circle;

void main()
{
    float angle = 6.28318530718 * float(gl_VertexID - 1) / float(kSegments);
    vec2 offset = (gl_VertexID == 0) ? vec2(0.0) : vec2(cos(angle), sin(angle));
    gl_Position = vec4(u_circle.xy + u_circle.z * offset, 0.0, 1.0);
}
