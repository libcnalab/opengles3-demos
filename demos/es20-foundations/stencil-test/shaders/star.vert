#version 300 es
// A five-pointed star (pentagram) as a GL_TRIANGLE_FAN around its center. Vertex 0 is the center;
// vertices 1..6 trace the closed pentagram path P0 P2 P4 P1 P3 P0, where Pk lies at angle
// 90 degrees + 72 degrees * k on the circle of radius u_star.z. Draw with 7 vertices.
//
// Every fan triangle is counter-clockwise and spans 144 degrees, so a point is covered by as many
// triangles as the polygon's winding number there: 1 in the five arms, 2 in the inner pentagon.
//
//   u_star = (center.x, center.y, radius) in normalized device coordinates
uniform vec3 u_star;

void main()
{
    vec2 offset = vec2(0.0);
    if (gl_VertexID != 0)
    {
        int k = (2 * (gl_VertexID - 1)) % 5;                    // 0 2 4 1 3 0
        float angle = 1.57079632679 + 1.25663706144 * float(k); // 90 deg + 72 deg * k
        offset = vec2(cos(angle), sin(angle));
    }
    gl_Position = vec4(u_star.xy + u_star.z * offset, 0.0, 1.0);
}
