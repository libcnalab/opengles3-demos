#version 320 es
// A camera-facing quad (billboard) around u_center, built from gl_VertexID and the camera's right/up vectors.
uniform mat4 u_viewproj;
uniform vec3 u_center;
uniform vec3 u_right;
uniform vec3 u_up;
uniform float u_radius;

out vec2 v_uv;  // -1 .. +1 across the billboard

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1)) * 2.0 - 1.0;
    v_uv = corner;
    vec3 world = u_center + (u_right * corner.x + u_up * corner.y) * u_radius;
    gl_Position = u_viewproj * vec4(world, 1.0);
}
