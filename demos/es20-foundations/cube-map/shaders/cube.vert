#version 300 es
// A quad covering the whole viewport (TRIANGLE_STRIP, 4 vertices, no vertex buffer). It provides
// two ways to produce a cube map direction in the fragment shader:
//   v_st  - face coordinates, (0,0) at the top-left corner of the viewport and t growing downwards,
//           used by the "face tile" mode to build a direction that lies on the plane of one face;
//   v_ray - a camera ray: forward + ndc.x * right + ndc.y * up, used by the skybox mode. The
//           vectors are uniforms, so a CPU model can evaluate exactly the same linear function.
uniform vec3 u_forward;
uniform vec3 u_right;  // already scaled by tan(fov / 2) * aspect
uniform vec3 u_up;     // already scaled by tan(fov / 2)
out vec2 v_st;
out vec3 v_ray;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    vec2 ndc = corner * 2.0 - 1.0;
    v_st = vec2(corner.x, 1.0 - corner.y);
    v_ray = u_forward + ndc.x * u_right + ndc.y * u_up;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
