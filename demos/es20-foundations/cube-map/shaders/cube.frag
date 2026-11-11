#version 300 es
precision highp float;
precision highp int;
// Samplers default to lowp in fragment shaders; ask for full precision.
precision highp samplerCube;

uniform samplerCube u_cube;
uniform int u_mode;  // 0: a point on the plane of face u_face, 1: skybox camera ray, 2: the direction u_dir
uniform int u_face;  // face index in the order of ES 3.0 Table 3.21: +X, -X, +Y, -Y, +Z, -Z
uniform vec3 u_dir;

in vec2 v_st;
in vec3 v_ray;
out vec4 o_color;

// The direction (r_x, r_y, r_z) whose cube map lookup lands on face `face` at the face
// coordinates (s, t) = st. It is Table 3.21 solved for r with |ma| = 1:
//   +X: sc = -rz, tc = -ry, ma = rx     -X: sc = +rz, tc = -ry, ma = rx
//   +Y: sc = +rx, tc = +rz, ma = ry     -Y: sc = +rx, tc = -rz, ma = ry
//   +Z: sc = +rx, tc = -ry, ma = rz     -Z: sc = -rx, tc = -ry, ma = rz
// with sc = 2s - 1 and tc = 2t - 1 (equations of section 3.8.9).
vec3 face_direction(int face, vec2 st)
{
    float sc = 2.0 * st.x - 1.0;
    float tc = 2.0 * st.y - 1.0;
    if (face == 0) return vec3(1.0, -tc, -sc);
    if (face == 1) return vec3(-1.0, -tc, sc);
    if (face == 2) return vec3(sc, 1.0, tc);
    if (face == 3) return vec3(sc, -1.0, -tc);
    if (face == 4) return vec3(sc, -tc, 1.0);
    return vec3(-sc, -tc, -1.0);
}

void main()
{
    // u_mode is a uniform, so these branches are uniform control flow. The lookup itself is
    // outside the branches because texture() uses implicit derivatives (GLSL ES 3.00 section 8.8).
    vec3 r;
    if (u_mode == 0) r = face_direction(u_face, v_st);
    else if (u_mode == 1) r = v_ray;
    else r = u_dir;
    o_color = texture(u_cube, r);  // (s, t, r) of the spec: the direction is NOT normalised
}
