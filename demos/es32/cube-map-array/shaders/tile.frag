#version 320 es
precision highp float;
precision highp int;

// One "layer-face tile": the 4 x 4 image of layer-face z = 6 * layer + face, looked up with the ordinary texture()
// function through a direction that lies on the plane of that face.
uniform highp samplerCubeArray u_env;
uniform int u_layer_face;  // z, as passed to glTexSubImage3D

in vec2 v_st;
out vec4 o_color;

// Table 8.20 solved for r with |ma| = 1 (face numbers are the layer numbers of Table 8.24):
//   +X: sc = -rz, tc = -ry   -X: sc = +rz, tc = -ry   +Y: sc = +rx, tc = +rz
//   -Y: sc = +rx, tc = -rz   +Z: sc = +rx, tc = -ry   -Z: sc = -rx, tc = -ry
// with sc = 2s - 1 and tc = 2t - 1 (section 8.13).
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
    int layer = u_layer_face / 6;
    int face = u_layer_face - 6 * layer;
    // texture(): the level of detail comes from the screen-space derivatives. A tile magnifies the 4 x 4 image
    // (a texel covers about 6 pixels), so the base level is used (lambda <= 0, 8.14).
    o_color = texture(u_env, vec4(face_direction(face, v_st), float(layer)));
}
