#version 320 es
// One face of the cube map on a tile of any size (the "cube net"): the position inside the tile, (s, t) = (u, 1 - v) with
// (u, v) = v_ndc / 2 + 1 / 2, becomes a DIRECTION through the face table of ES 3.2 Table 8.19 (the same table the hardware uses
// to map the direction back to a face and a texel); the sampler object is NEAREST. The row is flipped (t = 1 - v) so that the world "up"
// of the four side faces points up in the picture.
precision highp float;
precision highp samplerCube;

layout(binding = 0) uniform samplerCube u_cube;
layout(location = 0) uniform int u_face;

in vec2 v_ndc;
out vec4 frag_color;

vec3 face_direction(int face, float s, float t)
{
    float sc = 2.0 * s - 1.0;
    float tc = 2.0 * t - 1.0;
    if (face == 0) return vec3(1.0, -tc, -sc);   // +X
    if (face == 1) return vec3(-1.0, -tc, sc);   // -X
    if (face == 2) return vec3(sc, 1.0, tc);     // +Y
    if (face == 3) return vec3(sc, -1.0, -tc);   // -Y
    if (face == 4) return vec3(sc, -tc, 1.0);    // +Z
    return vec3(-sc, -tc, -1.0);                 // -Z
}

void main()
{
    vec2 uv = v_ndc * 0.5 + 0.5;
    frag_color = vec4(texture(u_cube, face_direction(u_face, uv.x, 1.0 - uv.y)).rgb, 1.0);
}
