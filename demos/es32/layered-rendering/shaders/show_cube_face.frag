#version 320 es
// One face of the cube map in a 32x32 tile of the "cube net": every tile pixel looks up the centre of texel
// (2x, 2 * (31 - y)) of the face (NEAREST sampler), through a DIRECTION vector built from the face table of
// ES 3.2 Table 8.19 (the same table the hardware uses to map the direction back to a face and a texel).
// The row is flipped so that the world "up" of the four side faces points up in the picture.
precision highp float;
precision highp samplerCube;

layout(binding = 0) uniform samplerCube u_cube;
layout(location = 0) uniform ivec2 u_origin;
layout(location = 1) uniform int u_face;

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
    ivec2 p = ivec2(gl_FragCoord.xy) - u_origin;
    float s = (float(2 * p.x) + 0.5) / 64.0;
    float t = (float(2 * (31 - p.y)) + 0.5) / 64.0;
    frag_color = vec4(texture(u_cube, face_direction(u_face, s, t)).rgb, 1.0);
}
