#version 320 es
// Cell B: flat shading with the face normal, colour chosen by the built-in gl_PrimitiveID that the geometry
// shader wrote (GLSL ES 3.20 section 7.1.5).
precision highp float;

layout(location = 2) uniform vec3 u_eye;
layout(location = 3) uniform vec3 u_light;  // unit vector towards the light, in world space

in GS_OUT
{
    vec3 world;
    flat vec3 normal;
    flat int prim;
} fs_in;

out vec4 frag_color;

const vec3 kPalette[6] = vec3[6](vec3(1.00, 0.35, 0.30), vec3(1.00, 0.80, 0.25), vec3(0.45, 0.90, 0.40),
                                 vec3(0.30, 0.80, 0.95), vec3(0.50, 0.45, 1.00), vec3(0.95, 0.40, 0.85));

void main()
{
    vec3 albedo = kPalette[gl_PrimitiveID % 6];
    vec3 n = normalize(fs_in.normal);
    vec3 v = normalize(u_eye - fs_in.world);
    vec3 h = normalize(u_light + v);
    float diff = max(dot(n, u_light), 0.0);
    float spec = pow(max(dot(n, h), 0.0), 24.0);
    frag_color = vec4(albedo * (0.18 + 0.82 * diff) + vec3(0.30 * spec), 1.0);
}
