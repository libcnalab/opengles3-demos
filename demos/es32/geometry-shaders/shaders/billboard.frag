#version 320 es
// Cell A: a lit sphere impostor on the camera-facing quad (discard outside the unit circle).
precision highp float;

layout(location = 2) uniform vec3 u_light_eye;  // unit vector towards the light, in eye space

in GS_OUT
{
    vec3 eye;
    vec2 uv;
    flat vec3 color;
    flat int prim;
} fs_in;

out vec4 frag_color;

void main()
{
    float r2 = dot(fs_in.uv, fs_in.uv);
    if (r2 > 1.0) discard;
    vec3 n = vec3(fs_in.uv, sqrt(1.0 - r2));  // sphere normal in eye space (the quad faces the camera)
    float diff = max(dot(n, u_light_eye), 0.0);
    frag_color = vec4(fs_in.color * (0.25 + 0.75 * diff), 1.0);
}
