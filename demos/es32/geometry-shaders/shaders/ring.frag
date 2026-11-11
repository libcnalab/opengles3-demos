#version 320 es
precision highp float;

layout(location = 15) uniform vec3 u_light;  // unit vector towards the light, in world space

in GS_OUT
{
    vec3 world;
    flat vec3 normal;
    flat int inst;
    flat int face;
} fs_in;

out vec4 frag_color;

const vec3 kPalette[6] = vec3[6](vec3(1.00, 0.35, 0.30), vec3(1.00, 0.80, 0.25), vec3(0.45, 0.90, 0.40),
                                 vec3(0.30, 0.80, 0.95), vec3(0.50, 0.45, 1.00), vec3(0.95, 0.40, 0.85));

void main()
{
    // Copy colour from gl_InvocationID, per-face brightness from gl_PrimitiveIDIn (both passed as flat integers).
    vec3 albedo = kPalette[fs_in.inst] * (0.5 + 0.25 * float(fs_in.face % 3));
    float diff = max(dot(normalize(fs_in.normal), u_light), 0.0);
    frag_color = vec4(albedo * (0.25 + 0.75 * diff), 1.0);
}
