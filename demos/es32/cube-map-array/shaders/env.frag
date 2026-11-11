#version 320 es
precision highp float;
precision highp int;

// Environment mapping with a cube map array: the reflected view direction selects the face and texel, the instance
// number selects the layer (the fourth coordinate, an integer-valued float: GLSL ES 3.20 8.9.2 rounds it).
// Samplers other than sampler2D / samplerCube have no default precision, so the type is qualified here.
uniform highp samplerCubeArray u_env;
uniform vec3 u_eye;

in vec3 v_world;
in vec3 v_normal;
flat in int v_layer;

out vec4 o_color;

void main()
{
    vec3 n = normalize(v_normal);
    vec3 incident = normalize(v_world - u_eye);
    vec3 r = reflect(incident, n);
    // textureLod with level 0: no implicit derivatives, the level is exactly the base level.
    o_color = textureLod(u_env, vec4(r, float(v_layer)), 0.0);
}
