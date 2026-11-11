#version 320 es
// Mirror ball: the reflected view vector looks up the cube map that the geometry shader rendered; a little rim
// light is added at grazing angles.
precision highp float;
precision highp samplerCube;

layout(binding = 0) uniform samplerCube u_cube;
layout(location = 1) uniform vec3 u_eye;

in vec3 v_position;
out vec4 frag_color;

void main()
{
    vec3 n = normalize(v_position);
    vec3 v = normalize(u_eye - v_position);
    vec3 r = reflect(-v, n);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    vec3 env = texture(u_cube, r).rgb;
    frag_color = vec4(env * (1.0 - 0.12 * rim) + vec3(0.12 * rim), 1.0);
}
