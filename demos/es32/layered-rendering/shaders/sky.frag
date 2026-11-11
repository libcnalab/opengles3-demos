#version 320 es
// Skybox: the world-space view ray of every pixel looks up the environment cube map.
precision highp float;
precision highp samplerCube;

layout(binding = 0) uniform samplerCube u_cube;
layout(location = 0) uniform mat4 u_inv_viewproj;
layout(location = 1) uniform vec3 u_eye;

in vec2 v_ndc;
out vec4 frag_color;

void main()
{
    vec4 far_point = u_inv_viewproj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = far_point.xyz / far_point.w - u_eye;
    frag_color = vec4(texture(u_cube, dir).rgb, 1.0);
}
