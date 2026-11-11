#version 300 es
// Pass 3: the skybox (es30-skybox-and-fog). A cube around the eye, rotated by the view matrix
// only (mat3 drops the translation), so the sky is infinitely far away. gl_Position.z = w puts
// every sky fragment at window depth 1.0; with DepthFunc(LEQUAL) the sky fills only the pixels
// that the main pass left at the cleared depth.
layout(location = 0) in vec3 a_position;

uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_dir;

void main()
{
    v_dir = a_position;
    vec4 clip = u_proj * vec4(mat3(u_view) * a_position, 1.0);
    gl_Position = clip.xyww;
}
