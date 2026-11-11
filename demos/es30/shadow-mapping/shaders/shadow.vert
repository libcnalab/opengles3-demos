#version 300 es
// Pass 1 (depth only): the vertex stage is the light's view-projection times the model matrix.
// There are no varyings: the only product of this pass is the depth that the rasterizer writes.
layout(location = 0) in vec3 a_position;

uniform mat4 u_light_vp;  // orthographic projection * look-at of the directional light
uniform mat4 u_model;

void main()
{
    gl_Position = u_light_vp * (u_model * vec4(a_position, 1.0));
}
