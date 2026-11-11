#version 300 es
// Ribbon of a TRIANGLE_STRIP drawn from an element array (primitive restart / glDrawRangeElements
// cells). The optional instancing parameters place copy number gl_InstanceID at u_step * i and fade
// its color by u_fade * i; for non-instanced draws u_step and u_fade are 0.
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_color;

uniform vec2 u_scale;
uniform vec2 u_origin;
uniform vec2 u_step;
uniform float u_fade;

out vec3 v_color;

void main()
{
    float i = float(gl_InstanceID);
    gl_Position = vec4(u_origin + a_pos * u_scale + u_step * i, 0.0, 1.0);
    v_color = a_color * (1.0 - u_fade * i);
}
