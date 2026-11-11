#version 300 es
// Pass 1 (depth only, from the sun). The same vertex inputs and the same world transform as
// scene.vert, so instanced pillars and trees cast shadows exactly where the main pass draws them.
layout(location = 0) in vec3 a_position;
layout(location = 5) in vec4 a_instance;  // instance offset + scale, or the current value (0, 0, 0, 1)

uniform mat4 u_model;
uniform mat4 u_light_vp;

void main()
{
    vec3 world = a_instance.xyz + a_instance.w * (u_model * vec4(a_position, 1.0)).xyz;
    gl_Position = u_light_vp * vec4(world, 1.0);
}
