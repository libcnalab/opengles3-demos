#version 310 es
// INVALID: the array u_arr takes the consecutive locations 1, 2, 3 (GLSL ES 3.10 section 4.4.3: "Individual
// elements of a uniform array are assigned consecutive locations with the first element taking location
// location"), and u_x asks for location 3 again.
precision highp float;

layout(location = 1) uniform float u_arr[3];
layout(location = 3) uniform float u_x;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(u_arr[0] + u_arr[1] + u_arr[2] + u_x);
}
