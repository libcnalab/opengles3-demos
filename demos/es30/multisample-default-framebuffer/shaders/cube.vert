#version 300 es
// One flat color per cube face. Every vertex of a face carries that face's color, and the `flat`
// qualifier passes the provoking vertex's value through unchanged: no interpolation, so every sample
// a face covers receives exactly the face color (GLSL ES 3.00 4.3.9, ES 3.0 2.17).

layout(location = 0) in vec3 a_position;  // cube corner, object space, coordinates +-1
layout(location = 1) in vec4 a_color;     // the face color, k / 255 per channel

uniform mat4 u_mvp;                       // projection * view * model, computed on the CPU

flat out vec4 v_color;

void main()
{
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
