#version 310 es
// Fills the viewport with one uniform color. Used in an ordinary (non-separable) program together with
// vs_fullscreen.vert to show that glUseProgram overrides a bound program pipeline.
precision highp float;

layout(location = 0) uniform vec3 u_color;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(u_color, 1.0);
}
