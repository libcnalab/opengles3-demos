#version 300 es
// Fragment shaders have no default float precision in GLSL ES; one must be declared.
precision mediump float;

in vec3 v_color;

// GLSL ES 3.00 replaces gl_FragColor with a user-declared output. With a single output,
// location 0 is implied; it is written to draw buffer 0 (the default framebuffer's back buffer).
layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(v_color, 1.0);
}
