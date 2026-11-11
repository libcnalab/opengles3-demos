#version 310 es
// INVALID: two default-block uniforms with the same explicit location. GLSL ES 3.10 section 4.4.3: "No two
// default-block uniform variables in the program can have the same location, even if they are unused,
// otherwise a compiler or linker error will be generated."
precision highp float;

layout(location = 3) uniform float u_a;
layout(location = 3) uniform float u_b;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(u_a + u_b);
}
