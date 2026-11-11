#version 310 es
// NEGATIVE: GLSL ES 3.10 section 4.4.1: with layout(early_fragment_tests) "it is an error to statically write
// to gl_FragDepth in the fragment shader". The demo expects this source to fail to compile or, linked with
// rect.vert, to link (GLSL ES 3.10 section 1.3: such errors may be reported at either stage).
precision highp float;

layout(early_fragment_tests) in;

out vec4 o_color;

void main()
{
    o_color = vec4(1.0);
    gl_FragDepth = 0.5;
}
