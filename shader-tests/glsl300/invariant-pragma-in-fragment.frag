#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.6.1 "#pragma STDGL invariant(all) ... It is an error to use this pragma in a fragment shader"
// @feature: GLSL300.INVARIANT
#pragma STDGL invariant(all)
precision mediump float;
out vec4 o;
void main() { o = vec4(1.0); }
