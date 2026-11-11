#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.4 / §6.1.1 in, out, inout, const in, unnamed parameters in a prototype, (void) parameter list
// @feature: GLSL300.PARAM_QUALIFIERS
precision mediump float;
out vec4 o;
void split(const in vec4 v, out vec2 lo, inout vec2 hi);
void split(const in vec4 v, out vec2 lo, inout vec2 hi) { lo = v.xy; hi += v.zw; }
float scale(float, float);
float scale(float a, float b) { return a * b; }
float one(void) { return 1.0; }
void main() { vec2 lo; vec2 hi = vec2(1.0); split(vec4(1.0, 2.0, 3.0, 4.0), lo, hi); o = vec4(lo, hi * scale(one(), 0.5)); }
