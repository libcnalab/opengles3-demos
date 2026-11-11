#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.9 "The logical binary operators and (&&), or ( | | ), and exclusive or (^^) operate only on two Boolean expressions"
// @feature: GLSL300.OPERATORS
precision mediump float;
out vec4 o;
void main() { int i = 1; bool b = i && true; o = vec4(float(b)); }
