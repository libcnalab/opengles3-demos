#version 300 es
// @expect: link-fail
// @spec: GLSL ES 3.00 §4.3.10 Linking of Vertex Outputs and Fragment Inputs - types must match
// @feature: GLSL300.INTERFACE_MATCHING
out vec3 v_value;
void main() { v_value = vec3(1.0); gl_Position = vec4(0.0); }
