#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.6 "comments cannot be nested" - the first */ ends the comment and the rest is parsed as code
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
/* outer /* inner */ this text is outside every comment */
void main() { o = vec4(1.0); }
