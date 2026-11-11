#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §3.8 (3.10/3.20 §3.7) "all identifiers containing two consecutive underscores (__) are reserved for use by underlying software layers. Defining such a name in a shader does not itself result in an error"; GLSL ES 3.00 change log: "Double underscores are allowed in macro names and identifiers" (GLSL ES 1.00 reserved them as possible future keywords); the macro form is the separate case double-underscore-macro
// @feature: GLSL300.COMMENTS_TOKENS
// @xfail: ANGLE | ANGLE 2.1.17841 applies the GLSL ES 1.00 rule to 3.00 shaders: "identifiers containing two consecutive underscores (__) are reserved as possible future keywords" is an error
precision mediump float;
out vec4 o;
float my__value = 0.25;
void main() { o = vec4(my__value); }
