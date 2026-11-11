#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §8.8 lists texture()/textureProj()/textureLod() only, and §6.1 "All functions must be either declared with a prototype or defined with a body before they are called" - texture2D does not exist in 3.00
// @xfail: Mesa | Mesa 25.0 still declares the deprecated texture2D() family in #version 300 es shaders
// @feature: GLSL100.TEXTURE2D
precision mediump float;
uniform sampler2D u_tex;
out vec4 o;
void main() { o = texture2D(u_tex, vec2(0.5)); }
