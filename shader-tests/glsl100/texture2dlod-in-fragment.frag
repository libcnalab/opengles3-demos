#version 100
// @expect: compile-fail
// @spec: GLSL ES 1.00 §8.7 "The built-ins suffixed with "Lod" are allowed only in a vertex shader" (no #extension directive enables EXT_shader_texture_lod here)
// @feature: GLSL100.TEXTURE2D
precision mediump float;
uniform sampler2D u_tex;
void main() { gl_FragColor = texture2DLod(u_tex, vec2(0.5), 0.0); }
