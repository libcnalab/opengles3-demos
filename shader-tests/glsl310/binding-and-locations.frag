#version 310 es
// @expect: compile-pass
// @spec: GLSL ES 3.10 §4.4.3 Uniform Variable Layout Qualifiers, §4.4.4 binding for opaque types
// @feature: GLSL310.LAYOUT.BINDING
// @feature: GLSL310.LAYOUT.UNIFORM_LOCATION
precision mediump float;
layout(binding = 3) uniform sampler2D u_tex;
layout(location = 7) uniform vec4 u_tint;
layout(std140, binding = 1) uniform Params { vec4 scale; };
out vec4 o;
void main() { o = texture(u_tex, vec2(0.5)) * u_tint * scale; }
