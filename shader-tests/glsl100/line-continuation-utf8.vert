#version 100
// @expect: compile-pass
// @spec: GLSL ES 3.00 §1.5 "support is mandated for both of these [line continuation, UTF-8 in comments] when a GLSL ES 1.00 shader is used with the OpenGL ES 3.0 API"
// @feature: GLSL100.VERSION
// UTF-8 in a comment: Příliš žluťoučký kůň úpěl ďábelské ódy - ∀x ∈ ℝ
#define SCALE(v) \
    ((v) * 2.0)
attribute vec4 a_position;
void main() { gl_Position = SCALE(a_position); }
