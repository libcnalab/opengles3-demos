#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §3.4 Preprocessor - GL_ES, __VERSION__, conditionals, #line, macros with arguments
// @feature: GLSL300.PREPROCESSOR
#if !defined(GL_ES) || GL_ES != 1 || __VERSION__ != 300
#error "GL_ES must be 1 and __VERSION__ must be 300"
#endif
#define SCALE(x) ((x) * 0.5)
#define TWO 2
#if TWO > 1 && defined(SCALE)
#  define PICK 1
#elif TWO == 1
#  error "unreachable"
#else
#  define PICK 0
#endif
#undef TWO
#ifdef TWO
#  error "TWO was undefined"
#endif
#line 100
#pragma optimize(on)
precision mediump float;
out vec4 o;
void main() { o = vec4(SCALE(float(PICK)), float(__LINE__ == 104), 0.0, 1.0); }
