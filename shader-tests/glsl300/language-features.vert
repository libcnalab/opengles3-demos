#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.1 Basic Types, §5 Operators and Expressions, §6 Statements
// @feature: GLSL300.TYPES.SCALAR_VECTOR
// @feature: GLSL300.TYPES.MATRIX
// @feature: GLSL300.TYPES.STRUCT
// @feature: GLSL300.OPERATORS
// @feature: GLSL300.OPERATORS.BITWISE
// @feature: GLSL300.CONSTRUCTORS
// @feature: GLSL300.SWIZZLE
// @feature: GLSL300.MATRIX_OPS
// @feature: GLSL300.CONTROL_FLOW
// @feature: GLSL300.PARAM_QUALIFIERS
// @feature: GLSL300.QUAL.CONST
struct Light { vec3 dir; float power[2]; };
const float kHalf = sin(0.5235988) ;           // built-in calls are constant expressions in 3.00
const int kArr[3] = int[3](1, 2, 3);
void split(in vec4 v, out vec2 lo, inout vec2 hi) { lo = v.xy; hi += v.zw; }
uint mix_bits(uint a, int s) { return (a << s) ^ (a >> 3u) | (~a & 0xFFu) % 7u; }
void main()
{
    Light l = Light(vec3(0.0, 1.0, 0.0), float[2](1.0, 2.0));
    mat2x3 m23 = mat2x3(1.0);
    mat3x2 m32 = transpose(m23);
    mat4 m = mat4(1.0) * mat4(mat3(2.0));
    vec2 lo; vec2 hi = vec2(1.0);
    split(vec4(1.0, 2.0, 3.0, 4.0), lo, hi);
    int acc = 0;
    for (int i = 0; i < kArr.length(); ++i) { if (i == 1) continue; acc += kArr[i]; }
    switch (acc) { case 4: acc = 1; break; default: acc = 0; }
    int k = 0; do { k++; } while (k < 3);
    while (k > 0) { --k; }
    bvec2 b = lessThan(lo, hi);
    float s = any(b) ? l.power[1] : kHalf;
    vec4 sw = vec4(lo, hi).wzyx; sw.xy = sw.yx;
    gl_Position = m * vec4((m32 * vec3(s)).xy, float(mix_bits(3u, acc)), 1.0) + sw * determinant(inverse(mat2(2.0)));
}
