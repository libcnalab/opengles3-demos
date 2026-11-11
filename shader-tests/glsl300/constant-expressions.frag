#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.3.3 - literals, const variables, operators, constant array elements / struct fields / vector components, length(), constructors and built-in calls are constant expressions (checked with array sizes, which must be constant integral expressions > 0 - §4.1.9)
// @feature: GLSL300.QUAL.CONST
precision mediump float;
out vec4 o;
struct P { int a; float b; };
const P kP = P(2, 0.5);
const int kArr[3] = int[3](4, 5, 6);
const ivec3 kV = ivec3(7, 8, 9);
const float kRoot = sqrt(16.0);
const int kMax = max(3, kArr[1]);
float data[3];
void main()
{
    float a1[kArr[2] - kArr[0]];                 // 2: element of a constant array
    float a2[kP.a];                              // 2: field of a constant structure
    float a3[kV.y - kV.x];                       // 1: component of a constant vector
    float a4[data.length()];                     // 3: length() of a NON-constant array is constant
    float a5[kMax];                              // 5: built-in call with constant arguments
    float a6[int(kRoot)];                        // built-in sqrt in a constant initializer
    float a7[(kArr.length() > 2) ? 1 : -1];      // static assertion: a size <= 0 does not compile
    float a8[(abs(-3) == 3 && clamp(7, 0, 5) == 5 && min(2u, 9u) == 2u) ? 1 : -1];
    float a9[(2147483647 + 1 == int(0x80000000u)) ? 1 : -1];  // §4.1.3 wrap-around holds in folding too
    o = vec4(float(a1.length() + a2.length() + a3.length() + a4.length() + a5.length() + a6.length()
                   + a7.length() + a8.length() + a9.length()));
}
