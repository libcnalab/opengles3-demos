#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.3.3 a constant expression is "a built-in function call whose arguments are all constant expressions, with the exception of the texture lookup functions. The built-in functions dFdx, dFdy, and fwidth must return 0 when evaluated inside an initializer with an argument that is a constant expression" (same text in 3.10 and 3.20)
// @feature: GLSL300.QUAL.CONST
// @xfail: ANGLE | ANGLE 2.1.17841 does not treat dFdx/dFdy/fwidth of a constant as a constant expression ("assigning non-constant to 'const mediump float'")
precision mediump float;
out vec4 o;
const float kDx = dFdx(3.0);
const float kDy = dFdy(2.0);
const float kW = fwidth(5.0);
void main()
{
    float assert_zero[(kDx == 0.0 && kDy == 0.0 && kW == 0.0) ? 1 : -1];
    o = vec4(float(assert_zero.length()));
}
