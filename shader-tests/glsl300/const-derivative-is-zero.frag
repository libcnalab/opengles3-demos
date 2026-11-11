#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.3.3 "The built-in functions dFdx, dFdy, and fwidth must return 0 when evaluated inside an initializer with an argument that is a constant expression"
// @feature: GLSL300.QUAL.CONST
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
