#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.2.2/§4.2.3 - nested scopes hide outer names, int x = x reads the outer x, S S = S(0), repeated prototypes and overloads are allowed
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
struct S { int x; };
int f(int k) { { int k = k + 3; return k; } }
void g(int);
void g(int);
void g(int v) { }
float g(float v) { return v; }
int x = 1;
void main()
{
    { int x = 2, y = x; g(y); }
    { int x = x; g(x); }
    { S S = S(0); g(S.x); }
    int i = 17;
    do int i = 4; while (i == 0);
    for (int j = 0; j < 2; ++j) { int k = j; g(k); }
    o = vec4(float(f(1)) + g(0.5));
}
