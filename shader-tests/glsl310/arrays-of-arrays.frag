#version 310 es
// @expect: compile-pass
// @spec: GLSL ES 3.10 §4.1.9 Arrays (arrays of arrays)
// @feature: GLSL310.ARRAYS_OF_ARRAYS
precision mediump float;
out vec4 o;
void main()
{
    float grid[2][3] = float[2][3](float[3](1.0, 2.0, 3.0), float[3](4.0, 5.0, 6.0));
    o = vec4(grid[1][2], float(grid.length()), float(grid[0].length()), 1.0);
}
