#version 300 es
precision highp float;
precision highp int;

// The cell is split into 4 columns (vector component x, y, z, w) and 2 rows: the upper row tests
// the ivec4 input, the lower row the uvec4 input. A sub-area is green when the *exact* integer
// the shader received equals the value the CPU expects, red otherwise. Values above 2^24 cannot
// survive a trip through a 32-bit float, so a green sub-area proves the integer path.
uniform ivec4 u_expect_i;
uniform uvec4 u_expect_u;

flat in ivec4 v_i;
flat in uvec4 v_u;
in vec2 v_cell;

layout(location = 0) out vec4 o_color;

void main()
{
    int c = clamp(int(v_cell.x * 4.0), 0, 3);
    bool upper = v_cell.y >= 0.5;
    bool ok = upper ? (v_i[c] == u_expect_i[c]) : (v_u[c] == u_expect_u[c]);
    // Matching sub-areas are green, with a different shade per component (1.0, 0.8, 0.6, 0.4) and a
    // blue tint in the uvec4 row so that the eight sub-areas are distinguishable in the picture.
    float g = 1.0 - 0.2 * float(c);
    o_color = ok ? vec4(0.0, g, upper ? 0.0 : 0.4, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
