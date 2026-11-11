#version 310 es
// A fragment shader that READS a shader storage buffer (GLSL ES 3.10 section 4.3.7): one cell per Probe member,
// coloured from the flags the compute shader wrote.
//   flag equals the expectation, flag = 1 : green    (the member read back correctly)
//   flag equals the expectation, flag = 0 : magenta  (a deliberately wrong layout broke this member, as predicted)
//   flag differs from the expectation     : red      (something is wrong)
precision highp float;
precision highp int;

layout(std430, binding = 0) readonly buffer Flags
{
    uint flags[];
} f;

uniform int  u_cell;         // cell width in pixels
uniform int  u_row_y;        // window y of the bottom pixel row of this grid row (its viewport)
uniform uint u_expect_mask;  // bit m set: member m is expected to verify

out vec4 o_color;

void main()
{
    // A one-pixel dark border on the left and bottom edge of every cell keeps neighbouring cells apart.
    if (int(gl_FragCoord.x) % u_cell == 0 || int(gl_FragCoord.y) == u_row_y)
    {
        o_color = vec4(0.06, 0.07, 0.10, 1.0);
        return;
    }
    int  m        = 1 + int(gl_FragCoord.x) / u_cell;       // member 1..16
    uint actual   = f.flags[m];
    uint expected = (u_expect_mask >> uint(m)) & 1u;
    if (actual != expected)       o_color = vec4(1.0, 0.0, 0.0, 1.0);
    else if (actual == 1u)        o_color = vec4(0.0, 1.0, 0.0, 1.0);
    else                          o_color = vec4(1.0, 0.0, 1.0, 1.0);
}
