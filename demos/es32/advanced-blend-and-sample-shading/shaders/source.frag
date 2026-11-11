#version 320 es
// The source of the advanced-blend grid: a disc inside one cell, colored by a gradient and premultiplied by a
// non-trivial alpha (disc A: 0.9 .. 0.45 along y, disc B: 0.5 .. 0.9 along x).
//
// main.cpp replaces the marker line below with the layout qualifier under test (GLSL ES 3.20 section 4.4.2, "Fragment Outputs"):
// "layout(blend_support_multiply) out;" ... "layout(blend_support_all_equations) out;", or with nothing for the
// program that a draw with an advanced equation must reject (ES 3.2 section 15.1.5.3).
precision highp float;

//@BLEND_SUPPORT@

uniform vec2 u_origin;   // lower-left corner of the cell in pixels
uniform vec2 u_cell;     // cell size in pixels
uniform vec2 u_center;   // disc centre relative to the cell
uniform float u_radius;
uniform int u_disc;      // 0 = disc A, 1 = disc B

out vec4 o_color;

vec3 hsv(float h, float s, float v)
{
    vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), k, s);
}

void main()
{
    vec2 p = gl_FragCoord.xy - u_origin;
    if (length(p - u_center) > u_radius) discard;
    vec2 l = p / u_cell;
    vec3 base;
    float a;
    if (u_disc == 0)
    {
        base = hsv(fract(0.55 + 0.35 * l.y), 0.85, 0.3 + 0.7 * l.x);
        a = 0.9 - 0.45 * l.y;
    }
    else
    {
        base = hsv(fract(0.05 + 0.15 * l.x), 0.9, 1.0 - 0.6 * l.y);
        a = 0.5 + 0.4 * l.x;
    }
    o_color = vec4(base * a, a);
}
