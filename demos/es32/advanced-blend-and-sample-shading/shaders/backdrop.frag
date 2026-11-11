#version 320 es
// The destination of the advanced-blend grid. In every 64 x 40 cell: a hue ramp along x (hue 0 .. 0.8), a value
// ramp along y (0.5 .. 0.95) and an alpha ramp along x (0.95 .. 0.45). The color is written premultiplied by
// alpha, the form that ES 3.2 section 15.1.5.3 expects of a destination. main.cpp repeats this arithmetic in double
// precision, but the validation reads the stored backdrop back instead of trusting the rounding.
precision highp float;

uniform vec2 u_cell;   // cell size in pixels

out vec4 o_color;

vec3 hsv(float h, float s, float v)
{
    vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), k, s);
}

void main()
{
    vec2 l = mod(gl_FragCoord.xy, u_cell) / u_cell;
    vec3 base = hsv(0.8 * l.x, 0.75, 0.5 + 0.45 * l.y);
    float a = 0.95 - 0.5 * l.x;
    o_color = vec4(base * a, a);
}
