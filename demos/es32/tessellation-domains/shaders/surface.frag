#version 320 es
precision highp float;   // fragment shaders have no default float precision

uniform vec3 u_front;    // colour of a front-facing (counter-clockwise) triangle
uniform vec3 u_back;     // colour of a back-facing (clockwise) triangle
uniform float u_height;  // bump height, to normalise v_h
uniform int u_lines;     // 1: lines (isolines): colour by height only

in float v_h;
in vec3 v_n;
out vec4 o_color;

void main()
{
    // gl_FrontFacing tells the winding of the generated triangle: the colour shows what cw / ccw did.
    vec3 base = (u_lines != 0 || gl_FrontFacing) ? u_front : u_back;
    float shade = 0.60 + 0.40 * clamp(v_h / u_height, 0.0, 1.0);
    if (u_lines == 0)
    {
        vec3 light = normalize(vec3(-0.45, -0.35, 0.82));
        shade = 0.28 + 0.72 * max(dot(normalize(v_n), light), 0.0);
    }
    o_color = vec4(base * shade, 1.0);
}
