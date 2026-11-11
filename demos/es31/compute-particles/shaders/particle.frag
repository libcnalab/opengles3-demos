#version 310 es
precision highp float;
precision highp int;  // u_flat is shared with the vertex shader, whose default int precision is highp

in vec2 v_corner;
in vec3 v_color;
uniform int u_flat;
layout(location = 0) out vec4 o_color;

void main()
{
    if (u_flat != 0)
    {
        o_color = vec4(v_color, 1.0);  // validation: the whole quad, exact colour
        return;
    }
    float r2 = dot(v_corner, v_corner);
    if (r2 > 1.0) discard;
    float a = 1.0 - r2;
    o_color = vec4(v_color * (a * a), 1.0);  // added to the framebuffer (ONE, ONE)
}
