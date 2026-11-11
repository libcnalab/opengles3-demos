#version 320 es
// One vertex shader for every pass. The corners come from a uniform array, in pixels of the current target, so
// v_pos interpolates to the window-space position (x_w, y_w) of the fragment (ES 3.2 section 12.5.1).
//
// The same value is written to three outputs with different auxiliary storage qualifiers (GLSL ES 3.20 section
// 4.5). Each fragment shader declares the one it needs; unread outputs are harmless.
precision highp float;

uniform vec2 u_corner[4];   // TRIANGLE_STRIP quad (4 corners) or TRIANGLES (the first 3)
uniform vec2 u_size;        // width and height of the viewport in pixels

out highp vec2 v_pos;
centroid out highp vec2 v_cpos;
sample out highp vec2 v_spos;

void main()
{
    vec2 p = u_corner[gl_VertexID];
    v_pos = p;
    v_cpos = p;
    v_spos = p;
    gl_Position = vec4(p / u_size * 2.0 - 1.0, 0.0, 1.0);
}
