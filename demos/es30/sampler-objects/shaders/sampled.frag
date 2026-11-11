#version 300 es
// Samples two sampler2D uniforms. Which texture and WHICH SAMPLING STATE each one sees is decided by
// the application: u_tex_a reads texture unit 0, u_tex_b reads texture unit 1, and each unit has a
// texture and (optionally) a sampler object bound. The shader does not know about sampler objects;
// the same GLSL sampler2D works with texture-object parameters or with a sampler object.
//
// With u_split the left half of the cell shows what unit 0 returns and the right half what unit 1
// returns; each half spans the whole [0,1] range of the cell, so two units bound to the SAME texture
// can be compared side by side.
precision highp float;

uniform sampler2D u_tex_a;
uniform sampler2D u_tex_b;
uniform vec2 u_uv_scale;   // texture coordinate = corner * scale + offset
uniform vec2 u_uv_offset;
uniform bool u_split;

in vec2 v_corner;
out vec4 o_color;

void main()
{
    vec2 c = v_corner;
    if (u_split) c.x = (c.x < 0.5) ? c.x * 2.0 : (c.x - 0.5) * 2.0;
    vec2 uv = c * u_uv_scale + u_uv_offset;

    // Sample both unconditionally (implicit-LOD lookups in non-uniform control flow have undefined
    // derivatives, GLSL ES 3.00 8.8) and select afterwards.
    vec4 a = texture(u_tex_a, uv);
    vec4 b = texture(u_tex_b, uv);
    o_color = (u_split && v_corner.x >= 0.5) ? b : a;
}
