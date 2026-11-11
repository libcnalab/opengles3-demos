#version 300 es
precision highp float;
// Two samplers, normally on the same unit. With u_split the right half of the cell shows the
// texture bound to a *different* texture unit (glActiveTexture + glUniform1i).
uniform sampler2D u_tex_a;
uniform sampler2D u_tex_b;
uniform bool u_split;
in vec2 v_uv;
out vec4 o_color;
void main()
{
    // Sample both unconditionally: implicit-LOD lookups inside non-uniform control flow have
    // undefined derivatives (GLSL ES 3.00 §8.8), so select the result afterwards.
    vec4 a = texture(u_tex_a, v_uv);
    vec4 b = texture(u_tex_b, v_uv);
    o_color = (u_split && v_uv.x >= 0.5) ? b : a;
}
