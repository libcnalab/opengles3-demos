#version 320 es
// The fragment shader reads gl_Layer (GLSL ES 3.20 section 7.1.5: "the fragment stage will read the same value
// written by the geometry stage") and stores a per-face tag in alpha: 40 * layer + 15, an exact 8-bit value.
precision highp float;

in GS_OUT
{
    flat vec3 color;
} fs_in;

out vec4 frag_color;

void main()
{
    frag_color = vec4(fs_in.color, float(gl_Layer * 40 + 15) / 255.0);
}
