#version 320 es
precision highp float;

in GS_OUT
{
    vec3 edge;
    float side;
} fs_in;

out vec4 frag_color;

void main()
{
    frag_color = vec4(1.0, 0.82, 0.20, 1.0);  // outline colour
}
