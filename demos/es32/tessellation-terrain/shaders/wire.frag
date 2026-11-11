#version 320 es
precision highp float;

out vec4 o_color;

void main()
{
    o_color = vec4(0.02, 0.03, 0.06, 0.45);   // translucent: dense meshes on steep slopes stay readable
}
