#version 320 es
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
in VertexData { vec3 color; } v_in[];
out vec3 g_color;
void main()
{
    for (int i = 0; i < 4; ++i)
    {
        g_color = v_in[0].color;
        gl_Position = gl_in[0].gl_Position + vec4(float(i & 1), float(i >> 1), 0.0, 0.0) * 0.1;
        EmitVertex();
    }
    EndPrimitive();
}
