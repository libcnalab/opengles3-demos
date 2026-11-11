#version 320 es
// Layered rendering into a 2D multisample array texture (ES 3.2 section 9.8): the framebuffer has the
// whole array attached with glFramebufferTexture, so it is layered, and gl_Layer of each emitted
// primitive selects the layer its fragments are written to. The one input triangle (the full-viewport
// triangle) is emitted once per layer, so one draw call fills all four layers.
layout(triangles) in;
layout(triangle_strip, max_vertices = 12) out;  // 3 vertices x 4 layers (kLayers in main.cpp)

flat out highp int v_layer;

void main()
{
    for (int layer = 0; layer < 4; ++layer)
    {
        for (int i = 0; i < 3; ++i)
        {
            gl_Position = gl_in[i].gl_Position;
            gl_Layer = layer;  // every vertex of the primitive gets the same layer (section 9.8)
            v_layer = layer;
            EmitVertex();
        }
        EndPrimitive();
    }
}
