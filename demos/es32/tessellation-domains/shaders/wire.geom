#version 320 es
// Geometry shader (ES 3.2 section 11.3): turns every tessellated triangle into a closed line strip. OpenGL ES has no
// polygon mode, so this is how the gallery draws the wireframe of the primitives the tessellator generated.
layout(triangles) in;
layout(line_strip, max_vertices = 4) out;

void main()
{
    for (int i = 0; i < 4; ++i)
    {
        gl_Position = gl_in[i % 3].gl_Position;
        gl_Position.z -= 0.004 * gl_Position.w;   // pull the lines slightly towards the camera (no polygon offset for lines)
        EmitVertex();
    }
    EndPrimitive();
}
